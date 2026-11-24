"""
Heterogeneous GNN for event-log repair (Dissegna et al., "Graph-based Event Log Repair", arXiv:2508.05145),
trained with GradNorm over the per-attribute losses plus the LTLf semantic loss (outside GradNorm).
"""
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import Linear, Module, ModuleDict, ModuleList
from torch.nn.functional import l1_loss
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, SAGEConv
from torcheval.metrics.functional import multiclass_accuracy
from tqdm.auto import tqdm


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class HGNN(Module):
    def __init__(self, output_cat, output_real, nodes_relations, parameters):
        super().__init__()
        hid = parameters["hid"]
        layers = parameters["layers"]
        aggregation = parameters["aggregation"]

        self.output_cat = output_cat
        self.output_real = output_real

        self.convs = ModuleList()
        for _ in range(layers):
            self.convs.append(
                HeteroConv(
                    {
                        rel: SAGEConv((-1, -1), out_channels=hid, aggr=aggregation, normalize=True)
                        for rel in nodes_relations
                    },
                    aggr=aggregation,
                )
            )

        self.FC = ModuleDict()
        for k in output_cat:
            self.FC[k] = Linear(hid, output_cat[k])
        for k in output_real:
            self.FC[k] = Linear(hid, 1)

    def forward(self, batch, idx):
        """idx: dict {task: LongTensor of node indices to predict}."""
        x_dict = batch.x_dict  # local copy: batch is not mutated
        for conv in self.convs:
            x_dict = {k: v.relu() for k, v in conv(x_dict, batch.edge_index_dict).items()}

        out = {}
        for k in self.output_cat:
            out[k] = self.FC[k](x_dict[k].index_select(0, idx[k]))
        for k in self.output_real:
            out[k] = self.FC[k](x_dict[k].index_select(0, idx[k])).squeeze(-1)
        return out


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def mask_to_idx(batch, keys):
    """Indices of the masked nodes per task. Run on the CPU batch BEFORE .to(device) (PyG moves batches in place)."""
    return {k: batch.masks[k].nonzero().squeeze(1) for k in keys}


def grad_norm(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    sq = [t.pow(2).sum() for t in g if t is not None]
    return torch.stack(sq).sum().sqrt() if sq else loss.new_zeros(())


def build_net(config, data, device, sample=None):
    """HGNN with its lazy (-1, -1) layers initialised by one forward pass (needed before load_state_dict)."""
    net = HGNN(parameters=config, output_cat=data.outputcat, output_real=data.outputreal,
               nodes_relations=data.edge_types).to(device)
    sample = sample if sample is not None else data.X_train[:8]
    with torch.no_grad():
        x0 = next(iter(DataLoader(sample, batch_size=8)))
        i0 = mask_to_idx(x0, data.keys)
        net(x0.to(device), {k: v.to(device) for k, v in i0.items()})
    return net


def load_net(path, config, data, device):
    net = build_net(config, data, device)
    net.load_state_dict(torch.load(Path(path), map_location=device))
    net.eval()
    return net


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train_hgnn(config, data, device, sem_loss=None, epochs=20, callback=None, patience=10):
    """Train on data.X_train, early-stop on the validation loss over data.X_valid.

    config keys: hid, layers, aggregation, lr, weight_decay, batch_size, lr_gradNorm, alpha, gn_every,
    amp (bf16 autocast, CUDA only), num_workers, use_gradnorm (default True), lambda_sem (weight of the semantic loss)."""
    print(config)
    output_cat, output_real = data.outputcat, data.outputreal

    use_amp = config.get("amp", True) and device.type == "cuda"
    gn_every = config.get("gn_every", 1)                 # GradNorm update every k steps
    use_gn = config.get("use_gradnorm", True)            # False = plain sum of the task losses
    num_workers = config.get("num_workers", 4)
    lam = config.get("lambda_sem", 0.0)                  # weight of the LTLf semantic loss (0 = no rules in training)
    assert lam == 0 or sem_loss is not None, "lambda_sem > 0 needs a semantic loss"
    amp_kwargs = dict(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)

    net = HGNN(parameters=config, output_cat=output_cat, output_real=output_real,
               nodes_relations=data.edge_types).to(device)

    task_keys = list(output_cat) + list(output_real)
    n_tasks = len(task_keys)
    criteria = {k: nn.CrossEntropyLoss() for k in output_cat}
    criteria.update({k: nn.L1Loss() for k in output_real})

    loader_kw = dict(batch_size=config["batch_size"], num_workers=num_workers, pin_memory=True,
                     persistent_workers=num_workers > 0)
    train_loader = DataLoader(data.X_train, shuffle=True, **loader_kw)
    valid_loader = DataLoader(data.X_valid, shuffle=False, **loader_kw)

    def prepare(x):
        idx = mask_to_idx(x, task_keys)  # CPU
        x = x.to(device, non_blocking=True)
        idx = {k: v.to(device, non_blocking=True) for k, v in idx.items()}
        labels = {k: x.y[k].index_select(0, idx[k]) for k in task_keys}
        return x, idx, labels

    def task_losses(x, idx, labels):
        with torch.autocast(**amp_kwargs):
            out = net(x, idx)
        return torch.stack([criteria[k](out[k].float(), labels[k]) for k in task_keys]), out

    # Dry run so lazy (-1,-1) layers are initialised before optimizer / param lists
    with torch.no_grad():
        x0, i0, _ = prepare(next(iter(train_loader)))
        net(x0, i0)
        del x0, i0

    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    shared = list(net.convs[-1].parameters())  # last shared layer for GradNorm

    # GradNorm state
    weights = torch.ones(n_tasks, device=device, requires_grad=True)
    w_opt = torch.optim.Adam([weights], lr=config["lr_gradNorm"])
    L0 = None
    step = 0

    best_state, best_loss, pat_count = None, float("inf"), 0

    for epoch in tqdm(range(epochs)):
        net.train()
        for x in train_loader:
            x, idx, labels = prepare(x)
            losses_values, out = task_losses(x, idx, labels)

            if L0 is None:
                L0 = losses_values.detach()

            do_gn = use_gn and step % gn_every == 0
            if do_gn:
                # per-task gradient norms on the last shared layer (no double backward)
                norms = torch.stack([grad_norm(losses_values[i], shared) for i in range(n_tasks)])
                gw = weights * norms  # differentiable wrt weights only
                loss_ratio = losses_values.detach() / L0
                rt = loss_ratio / loss_ratio.mean()
                target = (gw.mean() * rt ** config["alpha"]).detach()
                gn_loss = (gw - target).abs().sum()

                w_opt.zero_grad(set_to_none=True)
                gn_loss.backward()  # touches only `weights`

            optimizer.zero_grad(set_to_none=True)
            l_sem = sem_loss(x, idx, out["Activity"]) if lam > 0 else 0.0   # semantic loss, outside GradNorm
            (weights.detach() @ losses_values + lam * l_sem).backward()
            optimizer.step()

            if do_gn:
                w_opt.step()
                with torch.no_grad():
                    weights.mul_(n_tasks / weights.sum())  # renormalise in place
            step += 1

        # ---- validation: sum of the raw task losses ----
        net.eval()
        total = torch.zeros((), device=device)
        n_batches = 0
        with torch.no_grad():
            for x in valid_loader:
                x, idx, labels = prepare(x)
                l_v, _ = task_losses(x, idx, labels)
                total += l_v.sum()
                n_batches += 1
        val_loss = (total / n_batches).item()

        print(f"Epoch {epoch}: val loss [{val_loss:.5f}] best loss [{best_loss:.5f}]")

        if callback is not None:
            callback(epoch, val_loss)  # may raise optuna.TrialPruned

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            pat_count = 0
        else:
            pat_count += 1
            if pat_count >= patience:
                break

    net.load_state_dict(best_state)
    net.best_val_loss = best_loss
    return net


# ----------------------------------------------------------------------------
# Evaluation (argmax accuracy / MAE)
# ----------------------------------------------------------------------------
def _indices(x, keys, mode):
    """Node indices to evaluate per task. Runs on the CPU batch (no GPU sync)."""
    if mode == "full":  # every node of that type
        return {k: torch.arange(x.y[k].shape[0]) for k in keys}
    if mode == "correct":  # nodes NOT masked
        return {k: (~x.masks[k]).nonzero().squeeze(1) for k in keys}
    if mode == "missing":  # masked nodes (same as training/validation)
        return {k: x.masks[k].nonzero().squeeze(1) for k in keys}
    raise ValueError(f"unknown mode {mode!r}")


@torch.no_grad()
def evaluate_hgnn(net, output_cat, output_real, test_graphs, device, mode="missing", batch_size=128, num_workers=4):
    """Activity accuracy and timestamp MAE of the plain network output on the `mode` nodes."""
    keys = list(output_cat) + list(output_real)
    criteria = {k: nn.CrossEntropyLoss() for k in output_cat}
    criteria.update({k: nn.L1Loss() for k in output_real})

    loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=True, persistent_workers=num_workers > 0)

    pred_cat = {k: [] for k in output_cat}
    tgt_cat = {k: [] for k in output_cat}
    pred_real = {k: [] for k in output_real}
    tgt_real = {k: [] for k in output_real}
    batch_losses = []

    net.eval()
    for x in loader:
        idx = _indices(x, keys, mode)  # CPU
        x = x.to(device, non_blocking=True)
        idx = {k: v.to(device, non_blocking=True) for k, v in idx.items()}

        labels = {k: x.y[k].index_select(0, idx[k]) for k in keys}
        out = net(x, idx)

        batch_losses.append(torch.stack([criteria[k](out[k], labels[k]) for k in keys]))

        for k in output_cat:
            pred_cat[k].append(out[k].argmax(dim=1))
            tgt_cat[k].append(labels[k])
        for k in output_real:
            pred_real[k].append(out[k])
            tgt_real[k].append(labels[k])

    L = torch.stack(batch_losses).cpu()  # [n_batches, n_tasks]
    per_task = dict(zip(keys, L.mean(0).tolist()))  # mean of per-batch losses
    avg_total = L.sum(1).mean().item()

    accuracy = {
        k: multiclass_accuracy(torch.cat(pred_cat[k]), torch.cat(tgt_cat[k]), num_classes=output_cat[k]).item()
        for k in output_cat
    }
    mae_global = {k: l1_loss(torch.cat(pred_real[k]), torch.cat(tgt_real[k])).item() for k in output_real}

    res = (
        {f"{k}_acc": accuracy[k] for k in output_cat}
        | {f"{k}_mae": per_task[k] for k in output_real}
        | {"AVG_total_loss": avg_total}
        | {f"MAE_{k}": mae_global[k] for k in output_real}
    )
    print(res)
    return res
