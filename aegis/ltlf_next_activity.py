"""
Next-activity prediction with the models trained for repair.

For a test trace of length n, the prefix of length k (k = 1..n-1) is encoded as a repair graph with k visible events
plus one event after them that is MISSING (Activity one-hot at MISSING_ID, timestamp -1, masked for every attribute),
exactly like a masked event in the repair task. The network "repairs" that node: its prediction is the next activity.

Hard constraints on a prefix: the trace continues after the predicted event, so a rule must not be required to ACCEPT
at the end of prefix + a (that is what ltlf_decode.HardDecoder checks for complete traces). Instead, a candidate a is
allowed iff, for every rule still satisfiable after the prefix, the DFA state after prefix + a is LIVE: acceptance is
reachable by some continuation. Rules such as end(x) or response(x, y) never prune; chain/precedence/factor rules can.
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from .ltlf_decode import HardDecoder

ACT_KEY, TS_KEY = "Activity", "time:timestamp"


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------
def build_graph(g, length, missing, n_classes, missing_id, act_key=ACT_KEY, ts_key=TS_KEY):
    """Graph of the first `length` events of the (complete, directed, untransformed) graph g, built from its labels g.y.
    Events in `missing` (indices < length) are encoded as missing for every attribute, the others are visible.
    Edges: those of g with both ends < length (nodes are stored in temporal order)."""
    assert set(g.node_types) == {act_key, ts_key}, f"unexpected node types {g.node_types}"
    assert g[act_key].x.shape[1] == n_classes, "Activity features wider than the one-hot (not an AT_only graph?)"
    y_act, y_ts = g.y[act_key][:length], g.y[ts_key][:length]
    miss = torch.zeros(length, dtype=torch.bool)
    miss[list(missing)] = True

    h = HeteroData()
    cls = torch.where(miss, torch.full_like(y_act, missing_id), y_act)
    h[act_key].x = torch.nn.functional.one_hot(cls, n_classes).to(g[act_key].x.dtype)
    h[ts_key].x = torch.where(miss, torch.full_like(y_ts, -1.0), y_ts).to(g[ts_key].x.dtype)[:, None]
    for et in g.edge_types:
        ei = g[et].edge_index
        h[et].edge_index = ei[:, (ei < length).all(0)].clone()
    h.y = {act_key: y_act.clone(), ts_key: y_ts.clone()}
    h.masks = {k: miss.clone() for k in g.masks}
    return h


def prefix_graphs(graphs, lut, n_classes, missing_id, transform=None, act_key=ACT_KEY, ts_key=TS_KEY):
    """All prefixes of all traces. graphs: one complete-labelled test graph per trace (e.g. the 'even' test file).
    Returns (prefix graphs, meta): meta[i] = dict(trace_id, k, n, prefix, truth) with prefix/truth in compact ids."""
    out, meta = [], []
    for tid, g in enumerate(graphs):
        n = int(g.y[act_key].shape[0])
        seq = lut[g.y[act_key]].tolist()
        assert min(seq) >= 0, "labels contain the MISSING class"
        for k in range(1, n):
            h = build_graph(g, k + 1, [k], n_classes, missing_id, act_key, ts_key)
            out.append(transform(h) if transform is not None else h)
            meta.append(dict(trace_id=tid, k=k, n=n, prefix=seq[:k], truth=seq[k]))
    return out, meta


# ---------------------------------------------------------------------------
# Hard constraints on prefixes
# ---------------------------------------------------------------------------
class PrefixHardDecoder(HardDecoder):
    """HardDecoder whose last position is not the end of the trace: B[n] = live states instead of accepting states."""

    def __init__(self, dfas, A, rule_idx=None):
        super().__init__(dfas, A, rule_idx)
        Rr = np.arange(self.R)
        live = self.acc.copy()
        while True:
            new = live | live[Rr[:, None, None], self.delta].any(2)
            if (new == live).all():
                break
            live = new
        self.live = live                                          # [R, Q]

    def _backward(self, acts):
        n, A, R, Rr = len(acts), self.A, self.R, np.arange(self.R)
        B = np.zeros((n + 1, R, self.Q), bool)
        B[n] = self.live
        for t in range(n - 1, -1, -1):
            if acts[t] == A:
                B[t] = B[t + 1][Rr[:, None, None], self.delta].any(2)
            else:
                B[t] = B[t + 1][Rr[:, None], self.delta[:, :, acts[t]]]
        return B

    def allowed(self, prefix):
        """Bool [A]: next activities that keep every rule still satisfiable after `prefix` satisfiable."""
        Rr = np.arange(self.R)
        s = np.zeros(self.R, np.int64)
        for a in prefix:
            s = self.delta[Rr, s, a]
        active = self.live[Rr, s]
        nxt = self.delta[Rr, s]                                   # [R, A]
        ok = self.live[Rr[:, None], nxt] | ~active[:, None]
        return ok.all(0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_next(net, graphs, meta, decoder, class_ids, keys, device, mask_to_idx, DataLoader,
                  beam=16, batch_size=256, act_key=ACT_KEY):
    """One row per prefix: truth, raw argmax, hard-constrained prediction (compact ids), fallback flag, number of
    activities the rules allow, and whether the raw prediction breaks a rule that was still satisfiable."""
    A = len(class_ids)
    cid = torch.as_tensor(class_ids, device=device)
    rows, i0 = [], 0
    net.eval()
    with torch.no_grad():
        for x in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            idx = mask_to_idx(x, keys)                            # CPU, before x.to(device) moves x in place
            assert idx[act_key].numel() == x.num_graphs, "every prefix graph must have exactly one missing Activity"
            out = net(x.to(device), {k: v.to(device) for k, v in idx.items()})[act_key]
            logp = out.float().index_select(1, cid).log_softmax(-1).cpu().numpy()
            for j in range(x.num_graphs):
                m = meta[i0 + j]
                raw = int(logp[j].argmax())
                hard = decoder.decode(logp[j][None], m["prefix"] + [A], beam)
                ok = decoder.allowed(m["prefix"])
                rows.append(dict(trace_id=m["trace_id"], k=m["k"], n=m["n"], truth=m["truth"], pred_raw=raw,
                                 pred_hard=raw if hard is None else hard[0], fallback=hard is None,
                                 n_allowed=int(ok.sum()), truth_allowed=bool(ok[m["truth"]]),
                                 raw_violates=not ok[raw]))
            i0 += x.num_graphs
    assert i0 == len(meta)
    df = pd.DataFrame(rows)
    df["ok_raw"] = df.pred_raw == df.truth
    df["ok_hard"] = df.pred_hard == df.truth
    return df


def summarize_run(df):
    """One run: accuracies, fallback rate, share of raw predictions breaking a rule, share of prefixes whose true
    next activity the rules allow (upper bound of hard), mean number of allowed activities."""
    return dict(acc_raw=df.ok_raw.mean(), acc_hard=df.ok_hard.mean(), n_prefixes=len(df),
                fallback=df.fallback.mean(), raw_violation_rate=df.raw_violates.mean(),
                truth_allowed=df.truth_allowed.mean(), mean_allowed=df.n_allowed.mean())
