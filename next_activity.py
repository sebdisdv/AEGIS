#!/usr/bin/env python3
"""
Next-activity prediction with the repair models (no retraining).

The networks trained for repair with the semantic loss (trained_model/<dataset>_AT_only/<tag>/run{i}.pt) are evaluated on
next-activity prediction:
  * complete test traces come from the labels of one repair test file (_TEST_V2_repair_even.pkl; all test types are the
    same traces with a different masking);
  * a trace of length n gives n-1 prefixes; prefix k is a repair graph with k visible events plus one event after them
    encoded as missing. The network repairs that node: its prediction is the next activity;
  * raw:  argmax of the network output on that node;
  * hard: the same output under the LTLf hard constraints for prefixes (aegis.ltlf_next_activity.PrefixHardDecoder):
    a candidate is allowed iff every rule still satisfiable after the prefix stays satisfiable by SOME continuation
    (live DFA state). Rules: mined as in main.py, train support >= --sup-min.

Outputs: <out_dir>/<dataset>_AT_only/next_act_{predictions,RESULTS,MEAN_STD,by_prefix_len}_<tag>.csv and
<out_dir>/summary_<tag>.csv (rows of other datasets already in the file are kept).

Usage:
    python next_activity.py                                  # all datasets
    python next_activity.py --datasets bpi_2013_CZ --check   # one dataset, with sanity checks first
"""
import argparse
import itertools
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from aegis import ltlf_next_activity as NA
from aegis import ltlf_patterns as LP
from aegis.data import DATASETS, ROOT, full_name, infer_missing_id, load_graphs, seed_everything, transform
from aegis.ltlf_features import DFA
from aegis.model import HGNN, mask_to_idx

ACT_KEY, TS_KEY = NA.ACT_KEY, NA.TS_KEY
TRACE_SOURCE = "even"    # test file the complete traces are read from


def setup_dataset(name, top_k=60):
    ds = full_name(name)
    X_train = load_graphs(f"{ds}_TRAIN_V2_repair", undirected=False)
    X_src = load_graphs(f"{ds}_TEST_V2_repair_{TRACE_SOURCE}", undirected=False)
    n_classes = X_train[0][ACT_KEY].x.shape[1]
    missing_id = infer_missing_id(X_train, n_classes)
    class_ids = [c for c in range(n_classes) if c != missing_id]
    A = len(class_ids)
    lut = torch.full((n_classes,), -1, dtype=torch.long)
    lut[class_ids] = torch.arange(A)

    full_seqs = [lut[g.y[ACT_KEY]].tolist() for g in X_train]
    rules = LP.mine_rich_rules(full_seqs, A, families=LP.FAMILIES, top_k=top_k, verbose=False)
    dfas = [d for _, d, _ in rules]
    cnt = Counter(tuple(s) for s in full_seqs)
    train_sup = np.array([sum(c for s, c in cnt.items() if d.accepts(s)) / len(full_seqs) for d in dfas])
    print(f"{ds}: {len(X_src)} test traces, {n_classes} classes, MISSING_ID={missing_id}, {len(rules)} rules mined")
    return dict(ds=ds, X_src=X_src, n_classes=n_classes, missing_id=missing_id, class_ids=class_ids, A=A, lut=lut,
                rules=rules, dfas=dfas, train_sup=train_sup)


def sanity_checks(S, n_random=500):
    """(a) build_graph from the labels and the original mask reproduces every original test graph;
    (b) PrefixHardDecoder.allowed agrees with brute force on random DFAs."""
    for g in S["X_src"]:
        m = g.masks[ACT_KEY].bool()
        assert (g.masks[TS_KEY].bool() == m).all()
        h = NA.build_graph(g, g.y[ACT_KEY].shape[0], m.nonzero().squeeze(1).tolist(), S["n_classes"], S["missing_id"])
        assert torch.equal(h[ACT_KEY].x, g[ACT_KEY].x.detach())
        assert torch.allclose(h[TS_KEY].x, g[TS_KEY].x.detach(), atol=1e-6)
        for et in g.edge_types:
            assert torch.equal(h[et].edge_index, g[et].edge_index)
    print("(a) graph encoding reproduces the original test graphs")

    rng = np.random.default_rng(1)
    bad = 0
    for _ in range(n_random):
        A = int(rng.integers(2, 5)); R = int(rng.integers(1, 4))
        dfas_ = []
        for _ in range(R):
            Q = int(rng.integers(1, 5))
            dfas_.append(DFA(rng.integers(0, Q, (Q, A)), rng.random(Q) < 0.4))
        pre = [int(a) for a in rng.integers(0, A, int(rng.integers(1, 5)))]
        conts = [list(c) for l in range(7) for c in itertools.product(range(A), repeat=l)]
        sat = lambda d, s: any(d.accepts(s + c) for c in conts)
        active = [sat(d, pre) for d in dfas_]
        bf = np.array([all(sat(d, pre + [a]) for d, act in zip(dfas_, active) if act) for a in range(A)])
        bad += not np.array_equal(NA.PrefixHardDecoder(dfas_, A).allowed(pre), bf)
    assert bad == 0
    print("(b) PrefixHardDecoder.allowed matches brute force")


def make_net(cfg, n_classes, edge_types, sample, device):
    net = HGNN(parameters=cfg, output_cat={ACT_KEY: n_classes}, output_real=[TS_KEY],
               nodes_relations=edge_types).to(device)
    with torch.no_grad():                         # initialise the lazy (-1, -1) layers before load_state_dict
        x0 = next(iter(DataLoader(sample, batch_size=8)))
        i0 = mask_to_idx(x0, [ACT_KEY, TS_KEY])
        net(x0.to(device), {k: v.to(device) for k, v in i0.items()})
    return net


def run_dataset(S, a, device):
    ds = S["ds"]
    ridx = [k for k, s in enumerate(S["train_sup"]) if s >= a.sup_min]
    decoder = NA.PrefixHardDecoder(S["dfas"], S["A"], rule_idx=ridx)
    graphs, meta = NA.prefix_graphs(S["X_src"], S["lut"], S["n_classes"], S["missing_id"], transform)
    edge_types = sorted({et for g in graphs for et in g.edge_types})
    print(f"{ds}: {len(meta)} prefixes from {len(S['X_src'])} traces, {len(ridx)} rules enforced (support >= {a.sup_min})")

    model_dir = ROOT / "trained_model" / ds / a.tag
    n_runs = a.runs or len(list(model_dir.glob("run*_config.json")))
    assert n_runs > 0, f"no saved models in {model_dir}"
    preds, per_run = [], []
    for run in range(n_runs):
        base = model_dir / f"run{run}"
        cfg = json.load(open(f"{base}_config.json"))
        net = make_net(cfg, S["n_classes"], edge_types, graphs[:8], device)
        net.load_state_dict(torch.load(f"{base}.pt", map_location=device))
        df = NA.evaluate_next(net, graphs, meta, decoder, S["class_ids"], [ACT_KEY, TS_KEY], device,
                              mask_to_idx, DataLoader, beam=a.beam)
        df.insert(0, "run", run)
        preds.append(df)
        per_run.append({"run": run, **NA.summarize_run(df)})
        print(f"  run {run}: raw {per_run[-1]['acc_raw']:.4f}  hard {per_run[-1]['acc_hard']:.4f}")

    out = a.out_dir / ds
    out.mkdir(parents=True, exist_ok=True)
    preds = pd.concat(preds, ignore_index=True)
    res = pd.DataFrame(per_run)
    res["diff_hard_raw"] = res.acc_hard - res.acc_raw
    stats = res.drop(columns="run").agg(["mean", "std"])
    by_len = (preds.groupby("k").agg(acc_raw=("ok_raw", "mean"), acc_hard=("ok_hard", "mean"),
                                     raw_violation_rate=("raw_violates", "mean"), n=("ok_raw", "size"))
              .assign(n=lambda d: d.n // n_runs).reset_index())
    preds.to_csv(out / f"next_act_predictions_{a.tag}.csv", index=False)
    res.to_csv(out / f"next_act_RESULTS_{a.tag}.csv", index=False)
    stats.to_csv(out / f"next_act_MEAN_STD_{a.tag}.csv")
    by_len.to_csv(out / f"next_act_by_prefix_len_{a.tag}.csv", index=False)
    print(stats.round(4).to_string())
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=DATASETS, help="default: all six")
    ap.add_argument("--tag", default="sl", help="models in trained_model/<dataset>_AT_only/<tag>/")
    ap.add_argument("--runs", type=int, default=None, help="default: every saved run")
    ap.add_argument("--sup-min", type=float, default=0.99, help="rules with train support >= this are enforced")
    ap.add_argument("--beam", type=int, default=16, help="one missing position per prefix: any beam >= #activities is exact")
    ap.add_argument("--check", action="store_true", help="run the sanity checks on each dataset first")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "next_activity"))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    from pathlib import Path
    a.out_dir = Path(a.out_dir)
    device = torch.device(a.device)
    seed_everything(0)
    pd.set_option("display.max_columns", None)

    rows = []
    for name in a.datasets:
        S = setup_dataset(name)
        if a.check:
            sanity_checks(S)
        res = run_dataset(S, a, device)
        rows.append({"dataset": name, "n_prefixes": int(res.n_prefixes.iloc[0]),
                     "acc_raw": res.acc_raw.mean(), "acc_raw_std": res.acc_raw.std(),
                     "acc_hard": res.acc_hard.mean(), "acc_hard_std": res.acc_hard.std(),
                     "diff_hard_raw": res.diff_hard_raw.mean(), "diff_std": res.diff_hard_raw.std(),
                     "raw_violation_rate": res.raw_violation_rate.mean(), "truth_allowed": res.truth_allowed.mean(),
                     "mean_allowed": res.mean_allowed.mean(), "fallback": res.fallback.mean()})

    path = a.out_dir / f"summary_{a.tag}.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old):
        old = old[~old.dataset.isin(a.datasets)]
    summary = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
