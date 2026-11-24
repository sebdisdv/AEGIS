"""
Multi-run experiment: train `num_runs` HGNNs with the semantic loss and evaluate, on every test type,
    SL       argmax of the network output                        (columns acc_sl, sat_sl)
    SL+hard  hard-constrained beam decoding of the same output   (columns acc_sl_hard, sat_sl_hard)
plus the timestamp MAE of the network (the hard constraints change only activities).

Outputs (relative to the AEGIS folder):
    trained_model/<dataset>/<tag>/run{i}.pt, run{i}_config.json
    <results_dir>/<dataset>/{test_type}_RESULTS_{tag}.csv     one row per run
    <results_dir>/<dataset>/{test_type}_MEAN_STD_{tag}.csv
    <results_dir>/<dataset>/<dataset>_summary_{tag}.csv       one row per test type
"""
import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from . import ltlf_decode as LD
from .data import ACT_KEY, ROOT, TEST_TYPES, seed_everything
from .model import evaluate_hgnn, load_net, mask_to_idx, train_hgnn

MODEL_DIR = ROOT / "trained_model"
RESULTS_DIR = ROOT / "results" / "repair"
HARD_KEYS = {"acc_plain": "acc_sl", "acc_hard": "acc_sl_hard", "sat_plain": "sat_sl", "sat_hard": "sat_sl_hard",
             "traces_changed": "traces_changed", "fallback": "fallback", "n_missing": "n_missing"}


def evaluate_run(net, data, tests, decoder, device, beam=16):
    """{test_type: row} for one trained network."""
    rows = {}
    for test_type, X in tests.items():
        print(f"Test type {test_type}")
        r = evaluate_hgnn(net, data.outputcat, data.outputreal, X, device, mode="missing")
        h = LD.evaluate_hard(net, X, decoder, data.class_ids, data.missing_id, data.n_classes, data.keys, device,
                             mask_to_idx, DataLoader, beam=beam, act_key=ACT_KEY)
        if abs(h["acc_plain"] - r["Activity_acc"]) > 1e-3:
            print(f"  warning: argmax accuracy differs ({h['acc_plain']:.4f} vs {r['Activity_acc']:.4f})")
        r.update({HARD_KEYS[k]: v for k, v in h.items()})
        print(f"  SL {r['acc_sl']:.4f} | SL+hard {r['acc_sl_hard']:.4f} | timestamp MAE {r['time:timestamp_mae']:.5f}")
        rows[test_type] = r
    return rows


def write_results(res, dataset, tag, results_dir=RESULTS_DIR):
    """res: {test_type: [row per run]}. Writes per-run, mean/std and summary CSVs; returns the summary."""
    out = Path(results_dir) / dataset
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for test_type in [t for t in TEST_TYPES if t in res]:
        tab = pd.DataFrame(res[test_type])
        tab.to_csv(out / f"{test_type}_RESULTS_{tag}.csv", index=False)
        pd.DataFrame({"mean": tab.mean(), "std": tab.std()}).to_csv(out / f"{test_type}_MEAN_STD_{tag}.csv")
        summary.append({
            "tag": tag, "test_type": test_type, "runs": len(tab), "n_missing": tab["n_missing"].iloc[0],
            "acc_sl": tab["acc_sl"].mean(), "acc_sl_std": tab["acc_sl"].std(),
            "acc_sl_hard": tab["acc_sl_hard"].mean(), "acc_sl_hard_std": tab["acc_sl_hard"].std(),
            "sat_sl": tab["sat_sl"].mean(), "sat_sl_hard": tab["sat_sl_hard"].mean(),
            "timestamp_mae": tab["time:timestamp_mae"].mean(), "timestamp_mae_std": tab["time:timestamp_mae"].std(),
            "fallback": tab["fallback"].mean(),
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(out / f"{dataset}_summary_{tag}.csv", index=False)
    print(summary.round(4).to_string(index=False))
    return summary


def test_multi(config, data, tests, sem_loss, decoder, device, tag="sl", num_runs=10, num_epochs=200, beam=16,
               sup_min=0.99, seed=0, results_dir=RESULTS_DIR):
    """Train num_runs networks with `config` (lambda_sem > 0: semantic loss) and evaluate SL / SL+hard on `tests`."""
    model_path = MODEL_DIR / data.name / tag
    model_path.mkdir(parents=True, exist_ok=True)
    res = {}
    for run in range(num_runs):
        print(f"Run {run}")
        seed_everything(seed + run)
        net = train_hgnn(config, data, device, sem_loss=sem_loss, epochs=num_epochs)
        torch.save(net.state_dict(), model_path / f"run{run}.pt")
        with open(model_path / f"run{run}_config.json", "w") as f:
            json.dump({**config, "best_val_loss": float(net.best_val_loss), "sup_min": sup_min, "seed": seed + run},
                      f, indent=2)
        for tt, r in evaluate_run(net, data, tests, decoder, device, beam).items():
            res.setdefault(tt, []).append(r)
    return res, write_results(res, data.name, tag, results_dir)


def evaluate_saved(data, tests, decoder, device, tag="sl", num_runs=None, beam=16, results_dir=RESULTS_DIR):
    """Evaluate the networks saved in trained_model/<dataset>/<tag>/ (no training)."""
    model_path = MODEL_DIR / data.name / tag
    runs = sorted(model_path.glob("run*_config.json"), key=lambda p: int(p.name[3:].split("_")[0]))
    assert runs, f"no saved models in {model_path}"
    runs = runs[:num_runs] if num_runs else runs
    res = {}
    for cfg_path in runs:
        run = cfg_path.name.split("_")[0]
        print(f"Saved model {model_path / run}.pt")
        with open(cfg_path) as f:
            cfg = json.load(f)
        net = load_net(model_path / f"{run}.pt", cfg, data, device)
        for tt, r in evaluate_run(net, data, tests, decoder, device, beam).items():
            res.setdefault(tt, []).append(r)
    return res, write_results(res, data.name, tag, results_dir)
