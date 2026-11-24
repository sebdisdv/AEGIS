#!/usr/bin/env python3
"""
AEGIS: train and evaluate the HGNN with the LTLf semantic loss (SL) and with SL + hard constraints.

Pipeline: load graphs -> mine DECLARE/LTLf rules on the complete training traces -> semantic loss on all mined rules,
hard decoder on the rules with train support >= --sup-min -> [Optuna HPO] -> train --runs networks -> evaluate on the
five test sets (even, odd, window, random, attr_level).

Usage:
    python main.py --dataset bpi_2013_CZ                    # shipped best config, 10 runs x 200 epochs
    python main.py --dataset bpi_2013_CZ --hpo              # hyperparameter search first (30 trials x 200 epochs)
    python main.py --dataset bpi_2013_CZ --eval-only        # evaluate the pretrained models in trained_model/
    nohup python main.py --dataset sp2020_CZ --tag sl_new > sp2020_sl_new.log 2>&1 &
"""
import argparse
import json
import sys

import torch

from aegis import data as D
from aegis import rules as R
from aegis.experiment import MODEL_DIR, RESULTS_DIR, evaluate_saved, test_multi


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=D.DATASETS + [D.full_name(d) for d in D.DATASETS])
    ap.add_argument("--tag", default="sl", help="name of this configuration in trained_model/ and results/")
    ap.add_argument("--runs", type=int, default=10, help="independent training runs")
    ap.add_argument("--epochs", type=int, default=200, help="max epochs per run (early stopping, patience 10)")
    ap.add_argument("--lambda-sem", type=float, default=0.3, help="weight of the semantic loss")
    ap.add_argument("--sup-min", type=float, default=0.99, help="hard constraints: rules with train support >= this")
    ap.add_argument("--beam", type=int, default=16, help="beam width of the hard-constrained decoder")
    ap.add_argument("--top-k", type=int, default=60, help="max number of mined rules")
    ap.add_argument("--config", help="hyperparameter JSON (default: configs/<dataset>_AT_only.json)")
    ap.add_argument("--hpo", action="store_true", help="run the Optuna search and train with its best config")
    ap.add_argument("--hpo-trials", type=int, default=30)
    ap.add_argument("--hpo-epochs", type=int, default=200)
    ap.add_argument("--eval-only", action="store_true", help="evaluate saved models of --tag, no training")
    ap.add_argument("--overwrite", action="store_true", help="allow overwriting saved models of --tag")
    ap.add_argument("--seed", type=int, default=0, help="run i uses seed + i")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    a = parse_args()
    device = torch.device(a.device)
    ds = D.full_name(a.dataset)
    model_path = MODEL_DIR / ds / a.tag
    if not a.eval_only and any(model_path.glob("run*.pt")) and not a.overwrite:
        sys.exit(f"{model_path} already holds trained models: pick another --tag or pass --overwrite")

    D.seed_everything(a.seed)
    data = D.load_dataset(a.dataset)
    rules = R.mine(data, top_k=a.top_k)
    sem_loss = R.semantic_loss(data, rules, device)
    decoder, ridx = R.hard_decoder(data, rules, a.sup_min)
    print(f"{len(rules.dfas)} rules in the semantic loss, {len(ridx)} enforced by the hard decoder (support >= {a.sup_min})")
    tests = D.load_test_sets(a.dataset)

    if a.eval_only:
        evaluate_saved(data, tests, decoder, device, tag=a.tag, beam=a.beam, results_dir=a.results_dir)
        return

    if a.hpo:
        from aegis.hpo import run_hpo
        _, config = run_hpo(data, device, sem_loss, lambda_sem=a.lambda_sem, n_trials=a.hpo_trials,
                            epochs=a.hpo_epochs, seed=a.seed)
        out = D.ROOT / "results" / ds
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"best_config_{a.tag}.json", "w") as f:
            json.dump(config, f, indent=2)
    else:
        config = D.load_config(a.dataset, a.config)
    config = {**config, "lambda_sem": a.lambda_sem}

    test_multi(config, data, tests, sem_loss, decoder, device, tag=a.tag, num_runs=a.runs, num_epochs=a.epochs,
               beam=a.beam, sup_min=a.sup_min, seed=a.seed, results_dir=a.results_dir)


if __name__ == "__main__":
    main()
