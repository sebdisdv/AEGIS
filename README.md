# AEGIS

Code of the paper *Automata-enhanced Heterogeneous Graph Neural Networks for Informed Trace Repair*. The base repair model is
the HGNN of *Graph-based Event Log Repair* ([arXiv:2508.05145](https://arxiv.org/abs/2508.05145)).

## Requirements

Python 3.12. A CUDA GPU is recommended; everything also runs on the CPU, only slower.

| Library | Version |
|---|---|
| torch | 2.9.0 |
| torch_geometric | 2.7.0 |
| torcheval | 0.0.7 |
| optuna | 5.0.0 |
| numpy | 1.26.3 |
| pandas | 2.2.3 |
| scipy | 1.13.1 |
| matplotlib | 3.9.2 |
| tqdm | 4.66.5 |
| jupyter, ipywidgets 8.1.5 | only for `main.ipynb` |

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128   # pick the build for your CUDA version
pip install -r requirements.txt
```

The datasets and the trained models are stored with [Git LFS](https://git-lfs.com) (about 2 GB). Run `git lfs install`
before cloning, or `git lfs pull` after cloning.

## Data

The repository contains the six event logs used in the paper: `bpi_2012_CZ`, `bpi_2013_CZ`, `small_log_CZ`,
`large_log_CZ`, `sp2020_CZ` and `BPI20_RequestForPayment_CZ`. Each log is stored as graphs, one per trace, with the
activity and timestamp of each event:

```
data/datasets/graphs_repair/<dataset>_AT_only_{TRAIN,VALID,TEST}_V2_repair.pkl
data/datasets/graphs_repair/<dataset>_AT_only_TEST_V2_repair_{even,odd,window,random,attr_level}.pkl   (the five test sets)
data/datasets/_processed/<dataset>/<dataset>_processed_all.csv                                        (activity vocabulary)
```

Original logs:
- [BPI Challenge 2012](https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204/1)
- [BPI Challenge 2013, open problems](https://data.4tu.nl/articles/dataset/BPI_Challenge_2013_open_problems/12688556/1)
- [BPI Challenge 2020, Request For Payment](https://data.4tu.nl/articles/dataset/BPI_Challenge_2020_Request_For_Payment/12706886/1)
- [SP 2020](https://doi.org/10.5281/zenodo.3928487)
- Small Log and Large Log from [Autoencoders for improving quality of process event logs](https://doi.org/10.1016/j.eswa.2019.04.052)

Preprocessing, the train/validation/test split and the masking strategies follow the base paper.

## Reproduction

Every script can be run from any directory. `<dataset>` is one of the six names above.

**Repair, using the trained models.** This re-evaluates the 10 published models per dataset and gives the same values
as `results_reference/repair/`:

```bash
python main.py --dataset <dataset> --eval-only
```

**Repair, retraining from scratch.** This trains 10 networks and evaluates them on the five test sets:

```bash
python main.py --dataset <dataset> --tag sl_repro          # hyperparameters from configs/
python main.py --dataset <dataset> --tag sl_repro --hpo    # Optuna search first (30 trials)
```

Main options and their defaults:
- `--runs 10`
- `--epochs 200` (early stopping, patience 10)
- `--lambda-sem 0.3`
- `--sup-min 0.99`
- `--beam 16`
- `--top-k 60`
- `--seed 0` (run *i* uses seed + *i*)

`--tag` names the run. A run never overwrites existing models with the same tag unless you pass `--overwrite`.

Outputs:
- models: `trained_model/<dataset>_AT_only/<tag>/`
- results: `results/repair/<dataset>_AT_only/`

`main.ipynb` runs the same pipeline cell by cell.

On a GPU, retrained runs match the published numbers statistically (mean ± std over runs), not bit for bit. This is
because of non-deterministic scatter operations in PyTorch Geometric and bf16 autocast.

**Next-activity prediction.** This uses the saved models of `--tag` (default `sl`) and writes `results/next_activity/`:

```bash
python next_activity.py                              # all datasets
python next_activity.py --datasets <dataset> --check
```

**Plots and significance tests.** By default these read `results_reference/` and write to `plots/`:

```bash
python plot_sl_hard.py                               # SL / SL + hard vs baseline (BASELINE.json), Welch t-test + Holm
python plot_sl_hard.py --results-dir results/repair --tag sl_repro
python plot_next_activity.py
python plot_next_activity.py --results-dir results/next_activity
```

**Figures of the paper.** Each script writes its figure to `figures/`:

```bash
python figures/results_rq.py           # rq_summary, rq1_summary, rq2_summary
python figures/next_activity_dist.py   # next_activity_dist
python figures/dfa_run.py              # dfa_run
python figures/encoding_AT.py          # encoding_AT
```

## License

MIT, see [LICENSE](LICENSE).
