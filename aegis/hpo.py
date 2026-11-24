"""
Hyperparameter search with Optuna (TPE sampler, median pruner): train on X_train, score = best validation loss.
The semantic-loss weight is fixed (not tuned).
"""
import gc
import json
import math

import optuna
import torch

from .model import train_hgnn

FIXED = {"amp": True, "num_workers": 4}


def sample_config(trial, lambda_sem):
    return {
        "hid": trial.suggest_categorical("hid", [32, 64, 128, 256]),
        "layers": trial.suggest_int("layers", 1, 4),
        "aggregation": trial.suggest_categorical("aggregation", ["mean", "sum", "max"]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        # GradNorm
        "lr_gradNorm": trial.suggest_float("lr_gradNorm", 1e-3, 1e-1, log=True),
        "alpha": trial.suggest_float("alpha", 0.0, 2.0),
        "gn_every": trial.suggest_categorical("gn_every", [1, 2]),
        # fixed
        **FIXED,
        "lambda_sem": lambda_sem,
    }


def make_objective(data, device, sem_loss, lambda_sem, epochs):
    def objective(trial):
        config = sample_config(trial, lambda_sem)
        torch.manual_seed(trial.number)

        def callback(epoch, val_loss):
            if not math.isfinite(val_loss):
                raise optuna.TrialPruned()
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        net = None
        try:
            net = train_hgnn(config, data, device, sem_loss=sem_loss, epochs=epochs, callback=callback)
            return net.best_val_loss
        except torch.cuda.OutOfMemoryError:
            raise optuna.TrialPruned()  # config too big for the GPU
        finally:
            del net
            gc.collect()
            torch.cuda.empty_cache()

    return objective


def run_hpo(data, device, sem_loss, lambda_sem=0.3, n_trials=30, epochs=200, timeout=None, study_name="hgnn",
            storage=None, seed=0):
    """Returns (study, best_config). best_config is ready for train_hgnn."""
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(make_objective(data, device, sem_loss, lambda_sem, epochs), n_trials=n_trials, timeout=timeout,
                   gc_after_trial=True)

    print("Best val loss:", study.best_value)
    print("Best params:", json.dumps(study.best_params, indent=2))
    try:
        print("Importances:", optuna.importance.get_param_importances(study))
    except Exception:
        pass
    return study, {**study.best_params, **FIXED, "lambda_sem": lambda_sem}
