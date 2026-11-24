"""
Dataset loading for AEGIS.

Each dataset is a set of pickled lists of PyG HeteroData graphs (one graph per trace) with two node types:
`Activity` (one-hot activity, MISSING class for masked events) and `time:timestamp` (normalised timestamp).
Masked nodes are listed in `g.masks[node_type]`, labels in `g.y[node_type]`.

Files (in data/datasets/graphs_repair/):
    <dataset>_AT_only_TRAIN_V2_repair.pkl, _VALID_V2_repair.pkl, _TEST_V2_repair.pkl
    <dataset>_AT_only_TEST_V2_repair_<test_type>.pkl   for test_type in TEST_TYPES
"""
import json
import pickle
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.transforms import ToUndirected

ROOT = Path(__file__).resolve().parent.parent          # the AEGIS folder
DATA_DIR = ROOT / "data"
GRAPHS_DIR = DATA_DIR / "datasets" / "graphs_repair"
PROCESSED_DIR = DATA_DIR / "datasets" / "_processed"

DATASETS = ["bpi_2012_CZ", "bpi_2013_CZ", "small_log_CZ", "large_log_CZ", "sp2020_CZ", "BPI20_RequestForPayment_CZ"]
TEST_TYPES = ["even", "odd", "window", "random", "attr_level"]
ACT_KEY, TS_KEY = "Activity", "time:timestamp"
MISSING_VALUE = "MISSING_VALUE"

transform = ToUndirected()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def full_name(dataset):
    """'bpi_2013_CZ' -> 'bpi_2013_CZ_AT_only' (only activity + timestamp are used)."""
    return dataset if dataset.endswith("_AT_only") else f"{dataset}_AT_only"


def base_name(dataset):
    return dataset[: -len("_AT_only")] if dataset.endswith("_AT_only") else dataset


def load_graphs(name, undirected=True):
    """name: file stem without .pkl, e.g. 'bpi_2013_CZ_AT_only_TEST_V2_repair_even'."""
    with open(GRAPHS_DIR / f"{name}.pkl", "rb") as f:
        graphs = pickle.load(f)
    if undirected:
        with torch.no_grad():
            graphs = [transform(g) for g in graphs]
    return graphs


def load_test_sets(dataset, test_types=TEST_TYPES):
    ds = full_name(dataset)
    return {tt: load_graphs(f"{ds}_TEST_V2_repair_{tt}") for tt in test_types}


def infer_missing_id(graphs, n_classes, n=2000):
    """Class index used to encode masked Activity nodes (read from the data, not assumed)."""
    c = Counter()
    for g in graphs[:n]:
        m = g.masks[ACT_KEY].bool()
        c.update(g[ACT_KEY].x[m][:, :n_classes].argmax(1).tolist())
    assert len(c) == 1, f"masked Activity nodes use several encodings: {c}"
    return next(iter(c))


@dataclass
class Dataset:
    """Everything the pipeline needs about one dataset.

    Rules use a compact alphabet 0..A_REAL-1 of the real activities (the MISSING class is not a symbol);
    CLASS_IDS[k] is the model class of compact symbol k, lut maps a model class to its compact id (-1 for MISSING)."""
    name: str                       # e.g. bpi_2013_CZ_AT_only
    X_train: list
    X_valid: list
    X_test: list
    edge_types: list
    outputcat: dict                 # {"Activity": N_CLASSES}
    outputreal: list                # ["time:timestamp"]
    n_classes: int
    missing_id: int
    class_ids: list
    a_real: int
    lut: torch.Tensor
    activity_names: list            # compact id -> activity name
    full_seqs: list = field(repr=False)   # complete training traces in compact ids (from the labels)

    @property
    def keys(self):
        return list(self.outputcat) + list(self.outputreal)

    def compact(self, ids):
        return self.lut[torch.as_tensor(ids)].tolist()

    def act_ids(self, g):
        """Compact activity id per Activity node; masked nodes -> a_real. No labels used."""
        cls = g[ACT_KEY].x[:, : self.n_classes].argmax(1)
        out = self.lut[cls]
        out[cls == self.missing_id] = self.a_real
        return out.tolist()


def load_dataset(dataset, check=True):
    """Load train / valid / test graphs of `dataset` (e.g. 'bpi_2013_CZ') and derive the activity alphabet."""
    base, ds = base_name(dataset), full_name(dataset)
    tab_all = pd.read_csv(PROCESSED_DIR / base / f"{base}_processed_all.csv")
    unique_acts = list(tab_all[ACT_KEY].unique()) + [MISSING_VALUE]
    n_classes = len(unique_acts)
    print(f"{ds}: {len(tab_all)} events, {tab_all['CaseID'].nunique()} traces, {n_classes - 1} activities")

    X_train = load_graphs(f"{ds}_TRAIN_V2_repair")
    X_valid = load_graphs(f"{ds}_VALID_V2_repair")
    X_test = load_graphs(f"{ds}_TEST_V2_repair")

    edge_types = set()
    for X in (X_train, X_valid, X_test):
        for g in X:
            edge_types.update(g.metadata()[1])
    edge_types = sorted(edge_types)                  # fixed order (set order depends on the hash seed)

    missing_id = infer_missing_id(X_train, n_classes)
    class_ids = [c for c in range(n_classes) if c != missing_id]
    a_real = len(class_ids)
    lut = torch.full((n_classes,), -1, dtype=torch.long)
    lut[class_ids] = torch.arange(a_real)
    names = sorted(unique_acts)                      # graph class order is alphabetical
    activity_names = [names[c] for c in class_ids] if names[missing_id] == MISSING_VALUE else [str(c) for c in class_ids]
    print(f"MISSING class id: {missing_id} | real activities: {a_real}")

    if check:
        _sanity_checks(X_train[:200], n_classes, missing_id)

    full_seqs = [lut[g.y[ACT_KEY]].tolist() for g in X_train]
    return Dataset(name=ds, X_train=X_train, X_valid=X_valid, X_test=X_test, edge_types=edge_types,
                   outputcat={ACT_KEY: n_classes}, outputreal=[TS_KEY], n_classes=n_classes, missing_id=missing_id,
                   class_ids=class_ids, a_real=a_real, lut=lut, activity_names=activity_names, full_seqs=full_seqs)


def _sanity_checks(graphs, n_classes, missing_id):
    for g in graphs:
        m, cls, y = g.masks[ACT_KEY].bool(), g[ACT_KEY].x[:, :n_classes].argmax(1), g.y[ACT_KEY]
        assert (cls[m] == missing_id).all() and (cls[~m] != missing_id).all(), "masked <-> MISSING class mismatch"
        assert (cls[~m] == y[~m]).all(), "x.argmax != y on visible nodes"
        assert (y != missing_id).all(), "labels contain the MISSING class"
        ei = g[(ACT_KEY, "followed_by", ACT_KEY)].edge_index      # nodes must be stored in temporal order
        assert ((ei[0] - ei[1]).abs() == 1).all(), "nodes are not stored in consecutive (temporal) order"


def load_config(dataset, path=None):
    """Best hyperparameters shipped in configs/<dataset>_AT_only.json (or an explicit path)."""
    path = Path(path) if path else ROOT / "configs" / f"{full_name(dataset)}.json"
    with open(path) as f:
        return json.load(f)
