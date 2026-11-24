"""
Rule mining and the two rule-based components of AEGIS:
  * the LTLf semantic loss (training time), built on ALL mined rules, weighted by their mining score;
  * the hard-constrained decoder (inference time), built on the rules with train support >= sup_min.
"""
from collections import Counter
from dataclasses import dataclass

import numpy as np

from . import ltlf_patterns as LP
from .ltlf_decode import HardDecoder
from .ltlf_semantic_loss import LTLfSemanticLoss
from .data import ACT_KEY


@dataclass
class Rules:
    labels: list          # human-readable rule names (compact activity ids)
    dfas: list            # one DFA per rule
    weights: list         # mining score of each rule (semantic-loss weights)
    train_sup: np.ndarray # share of complete training traces each rule accepts

    def subset(self, sup_min):
        """Indices of the rules with train support >= sup_min."""
        return [k for k, s in enumerate(self.train_sup) if s >= sup_min]


def mine(data, top_k=60, verbose=True):
    """Mine DECLARE/LTLf rules (all pattern families) on the complete training traces, compiled to DFAs."""
    rules = LP.mine_rich_rules(data.full_seqs, data.a_real, families=LP.FAMILIES, top_k=top_k, verbose=verbose)
    dfas = [d for _, d, _ in rules]
    cnt = Counter(tuple(s) for s in data.full_seqs)
    n = len(data.full_seqs)
    train_sup = np.array([sum(c for s, c in cnt.items() if d.accepts(s)) / n for d in dfas])
    if verbose:
        print("symbol -> activity:", dict(enumerate(data.activity_names)))
        for lab, _, s in rules[:15]:
            print(round(s, 3), lab)
    return Rules([lab for lab, _, _ in rules], dfas, [s for _, _, s in rules], train_sup)


def semantic_loss(data, rules, device):
    return LTLfSemanticLoss(rules.dfas, data.class_ids, data.n_classes, act_key=ACT_KEY,
                            rule_weights=rules.weights).to(device)


def hard_decoder(data, rules, sup_min=0.99):
    ridx = rules.subset(sup_min)
    return HardDecoder(rules.dfas, data.a_real, rule_idx=ridx), ridx
