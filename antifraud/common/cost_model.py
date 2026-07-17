"""Cost-sensitive evaluation (IMPROVEMENTS §2.1 / plan A6).

Fraud detection is an economic problem: a missed fraud (false negative) costs the
transaction amount; a false alarm (false positive) costs a fixed manual-review fee.
F1 treats both errors equally — banks do not. These helpers let us choose the
operating threshold that *minimises expected cost* and report money protected.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostResult:
    threshold: float
    total_cost: float          # £ lost to FN amounts + FP review fees
    amount_saved: float        # fraud £ correctly blocked
    false_negatives: int
    false_positives: int
    true_positives: int


def evaluate_cost(y_true, y_proba, amounts, threshold, review_cost: float) -> CostResult:
    """Cost of operating at a given threshold.

    FN cost = summed amount of frauds we let through.
    FP cost = review_cost per legit transaction we wrongly flag.
    amount_saved = summed amount of frauds we correctly blocked.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    amounts = np.asarray(amounts, dtype=float)

    y_pred = (y_proba >= threshold).astype(int)
    is_fraud = y_true == 1

    fn_mask = is_fraud & (y_pred == 0)
    fp_mask = (~is_fraud) & (y_pred == 1)
    tp_mask = is_fraud & (y_pred == 1)

    fn_cost = amounts[fn_mask].sum()
    fp_cost = review_cost * fp_mask.sum()
    amount_saved = amounts[tp_mask].sum()

    return CostResult(
        threshold=float(threshold),
        total_cost=float(fn_cost + fp_cost),
        amount_saved=float(amount_saved),
        false_negatives=int(fn_mask.sum()),
        false_positives=int(fp_mask.sum()),
        true_positives=int(tp_mask.sum()),
    )


def best_cost_threshold(y_true, y_proba, amounts, review_cost: float,
                        n_steps: int = 200) -> CostResult:
    """Sweep thresholds and return the one that minimises expected cost."""
    grid = np.linspace(0.001, 0.999, n_steps)
    results = [evaluate_cost(y_true, y_proba, amounts, t, review_cost) for t in grid]
    return min(results, key=lambda r: r.total_cost)
