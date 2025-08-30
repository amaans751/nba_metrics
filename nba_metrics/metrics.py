import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, precision_score, recall_score, brier_score_loss
)

def precision_recall_at_k(y_true, y_prob, ks=(20,50,100)):
    order = np.argsort(-np.asarray(y_prob))
    y_sorted = np.asarray(y_true)[order]
    total_pos = y_sorted.sum()
    out = {}
    for k in ks:
        topk = y_sorted[:k]
        out[f"precision@{k}"] = float(topk.mean() if k>0 else 0.0)
        out[f"recall@{k}"]    = float(topk.sum()/total_pos) if total_pos>0 else 0.0
    return out

def best_f1_threshold(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    idx = int(np.nanargmax(f1s))
    return float(thr[max(idx-1, 0)] if idx < len(thr) else 0.5), float(f1s[idx])

def pr_curve_data(y_true, y_prob):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    return pd.DataFrame({"recall": rec, "precision": prec})

def evaluate_classification(y_true, y_prob, ks=(20,50,100)):
    roc = roc_auc_score(y_true, y_prob)
    ap  = average_precision_score(y_true, y_prob)
    thr, f1 = best_f1_threshold(y_true, y_prob)
    y_hat = (np.asarray(y_prob) >= thr).astype(int)
    p = precision_score(y_true, y_hat, zero_division=0)
    r = recall_score(y_true, y_hat, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)
    topk = precision_recall_at_k(y_true, y_prob, ks)
    return {
        "ROC-AUC": roc,
        "PR-AUC": ap,
        "best_thr": thr,
        "F1@best_thr": f1,
        "Precision@best_thr": p,
        "Recall@best_thr": r,
        "Brier": brier,
        **topk,
    }
