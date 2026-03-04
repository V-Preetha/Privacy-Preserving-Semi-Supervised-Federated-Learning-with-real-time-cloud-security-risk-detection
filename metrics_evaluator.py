"""Compute evaluation metrics from sklearn for a test dataset."""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import torch


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Compute basic metrics. y_test should be binary 0/1.

    Returns dict with accuracy, precision, recall, f1, roc_auc (if computable)
    """
    if len(y_test) == 0:
        return {}
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(X_test.astype('float32'))
        preds = model(X).cpu().numpy().flatten()
    # binary predictions
    y_pred = (preds >= 0.5).astype(int)
    y_true = y_test.astype(int)
    res = {}
    res['accuracy'] = float(accuracy_score(y_true, y_pred))
    # handle cases with no positive predictions / labels
    try:
        res['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
        res['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
        res['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception:
        res['precision'] = res['recall'] = res['f1'] = 0.0
    try:
        res['roc_auc'] = float(roc_auc_score(y_true, preds))
    except Exception:
        res['roc_auc'] = None
    return res
