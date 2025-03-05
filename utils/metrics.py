"""
Evaluation metrics for multi-label classification.

This module implements various evaluation metrics for multi-label classification,
including mean average precision, F1 score, precision, recall, and others.
"""
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score


def mean_average_precision(
        y_true: Union[np.ndarray, torch.Tensor],
        y_score: Union[np.ndarray, torch.Tensor],
        average: str = 'macro'
) -> float:
    """
    Compute Mean Average Precision (mAP).

    Args:
        y_true: True binary labels of shape [n_samples, n_classes].
        y_score: Target scores of shape [n_samples, n_classes].
        average: Averaging method ('micro', 'macro', 'samples', 'weighted', 'none').

    Returns:
        Mean Average Precision score.
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_score, torch.Tensor):
        y_score = y_score.detach().cpu().numpy()

    return average_precision_score(y_true, y_score, average=average)


def per_class_average_precision(
        y_true: Union[np.ndarray, torch.Tensor],
        y_score: Union[np.ndarray, torch.Tensor]
) -> np.ndarray:
    """
    Compute Average Precision for each class.

    Args:
        y_true: True binary labels of shape [n_samples, n_classes].
        y_score: Target scores of shape [n_samples, n_classes].

    Returns:
        Array of Average Precision scores for each class.
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_score, torch.Tensor):
        y_score = y_score.detach().cpu().numpy()

    return average_precision_score(y_true, y_score, average=None)


def precision_recall_f1(
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        average: str = 'macro'
) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, and F1 Score.

    Args:
        y_true: True binary labels of shape [n_samples, n_classes].
        y_pred: Predicted binary labels of shape [n_samples, n_classes].
        average: Averaging method ('micro', 'macro', 'samples', 'weighted', 'none').

    Returns:
        Tuple of (precision, recall, f1).
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    return precision, recall, f1


def per_class_precision_recall_f1(
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Precision, Recall, and F1 Score for each class.

    Args:
        y_true: True binary labels of shape [n_samples, n_classes].
        y_pred: Predicted binary labels of shape [n_samples, n_classes].

    Returns:
        Tuple of (precision, recall, f1) arrays.
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    return precision, recall, f1


def hamming_loss(
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute Hamming Loss.

    Args:
        y_true: True binary labels of shape [n_samples, n_classes].
        y_pred: Predicted binary labels of shape [n_samples, n_classes].

    Returns:
        Hamming Loss score.
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return np.mean(y_true != y_pred)


def subset_accuracy(
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute Subset Accuracy (Exact Match Ratio).

    Args:
        y_true: True binary labels of shape [n_samples, n_classes].
        y_pred: Predicted binary labels of shape [n_samples, n_classes].

    Returns:
        Subset Accuracy score.
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return np.mean(np.all(y_true == y_pred, axis=1))


def compute_metrics(
        y_true: Union[np.ndarray, torch.Tensor],
        y_score: Union[np.ndarray, torch.Tensor],
        threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics for multi-label classification.

    Args:
        y_true: True binary labels of shape [n_samples, n_classes].
        y_score: Target scores of shape [n_samples, n_classes].
        threshold: Threshold for converting scores to binary predictions.

    Returns:
        Dictionary of metrics.
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_score, torch.Tensor):
        y_score = y_score.detach().cpu().numpy()

    # Convert scores to binary predictions
    y_pred = (y_score >= threshold).astype(int)

    # Compute Mean Average Precision
    mAP = mean_average_precision(y_true, y_score)

    # Compute per-class Average Precision
    ap = per_class_average_precision(y_true, y_score)

    # Compute precision, recall, F1 (macro-averaged)
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)

    # Compute per-class precision, recall, F1
    per_class_precision, per_class_recall, per_class_f1 = per_class_precision_recall_f1(y_true, y_pred)

    # Compute Hamming Loss
    h_loss = hamming_loss(y_true, y_pred)

    # Compute Subset Accuracy
    subset_acc = subset_accuracy(y_true, y_pred)

    # Calculate total positives and true positives for sensitivity and specificity
    n_samples, n_classes = y_true.shape

    # True positives (TP): Correctly predicted positive observations
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1), axis=0)

    # True negatives (TN): Correctly predicted negative observations
    tn = np.sum(np.logical_and(y_pred == 0, y_true == 0), axis=0)

    # False positives (FP): Incorrectly predicted positive observations
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0), axis=0)

    # False negatives (FN): Incorrectly predicted negative observations
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1), axis=0)

    # Sensitivity (Recall)
    sensitivity = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)

    # Specificity
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) != 0)

    # Average sensitivity and specificity
    avg_sensitivity = np.mean(sensitivity)
    avg_specificity = np.mean(specificity)

    # Return dictionary of metrics
    return {
        "mAP": mAP,
        "AP": ap,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "hamming_loss": h_loss,
        "subset_accuracy": subset_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "avg_sensitivity": avg_sensitivity,
        "avg_specificity": avg_specificity
    }

