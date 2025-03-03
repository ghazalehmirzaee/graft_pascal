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

    # Handle empty arrays
    if y_true.size == 0 or y_score.size == 0:
        return 0.0

    # Check for NaNs or infinities
    if np.isnan(y_score).any() or np.isinf(y_score).any():
        y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)

    # Calculate mAP with error handling
    try:
        return average_precision_score(y_true, y_score, average=average)
    except Exception as e:
        print(f"Error computing mAP: {e}")
        return 0.0


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

    # Handle empty arrays
    if y_true.size == 0 or y_score.size == 0:
        return np.zeros(y_true.shape[1])

    # Check for NaNs or infinities
    if np.isnan(y_score).any() or np.isinf(y_score).any():
        y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)

    # Calculate AP with error handling
    try:
        ap_scores = average_precision_score(y_true, y_score, average=None)
        # Handle possible None values from sklearn
        if ap_scores is None:
            return np.zeros(y_true.shape[1])
        return ap_scores
    except Exception as e:
        print(f"Error computing per-class AP: {e}")
        return np.zeros(y_true.shape[1])


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

    # Handle empty arrays
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0, 0.0, 0.0

    # Calculate metrics with error handling
    try:
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        return precision, recall, f1
    except Exception as e:
        print(f"Error computing precision/recall/f1: {e}")
        return 0.0, 0.0, 0.0


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

    # Handle empty arrays
    if y_true.size == 0 or y_pred.size == 0:
        n_classes = y_true.shape[1] if y_true.size > 0 else y_pred.shape[1]
        return np.zeros(n_classes), np.zeros(n_classes), np.zeros(n_classes)

    # Calculate per-class metrics with error handling
    try:
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        return precision, recall, f1
    except Exception as e:
        print(f"Error computing per-class precision/recall/f1: {e}")
        n_classes = y_true.shape[1]
        return np.zeros(n_classes), np.zeros(n_classes), np.zeros(n_classes)


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

    # Handle empty arrays
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0

    # Calculate Hamming Loss with error handling
    try:
        return np.mean(y_true != y_pred)
    except Exception as e:
        print(f"Error computing Hamming Loss: {e}")
        return 0.0


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

    # Handle empty arrays
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0

    # Calculate Subset Accuracy with error handling
    try:
        return np.mean(np.all(y_true == y_pred, axis=1))
    except Exception as e:
        print(f"Error computing Subset Accuracy: {e}")
        return 0.0


def compute_metrics(
        y_true: Union[np.ndarray, torch.Tensor],
        y_score: Union[np.ndarray, torch.Tensor],
        threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics for multi-label classification with improved error handling.

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

    # Handle empty arrays
    if y_true.size == 0 or y_score.size == 0:
        return {"mAP": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "hamming_loss": 0.0, "subset_accuracy": 0.0}

    # Ensure arrays have proper shape
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    if len(y_score.shape) == 1:
        y_score = y_score.reshape(-1, 1)

    # Ensure arrays have the same shape
    if y_true.shape != y_score.shape:
        print(f"Shape mismatch: y_true {y_true.shape}, y_score {y_score.shape}")
        # Try to reshape if possible
        if y_true.size == y_score.size:
            y_score = y_score.reshape(y_true.shape)
        else:
            # Return zeros for all metrics
            return {"mAP": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "hamming_loss": 0.0, "subset_accuracy": 0.0}

    # Check for NaNs or infinities
    if np.isnan(y_score).any() or np.isinf(y_score).any():
        y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)

    # Convert scores to binary predictions
    y_pred = (y_score >= threshold).astype(int)

    try:
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
            "AP": ap.tolist(),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "per_class_precision": per_class_precision.tolist(),
            "per_class_recall": per_class_recall.tolist(),
            "per_class_f1": per_class_f1.tolist(),
            "hamming_loss": h_loss,
            "subset_accuracy": subset_acc,
            "sensitivity": sensitivity.tolist(),
            "specificity": specificity.tolist(),
            "avg_sensitivity": avg_sensitivity,
            "avg_specificity": avg_specificity
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        # Return zeros for all metrics on error
        return {"mAP": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "hamming_loss": 0.0, "subset_accuracy": 0.0}

