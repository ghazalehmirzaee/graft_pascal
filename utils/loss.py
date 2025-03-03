"""
Loss functions for GRAFT multi-label classification.

This module implements various loss functions for multi-label classification,
including weighted binary cross-entropy, focal loss, and asymmetric loss.
"""
from typing import Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WeightedBinaryCrossEntropyLoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss.

    This loss applies class-specific weights to handle class imbalance.
    """

    def __init__(self, weight: Optional[torch.Tensor] = None, reduction: str = 'mean',
                 pos_weight: Optional[torch.Tensor] = None):
        """
        Initialize Weighted BCE Loss.

        Args:
            weight: Optional tensor of class weights.
            reduction: Reduction method ('mean', 'sum', or 'none').
            pos_weight: Optional positive class weights for BCEWithLogitsLoss.
        """
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight

        # Use BCEWithLogitsLoss for numerical stability
        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction='none',
            pos_weight=pos_weight
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input: Predicted logits of shape [batch_size, num_classes].
            target: Target labels of shape [batch_size, num_classes].

        Returns:
            Loss tensor.
        """
        # Compute BCE loss with logits for numerical stability
        loss = self.bce_loss(input, target)

        # Apply class weights if provided
        if self.weight is not None:
            # Expand weights to match batch size
            weight = self.weight.unsqueeze(0).expand_as(target)
            loss = loss * weight

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for Multi-Label Classification.

    This loss down-weights well-classified examples and focuses on hard examples.
    """

    def __init__(
            self,
            gamma: float = 2.0,
            weight: Optional[torch.Tensor] = None,
            reduction: str = 'mean',
            alpha: Optional[float] = None  # Added alpha for balancing positive/negative
    ):
        """
        Initialize Focal Loss.

        Args:
            gamma: Focusing parameter.
            weight: Optional tensor of class weights.
            reduction: Reduction method ('mean', 'sum', or 'none').
            alpha: Balancing parameter for positive/negative examples.
        """
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input: Predicted logits of shape [batch_size, num_classes].
            target: Target labels of shape [batch_size, num_classes].

        Returns:
            Loss tensor.
        """
        # Apply sigmoid to get probabilities
        input_prob = torch.sigmoid(input)

        # Compute binary cross entropy loss (using logits for stability)
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')

        # Compute focal weights
        pt = torch.where(target == 1, input_prob, 1 - input_prob)
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha balancing if provided
        if self.alpha is not None:
            alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
            focal_weight = focal_weight * alpha_weight

        # Apply focal weights
        loss = bce_loss * focal_weight

        # Apply class weights if provided
        if self.weight is not None:
            # Expand weights to match batch size
            weight = self.weight.unsqueeze(0).expand_as(target)
            loss = loss * weight

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification.

    This loss applies different focusing parameters to positive and negative examples.
    """

    def __init__(
            self,
            gamma_pos: float = 0.0,
            gamma_neg: float = 4.0,
            beta: float = 0.0,
            weight: Optional[torch.Tensor] = None,
            reduction: str = 'mean',
            eps: float = 1e-8  # Added epsilon for numerical stability
    ):
        """
        Initialize Asymmetric Loss.

        Args:
            gamma_pos: Focusing parameter for positive examples.
            gamma_neg: Focusing parameter for negative examples.
            beta: Threshold for shifting the negative loss.
            weight: Optional tensor of class weights.
            reduction: Reduction method ('mean', 'sum', or 'none').
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.beta = beta
        self.weight = weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input: Predicted logits of shape [batch_size, num_classes].
            target: Target labels of shape [batch_size, num_classes].

        Returns:
            Loss tensor.
        """
        # Apply sigmoid to get probabilities
        input_prob = torch.sigmoid(input)

        # Separate positive and negative examples
        pos_mask = target == 1
        neg_mask = target == 0

        # Positive examples
        pos_loss = torch.zeros_like(input_prob)
        if pos_mask.sum() > 0:
            pos_weight = (1 - input_prob[pos_mask]) ** self.gamma_pos
            pos_loss[pos_mask] = -pos_weight * torch.log(input_prob[pos_mask] + self.eps)

        # Negative examples
        neg_loss = torch.zeros_like(input_prob)
        if neg_mask.sum() > 0:
            # Apply beta threshold
            input_prob_neg = input_prob[neg_mask]
            input_prob_neg = torch.where(input_prob_neg < self.beta, torch.zeros_like(input_prob_neg),
                                         input_prob_neg - self.beta)
            # Scale input_prob_neg by (1-beta) to ensure values fall between 0 and 1
            input_prob_neg = input_prob_neg / (1 - self.beta)

            neg_weight = input_prob_neg ** self.gamma_neg
            neg_loss[neg_mask] = -neg_weight * torch.log(1 - input_prob_neg + self.eps)

        # Combine losses
        loss = pos_loss + neg_loss

        # Apply class weights if provided
        if self.weight is not None:
            # Expand weights to match batch size
            weight = self.weight.unsqueeze(0).expand_as(target)
            loss = loss * weight

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss for Multi-Label Classification.

    This loss addresses class imbalance by reweighting based on the effective
    number of samples for each class.
    """

    def __init__(
            self,
            samples_per_class: List[int],
            beta: float = 0.9999,
            base_loss: nn.Module = None,
            reduction: str = 'mean'
    ):
        """
        Initialize Class-Balanced Loss.

        Args:
            samples_per_class: Number of samples for each class.
            beta: Hyperparameter for computing effective number of samples.
            base_loss: Base loss function to be reweighted.
            reduction: Reduction method ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.samples_per_class = samples_per_class
        self.beta = beta
        self.reduction = reduction

        # Compute effective number of samples for each class
        effective_num = 1.0 - np.power(beta, samples_per_class)
        # Compute weights based on effective number
        weights = (1.0 - beta) / np.array(effective_num)
        # Normalize weights
        weights = weights / np.sum(weights) * len(weights)

        # Convert weights to tensor
        self.class_weights = torch.tensor(weights, dtype=torch.float32)

        # Set base loss
        if base_loss is None:
            self.base_loss = WeightedBinaryCrossEntropyLoss(weight=None, reduction='none')
        else:
            self.base_loss = base_loss

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input: Predicted logits of shape [batch_size, num_classes].
            target: Target labels of shape [batch_size, num_classes].

        Returns:
            Loss tensor.
        """
        # Move class weights to input device
        weights = self.class_weights.to(input.device)

        # Compute base loss (returns per-element loss)
        loss = self.base_loss(input, target)

        # Apply class-balancing weights
        if loss.ndim > 1:
            # For per-element loss (shape [batch_size, num_classes])
            weights = weights.unsqueeze(0).expand_as(loss)
            loss = loss * weights

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class MultiComponentLoss(nn.Module):
    """
    Multi-Component Loss for GRAFT framework.

    This loss combines multiple loss functions with learnable weights.
    """

    def __init__(
            self,
            num_classes: int,
            class_weights: Optional[torch.Tensor] = None,
            samples_per_class: Optional[List[int]] = None,
            wbce_weight: float = 1.0,
            focal_weight: float = 1.0,
            asl_weight: float = 1.0,
            cb_weight: float = 1.0,  # Added class-balanced loss weight
            focal_gamma: float = 2.0,
            focal_alpha: float = 0.25,  # Added alpha parameter for focal loss
            asl_gamma_neg: float = 4.0,
            asl_beta: float = 0.0,
            learn_weights: bool = True
    ):
        """
        Initialize Multi-Component Loss.

        Args:
            num_classes: Number of classes.
            class_weights: Optional tensor of class weights.
            samples_per_class: Optional list of sample counts per class.
            wbce_weight: Weight for Weighted BCE Loss.
            focal_weight: Weight for Focal Loss.
            asl_weight: Weight for Asymmetric Loss.
            cb_weight: Weight for Class-Balanced Loss.
            focal_gamma: Focusing parameter for Focal Loss.
            focal_alpha: Alpha balancing parameter for Focal Loss.
            asl_gamma_neg: Focusing parameter for negative examples in Asymmetric Loss.
            asl_beta: Threshold for shifting negative loss in Asymmetric Loss.
            learn_weights: Whether to learn loss component weights.
        """
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.learn_weights = learn_weights

        # Compute positive class weights for BCEWithLogitsLoss based on class distribution
        pos_weight = None
        if samples_per_class is not None:
            # Convert to numpy array for operations
            samples = np.array(samples_per_class)
            # Compute pos_weight as inverse of frequency
            pos_weight = torch.tensor(np.sum(samples) / (num_classes * samples), dtype=torch.float32)

        # Initialize loss components
        self.wbce_loss = WeightedBinaryCrossEntropyLoss(
            weight=class_weights,
            reduction='mean',
            pos_weight=pos_weight
        )

        self.focal_loss = FocalLoss(
            gamma=focal_gamma,
            alpha=focal_alpha,
            weight=class_weights,
            reduction='mean'
        )

        self.asl_loss = AsymmetricLoss(
            gamma_pos=0.0,
            gamma_neg=asl_gamma_neg,
            beta=asl_beta,
            weight=class_weights,
            reduction='mean'
        )

        # Add Class-Balanced Loss
        if samples_per_class is not None:
            self.cb_loss = ClassBalancedLoss(
                samples_per_class=samples_per_class,
                beta=0.9999,
                base_loss=WeightedBinaryCrossEntropyLoss(reduction='none'),
                reduction='mean'
            )
            self.has_cb_loss = True
        else:
            self.has_cb_loss = False

        # Initialize loss weights
        if learn_weights:
            # Use log space to ensure positivity
            self.log_wbce_weight = nn.Parameter(torch.tensor(np.log(wbce_weight)))
            self.log_focal_weight = nn.Parameter(torch.tensor(np.log(focal_weight)))
            self.log_asl_weight = nn.Parameter(torch.tensor(np.log(asl_weight)))
            if self.has_cb_loss:
                self.log_cb_weight = nn.Parameter(torch.tensor(np.log(cb_weight)))
        else:
            self.register_buffer("wbce_weight", torch.tensor(wbce_weight))
            self.register_buffer("focal_weight", torch.tensor(focal_weight))
            self.register_buffer("asl_weight", torch.tensor(asl_weight))
            if self.has_cb_loss:
                self.register_buffer("cb_weight", torch.tensor(cb_weight))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input: Predicted logits of shape [batch_size, num_classes].
            target: Target labels of shape [batch_size, num_classes].

        Returns:
            Combined loss tensor.
        """
        # Compute individual losses
        wbce = self.wbce_loss(input, target)
        focal = self.focal_loss(input, target)
        asl = self.asl_loss(input, target)

        if self.has_cb_loss:
            cb = self.cb_loss(input, target)

        # Get loss weights
        if hasattr(self, "learn_weights") and self.learn_weights:
            wbce_weight = torch.exp(self.log_wbce_weight)
            focal_weight = torch.exp(self.log_focal_weight)
            asl_weight = torch.exp(self.log_asl_weight)
            if self.has_cb_loss:
                cb_weight = torch.exp(self.log_cb_weight)
        else:
            wbce_weight = self.wbce_weight
            focal_weight = self.focal_weight
            asl_weight = self.asl_weight
            if self.has_cb_loss:
                cb_weight = self.cb_weight

        # Combine losses with weights
        if self.has_cb_loss:
            total_loss = wbce_weight * wbce + focal_weight * focal + asl_weight * asl + cb_weight * cb
        else:
            total_loss = wbce_weight * wbce + focal_weight * focal + asl_weight * asl

        return total_loss


def create_loss_function(
        config: dict,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
        samples_per_class: Optional[List[int]] = None
) -> nn.Module:
    """
    Create a loss function based on configuration.

    Args:
        config: Loss configuration dictionary.
        num_classes: Number of classes.
        class_weights: Optional tensor of class weights.
        samples_per_class: Optional list of sample counts per class.

    Returns:
        Loss function module.
    """
    wbce_weight = config.get("wbce_weight", 1.0)
    focal_weight = config.get("focal_weight", 1.0)
    asl_weight = config.get("asl_weight", 1.0)
    cb_weight = config.get("cb_weight", 1.0)  # Added class-balanced loss weight
    focal_gamma = config.get("focal_gamma", 2.0)
    focal_alpha = config.get("focal_alpha", 0.25)  # Added alpha parameter
    asl_gamma_neg = config.get("asl_gamma_neg", 4.0)
    asl_beta = config.get("asl_beta", 0.0)

    return MultiComponentLoss(
        num_classes=num_classes,
        class_weights=class_weights,
        samples_per_class=samples_per_class,
        wbce_weight=wbce_weight,
        focal_weight=focal_weight,
        asl_weight=asl_weight,
        cb_weight=cb_weight,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        asl_gamma_neg=asl_gamma_neg,
        asl_beta=asl_beta,
        learn_weights=True
    )

