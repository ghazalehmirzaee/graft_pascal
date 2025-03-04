"""
Co-occurrence graph module for GRAFT framework.

This module implements the statistical co-occurrence graph component
that captures label relationships based on their frequency of co-occurrence
in the dataset.
"""
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CoOccurrenceGraph(nn.Module):
    """
    Statistical Co-occurrence Graph module.

    This graph captures the statistical relationships between labels
    based on their co-occurrence patterns in the training dataset.
    """

    def __init__(
            self,
            num_classes: int,
            co_occurrence_matrix: Optional[torch.Tensor] = None,
            context_types: int = 4,
            context_similarity_threshold: float = 0.5,
            scaling_factor: float = 5.0,
            smoothing_factor: float = 0.01
    ):
        """
        Initialize Co-occurrence Graph.

        Args:
            num_classes: Number of classes/labels.
            co_occurrence_matrix: Pre-computed co-occurrence matrix.
            context_types: Number of context types for grouping.
            context_similarity_threshold: Threshold for context similarity.
            scaling_factor: Scaling factor for confidence calculation.
            smoothing_factor: Add to denominators to prevent div by zero.
        """
        super().__init__()
        self.num_classes = num_classes
        self.context_types = context_types
        self.context_similarity_threshold = context_similarity_threshold
        self.scaling_factor = scaling_factor
        self.smoothing_factor = smoothing_factor

        # Initialize adjacency matrix
        if co_occurrence_matrix is not None:
            self.register_buffer("co_occurrence", co_occurrence_matrix)
        else:
            self.register_buffer("co_occurrence", torch.zeros(num_classes, num_classes))

        # Initialize context embedding (learnable)
        self.context_embeddings = nn.Parameter(torch.randn(num_classes, context_types))
        nn.init.xavier_uniform_(self.context_embeddings)

        # Class counts (will be set during update)
        self.register_buffer("class_counts", torch.ones(num_classes) * self.smoothing_factor)

        # Flag to track whether the graph has been built
        self.graph_built = False

        # Edge weights
        self.register_buffer("edge_weights", torch.zeros(num_classes, num_classes))

    def build_graph(self):
        """
        Build the co-occurrence graph from statistics.
        """
        if not self.graph_built or torch.any(self.co_occurrence > 0):
            # Compute class balance statistics
            total_samples = torch.sum(self.class_counts)
            max_count = torch.max(self.class_counts)
            avg_count = torch.mean(self.class_counts)

            # Normalize co-occurrence with Laplace smoothing
            normalized_co_occurrence = torch.zeros_like(self.co_occurrence)
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i != j:
                        # Normalized co-occurrence with smoothing
                        normalized_co_occurrence[i, j] = (self.co_occurrence[i, j] + self.smoothing_factor) / (
                            torch.sqrt((self.class_counts[i] + self.smoothing_factor) *
                                       (self.class_counts[j] + self.smoothing_factor))
                        )

            # Compute context affinity
            context_affinity = torch.zeros_like(self.co_occurrence)

            # Normalize context embeddings
            norm_embeddings = F.normalize(self.context_embeddings, p=2, dim=1)

            # Compute context similarity
            context_sim = torch.mm(norm_embeddings, norm_embeddings.t())

            # Apply threshold with smooth transition
            context_aff_mask = torch.sigmoid((context_sim - self.context_similarity_threshold) * 10)

            # Compute context affinity
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i != j:
                        # Context affinity based on similarity in embedding space
                        context_affinity[i, j] = (context_sim[i, j] * context_aff_mask[i, j]).item()

            # Class balance correction with improved weighting for rare classes
            class_balance_correction = torch.zeros_like(self.co_occurrence)
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i != j:
                        # Class balance correction factor
                        # Boost relationships involving rare classes
                        min_count = min(self.class_counts[i], self.class_counts[j])
                        max_count = max(self.class_counts[i], self.class_counts[j])

                        # Using logarithmic scaling to better handle class imbalance
                        if min_count > self.smoothing_factor and max_count > self.smoothing_factor:
                            balance_factor = torch.log1p(max_count / avg_count) * (min_count / max_count)
                            class_balance_correction[i, j] = balance_factor
                        else:
                            class_balance_correction[i, j] = self.smoothing_factor

            # Apply confidence weighting with sigmoid shaping for better scaling
            confidence = torch.zeros_like(self.co_occurrence)
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i != j:
                        # Confidence based on number of co-occurrences
                        # More co-occurrences -> higher confidence
                        # Using sigmoid to create a smoother confidence curve
                        confidence[i, j] = 2.0 / (
                                    1.0 + torch.exp(-self.co_occurrence[i, j] / self.scaling_factor)) - 1.0

            # Compute final edge weights with careful normalization
            self.edge_weights = normalized_co_occurrence * context_affinity * class_balance_correction * confidence

            # Apply row-wise softmax to create proper probability distributions
            self.edge_weights = F.softmax(self.edge_weights * 5.0,
                                          dim=1)  # Temperature scaling for sharper distribution

            # Add self-loops with small weight
            eye = torch.eye(self.num_classes, device=self.edge_weights.device)
            self.edge_weights = self.edge_weights * (1 - 0.1) + eye * 0.1

            self.graph_built = True

    def update_statistics(self, labels: torch.Tensor):
        """
        Update co-occurrence statistics from batch of labels.

        Args:
            labels: Multi-hot encoded labels of shape [batch_size, num_classes].
        """
        batch_size = labels.shape[0]

        # Update class counts
        class_counts_batch = torch.sum(labels, dim=0)
        self.class_counts += class_counts_batch

        # Update co-occurrence matrix
        for i in range(batch_size):
            label_indices = torch.nonzero(labels[i]).squeeze(-1)

            # Skip samples with no labels
            if label_indices.numel() == 0:
                continue

            # Handle single label case
            if label_indices.numel() == 1:
                continue

            # Update co-occurrence for each pair of labels
            for idx1 in label_indices:
                for idx2 in label_indices:
                    if idx1 != idx2:
                        self.co_occurrence[idx1, idx2] += 1

        # Mark graph as not built to trigger rebuild
        self.graph_built = False

    def get_adjacency_matrix(self) -> torch.Tensor:
        """
        Get the adjacency matrix of the graph.

        Returns:
            Adjacency matrix tensor.
        """
        if not self.graph_built:
            self.build_graph()

        return self.edge_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input node features of shape [batch_size, num_classes, feature_dim].

        Returns:
            Updated node features of same shape as input.
        """
        if not self.graph_built:
            self.build_graph()

        # Apply graph convolution
        # x: [batch_size, num_classes, feature_dim]
        # edge_weights: [num_classes, num_classes]
        x_transformed = torch.bmm(self.edge_weights.unsqueeze(0).expand(x.size(0), -1, -1), x)

        # Skip connection with gating mechanism
        gate = torch.sigmoid(torch.sum(x * x_transformed, dim=-1, keepdim=True) / np.sqrt(x.size(-1)))
        x_out = x * (1 - gate) + x_transformed * gate

        return x_out


def create_co_occurrence_graph(
        num_classes: int,
        co_occurrence_matrix: Optional[torch.Tensor] = None,
        context_types: int = 4,
        context_similarity_threshold: float = 0.5,
        scaling_factor: float = 5.0,
        smoothing_factor: float = 0.01
) -> CoOccurrenceGraph:
    """
    Create a Co-occurrence Graph module.

    Args:
        num_classes: Number of classes/labels.
        co_occurrence_matrix: Pre-computed co-occurrence matrix.
        context_types: Number of context types for grouping.
        context_similarity_threshold: Threshold for context similarity.
        scaling_factor: Scaling factor for confidence calculation.
        smoothing_factor: Add to denominators to prevent div by zero.

    Returns:
        CoOccurrenceGraph module.
    """
    return CoOccurrenceGraph(
        num_classes=num_classes,
        co_occurrence_matrix=co_occurrence_matrix,
        context_types=context_types,
        context_similarity_threshold=context_similarity_threshold,
        scaling_factor=scaling_factor,
        smoothing_factor=smoothing_factor
    )

