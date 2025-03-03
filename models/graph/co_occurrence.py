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
            scaling_factor: float = 5.0
    ):
        """
        Initialize Co-occurrence Graph.

        Args:
            num_classes: Number of classes/labels.
            co_occurrence_matrix: Pre-computed co-occurrence matrix.
            context_types: Number of context types for grouping.
            context_similarity_threshold: Threshold for context similarity.
            scaling_factor: Scaling factor for confidence calculation.
        """
        super().__init__()
        self.num_classes = num_classes
        self.context_types = context_types
        self.context_similarity_threshold = context_similarity_threshold
        self.scaling_factor = scaling_factor

        # Initialize adjacency matrix
        if co_occurrence_matrix is not None:
            self.register_buffer("co_occurrence", co_occurrence_matrix)
        else:
            self.register_buffer("co_occurrence", torch.zeros(num_classes, num_classes))

        # Initialize context embedding (learnable)
        self.context_embeddings = nn.Parameter(torch.randn(num_classes, context_types))
        nn.init.xavier_uniform_(self.context_embeddings)

        # Class counts (will be set during update)
        self.register_buffer("class_counts", torch.ones(num_classes))

        # Flag to track whether the graph has been built
        self.graph_built = False

        # Edge weights
        self.register_buffer("edge_weights", torch.zeros(num_classes, num_classes))

    def build_graph(self):
        """
        Build the co-occurrence graph from statistics.
        """
        if not self.graph_built and torch.any(self.co_occurrence > 0):
            # Compute class balance statistics
            total_samples = torch.sum(self.class_counts)
            max_count = torch.max(self.class_counts)
            avg_count = torch.mean(self.class_counts)

            # Add small epsilon to avoid division by zero
            epsilon = 1e-8

            # Normalize co-occurrence
            normalized_co_occurrence = torch.zeros_like(self.co_occurrence)
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i != j and self.class_counts[i] > 0 and self.class_counts[j] > 0:
                        # Normalized co-occurrence
                        normalized_co_occurrence[i, j] = self.co_occurrence[i, j] / torch.sqrt(
                            (self.class_counts[i] + epsilon) * (self.class_counts[j] + epsilon))

            # Compute context affinity
            context_affinity = torch.zeros_like(self.co_occurrence)

            # Normalize context embeddings
            norm_embeddings = F.normalize(self.context_embeddings, p=2, dim=1)

            # Compute context similarity
            context_sim = torch.mm(norm_embeddings, norm_embeddings.t())

            # Apply threshold
            context_aff_mask = (context_sim > self.context_similarity_threshold).float()

            # Compute context affinity
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i != j:
                        # Context affinity based on similarity in embedding space
                        context_affinity[i, j] = (context_sim[i, j] * context_aff_mask[i, j]).item()

            # Class balance correction for PASCAL VOC
            # This addresses the imbalance between common and rare classes
            class_balance_correction = torch.zeros_like(self.co_occurrence)
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i != j:
                        # Class balance correction factor
                        # Boost relationships involving rare classes
                        min_count = min(self.class_counts[i].item(), self.class_counts[j].item())
                        max_count = max(self.class_counts[i].item(), self.class_counts[j].item())

                        # Calculate balance factor to boost rare class relationships
                        if min_count < 0.1 * avg_count:  # Very rare classes
                            balance_factor = 1.5  # Higher boosting for very rare classes
                        elif min_count < 0.3 * avg_count:  # Moderately rare classes
                            balance_factor = 1.3  # Medium boosting for moderately rare classes
                        else:  # Common classes
                            balance_factor = 1.0  # No boosting for common classes

                        # Additional adjustment based on ratio between min and max count
                        ratio_factor = (min_count + epsilon) / (max_count + epsilon)
                        balance_factor *= (0.5 + 0.5 * ratio_factor)  # Balance based on class ratio

                        class_balance_correction[i, j] = balance_factor

            # Apply confidence weighting
            confidence = torch.zeros_like(self.co_occurrence)
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i != j:
                        # Confidence based on number of co-occurrences
                        # More co-occurrences -> higher confidence
                        # Using a sigmoid-like function for smoother scaling
                        confidence[i, j] = 1.0 - torch.exp(-self.co_occurrence[i, j] / self.scaling_factor)

            # Compute final edge weights with additional sparsity for PASCAL VOC
            raw_weights = normalized_co_occurrence * context_affinity * class_balance_correction * confidence

            # Apply threshold to enforce sparsity
            # Keep only top connections for each class
            sparse_weights = torch.zeros_like(raw_weights)
            for i in range(self.num_classes):
                # Get non-diagonal weights for this class
                class_weights = raw_weights[i, :]
                class_weights[i] = 0  # Ignore self-connection

                # Keep only top connections (adaptive based on class frequency)
                if self.class_counts[i] > avg_count:
                    k = 5  # More connections for common classes
                else:
                    k = 3  # Fewer connections for rare classes

                # Get top-k indices
                if torch.sum(class_weights > 0) > k:
                    topk_values, topk_indices = torch.topk(class_weights, k)
                    # Set weights for top-k connections
                    for idx in topk_indices:
                        sparse_weights[i, idx] = raw_weights[i, idx]
                else:
                    # Keep all positive connections if fewer than k
                    sparse_weights[i, :] = class_weights * (class_weights > 0).float()

            # Make symmetric
            self.edge_weights = 0.5 * (sparse_weights + sparse_weights.t())

            # Normalize
            max_weight = torch.max(self.edge_weights)
            if max_weight > 0:
                self.edge_weights = self.edge_weights / max_weight

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

            # Skip empty samples (no labels)
            if label_indices.numel() == 0:
                continue

            # Handle case where there's only one label
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

        # Skip connection
        x_out = x + x_transformed

        return x_out


def create_co_occurrence_graph(
        num_classes: int,
        co_occurrence_matrix: Optional[torch.Tensor] = None,
        context_types: int = 4,
        context_similarity_threshold: float = 0.5,
        scaling_factor: float = 5.0
) -> CoOccurrenceGraph:
    """
    Create a Co-occurrence Graph module.

    Args:
        num_classes: Number of classes/labels.
        co_occurrence_matrix: Pre-computed co-occurrence matrix.
        context_types: Number of context types for grouping.
        context_similarity_threshold: Threshold for context similarity.
        scaling_factor: Scaling factor for confidence calculation.

    Returns:
        CoOccurrenceGraph module.
    """
    return CoOccurrenceGraph(
        num_classes=num_classes,
        co_occurrence_matrix=co_occurrence_matrix,
        context_types=context_types,
        context_similarity_threshold=context_similarity_threshold,
        scaling_factor=scaling_factor
    )

