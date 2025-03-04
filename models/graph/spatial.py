"""
Spatial relationship graph module for GRAFT framework.

This module implements the multi-scale spatial relationship graph
component that captures label relationships based on their spatial
arrangements in the dataset.
"""
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialRelationshipGraph(nn.Module):
    """
    Multi-Scale Spatial Relationship Graph module.

    This graph captures the spatial relationships between labels
    based on their relative positions and overlaps in the dataset.
    """

    def __init__(
            self,
            num_classes: int,
            scales: List[int] = [5, 15, 25],
            scale_weights: List[float] = [0.2, 0.3, 0.5],
            positional_adjustment: float = 0.2,
            spatial_statistics: Optional[Dict[str, torch.Tensor]] = None,
            smoothing_factor: float = 0.01
    ):
        """
        Initialize Spatial Relationship Graph.

        Args:
            num_classes: Number of classes/labels.
            scales: List of grid scales for multi-scale representation.
            scale_weights: Weights for each scale.
            positional_adjustment: Factor for adjusting positional relationships.
            spatial_statistics: Pre-computed spatial statistics.
            smoothing_factor: Small value to avoid division by zero.
        """
        super().__init__()
        self.num_classes = num_classes
        self.scales = scales
        self.scale_weights = scale_weights
        self.positional_adjustment = positional_adjustment
        self.smoothing_factor = smoothing_factor

        assert len(scales) == len(scale_weights), "Scales and weights must have the same length"
        assert sum(scale_weights) == 1.0, "Scale weights must sum to 1.0"

        # Initialize spatial statistics
        if spatial_statistics is not None:
            self.register_buffer("positions", spatial_statistics["positions"])
            self.register_buffer("sizes", spatial_statistics["sizes"])
            self.register_buffer("overlaps", spatial_statistics["overlaps"])
        else:
            self.register_buffer("positions", torch.zeros(num_classes, 2))  # [num_classes, 2]
            self.register_buffer("sizes", torch.zeros(num_classes, 2))  # [num_classes, 2]
            self.register_buffer("overlaps", torch.zeros(num_classes, num_classes))  # [num_classes, num_classes]

        # Track number of observations for each class
        self.register_buffer("position_counts", torch.zeros(num_classes))

        # Initialize multi-scale spatial distributions
        for scale in scales:
            self.register_buffer(f"distribution_{scale}", torch.zeros(num_classes, scale, scale))

        # Initialize edge weights
        self.register_buffer("edge_weights", torch.zeros(num_classes, num_classes))

        # Flag to track whether the graph has been built
        self.graph_built = False

        # Learnable position embedding for relative positions
        self.relative_pos_embedding = nn.Parameter(torch.randn(3, 3, 16))  # 3x3 for above/below/same, left/right/same
        nn.init.xavier_uniform_(self.relative_pos_embedding)

        # Position projection layer
        self.pos_proj = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def _compute_spatial_distribution(self, scale: int):
        """
        Compute spatial distribution for a specific scale.

        Args:
            scale: Grid scale.
        """
        distribution = torch.zeros(self.num_classes, scale, scale, device=self.positions.device)

        # For each class, generate a distribution based on position and size
        for c in range(self.num_classes):
            if self.position_counts[c] > 0:
                # Get normalized position and size
                pos_x, pos_y = self.positions[c] / self.position_counts[c]
                size_x, size_y = self.sizes[c] / self.position_counts[c]

                # Create grid
                x = torch.linspace(0, 1, scale, device=self.positions.device)
                y = torch.linspace(0, 1, scale, device=self.positions.device)
                grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

                # Compute gaussian distribution
                sigma_x = max(0.05, size_x / 4)  # Scale sigma based on object size
                sigma_y = max(0.05, size_y / 4)

                # Avoid zero/nan values
                if torch.abs(pos_x) < 1e-5 and torch.abs(pos_y) < 1e-5:
                    # If no valid position data, create uniform distribution
                    distribution[c] = 1.0 / (scale * scale)
                else:
                    dist_x = -((grid_x - pos_x) ** 2) / (2 * sigma_x ** 2 + self.smoothing_factor)
                    dist_y = -((grid_y - pos_y) ** 2) / (2 * sigma_y ** 2 + self.smoothing_factor)
                    dist = torch.exp(dist_x + dist_y)

                    # Normalize
                    if torch.sum(dist) > 0:
                        dist = dist / torch.sum(dist)
                    else:
                        dist = torch.ones_like(dist) / (scale * scale)

                    distribution[c] = dist
            else:
                # No spatial information for this class
                # Create uniform distribution
                distribution[c] = 1.0 / (scale * scale)

        return distribution

    def _compute_similarity(self, dist1: torch.Tensor, dist2: torch.Tensor) -> float:
        """
        Compute similarity between two spatial distributions.

        Args:
            dist1: First distribution of shape [H, W].
            dist2: Second distribution of shape [H, W].

        Returns:
            Similarity score (higher means more similar).
        """
        # Flatten distributions
        dist1_flat = dist1.flatten()
        dist2_flat = dist2.flatten()

        # Compute cosine similarity
        similarity = F.cosine_similarity(dist1_flat.unsqueeze(0), dist2_flat.unsqueeze(0), dim=1)

        return similarity.item()

    def build_graph(self):
        """
        Build the spatial relationship graph.
        """
        if not self.graph_built:
            # Compute multi-scale distributions
            for scale in self.scales:
                distribution = self._compute_spatial_distribution(scale)
                self.register_buffer(f"distribution_{scale}", distribution)

            # Compute edge weights based on multi-scale distribution similarity
            edge_weights = torch.zeros(self.num_classes, self.num_classes, device=self.positions.device)

            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i != j:
                        similarity_sum = 0.0

                        # Compute similarity at each scale
                        for scale_idx, scale in enumerate(self.scales):
                            dist_i = getattr(self, f"distribution_{scale}")[i]
                            dist_j = getattr(self, f"distribution_{scale}")[j]

                            # Compute similarity
                            similarity = self._compute_similarity(dist_i, dist_j)

                            # Apply scale weight
                            similarity_sum += self.scale_weights[scale_idx] * similarity

                        # Add positional relationship from overlaps if available
                        if self.position_counts[i] > 0 and self.position_counts[j] > 0:
                            # Normalize positions
                            pos_i = self.positions[i] / self.position_counts[i]
                            pos_j = self.positions[j] / self.position_counts[j]

                            # Compute relative position
                            rel_x = 1 if pos_i[0] < pos_j[0] else (2 if pos_i[0] > pos_j[0] else 0)
                            rel_y = 1 if pos_i[1] < pos_j[1] else (2 if pos_i[1] > pos_j[1] else 0)

                            # Get embedding for relative position
                            pos_emb = self.relative_pos_embedding[rel_y, rel_x]

                            # Project to edge weight adjustment
                            pos_adjustment = self.pos_proj(pos_emb)

                            # Apply adjustment
                            edge_weights[i, j] = similarity_sum * (1.0 + self.positional_adjustment * pos_adjustment)
                        else:
                            edge_weights[i, j] = similarity_sum

            # Apply row-wise softmax to create proper probability distributions
            edge_weights = F.softmax(edge_weights * 5.0, dim=1)  # Temperature scaling for sharper distribution

            # Add self-loops with small weight
            eye = torch.eye(self.num_classes, device=edge_weights.device)
            edge_weights = edge_weights * (1 - 0.1) + eye * 0.1

            self.edge_weights = edge_weights
            self.graph_built = True

    def update_statistics(self, batch_positions: torch.Tensor, batch_sizes: torch.Tensor, batch_classes: torch.Tensor):
        """
        Update spatial statistics from batch of object detections.

        Args:
            batch_positions: Positions of objects [batch_size, 2].
            batch_sizes: Sizes of objects [batch_size, 2].
            batch_classes: Class indices of objects [batch_size].
        """
        # For each class, update position and size
        for i, cls_idx in enumerate(batch_classes):
            # Update position counts
            self.position_counts[cls_idx] += 1.0

            # Update position and size (accumulate)
            self.positions[cls_idx] += batch_positions[i]
            self.sizes[cls_idx] += batch_sizes[i]

        # Update overlap statistics
        # Group by image
        # This is a simplification - in practice, you'd need to track which objects are from the same image
        for i in range(len(batch_classes)):
            for j in range(i + 1, len(batch_classes)):
                cls_i = batch_classes[i]
                cls_j = batch_classes[j]

                # Compute overlap (IoU)
                pos_i, size_i = batch_positions[i], batch_sizes[i]
                pos_j, size_j = batch_positions[j], batch_sizes[j]

                # Convert to box coordinates
                box_i = [
                    pos_i[0] - size_i[0] / 2,
                    pos_i[1] - size_i[1] / 2,
                    pos_i[0] + size_i[0] / 2,
                    pos_i[1] + size_i[1] / 2
                ]

                box_j = [
                    pos_j[0] - size_j[0] / 2,
                    pos_j[1] - size_j[1] / 2,
                    pos_j[0] + size_j[0] / 2,
                    pos_j[1] + size_j[1] / 2
                ]

                # Compute intersection
                x_min = max(box_i[0], box_j[0])
                y_min = max(box_i[1], box_j[1])
                x_max = min(box_i[2], box_j[2])
                y_max = min(box_i[3], box_j[3])

                if x_max > x_min and y_max > y_min:
                    intersection = (x_max - x_min) * (y_max - y_min)
                    area_i = size_i[0] * size_i[1]
                    area_j = size_j[0] * size_j[1]
                    union = area_i + area_j - intersection
                    iou = intersection / union if union > 0 else 0

                    # Update overlap
                    self.overlaps[cls_i, cls_j] += iou
                    self.overlaps[cls_j, cls_i] += iou

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


def create_spatial_relationship_graph(
        num_classes: int,
        scales: List[int] = [5, 15, 25],
        scale_weights: List[float] = [0.2, 0.3, 0.5],
        positional_adjustment: float = 0.2,
        spatial_statistics: Optional[Dict[str, torch.Tensor]] = None,
        smoothing_factor: float = 0.01
) -> SpatialRelationshipGraph:
    """
    Create a Spatial Relationship Graph module.

    Args:
        num_classes: Number of classes/labels.
        scales: List of grid scales for multi-scale representation.
        scale_weights: Weights for each scale.
        positional_adjustment: Factor for adjusting positional relationships.
        spatial_statistics: Pre-computed spatial statistics.
        smoothing_factor: Small value to avoid division by zero.

    Returns:
        SpatialRelationshipGraph module.
    """
    return SpatialRelationshipGraph(
        num_classes=num_classes,
        scales=scales,
        scale_weights=scale_weights,
        positional_adjustment=positional_adjustment,
        spatial_statistics=spatial_statistics,
        smoothing_factor=smoothing_factor
    )

