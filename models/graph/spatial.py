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
from scipy.optimize import linear_sum_assignment


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
            spatial_statistics: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Initialize Spatial Relationship Graph.

        Args:
            num_classes: Number of classes/labels.
            scales: List of grid scales for multi-scale representation.
            scale_weights: Weights for each scale.
            positional_adjustment: Factor for adjusting positional relationships.
            spatial_statistics: Pre-computed spatial statistics.
        """
        super().__init__()
        self.num_classes = num_classes
        self.scales = scales
        self.scale_weights = scale_weights
        self.positional_adjustment = positional_adjustment

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

        # Initialize multi-scale spatial distributions
        self.distributions = {}
        for scale in scales:
            self.register_buffer(f"distribution_{scale}", torch.zeros(num_classes, scale, scale))

        # Initialize edge weights
        self.register_buffer("edge_weights", torch.zeros(num_classes, num_classes))

        # Flag to track whether the graph has been built
        self.graph_built = False

        # Learnable position embedding for relative positions
        self.relative_pos_embedding = nn.Parameter(torch.randn(3, 3, 16))  # 3x3 for above/below/same, left/right/same
        nn.init.xavier_uniform_(self.relative_pos_embedding)

        # Learnable position embedding for overlap
        self.overlap_embedding = nn.Parameter(torch.randn(16))
        nn.init.xavier_uniform_(self.overlap_embedding)

        # Position projection layer
        self.pos_proj = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
            if torch.all(self.sizes[c] == 0):
                # No spatial information for this class
                # Create a uniform distribution
                distribution[c] = 1.0 / (scale * scale)
                continue

            # Get position and size
            pos_x, pos_y = self.positions[c]
            size_x, size_y = self.sizes[c]

            # Create grid
            x = torch.linspace(0, 1, scale, device=self.positions.device)
            y = torch.linspace(0, 1, scale, device=self.positions.device)
            grid_y, grid_x = torch.meshgrid(y, x)

            # Compute gaussian distribution
            sigma_x = max(0.01, size_x / 4)  # Scale sigma based on object size
            sigma_y = max(0.01, size_y / 4)

            dist_x = -((grid_x - pos_x) ** 2) / (2 * sigma_x ** 2)
            dist_y = -((grid_y - pos_y) ** 2) / (2 * sigma_y ** 2)
            dist = torch.exp(dist_x + dist_y)

            # Normalize
            if torch.sum(dist) > 0:
                dist = dist / torch.sum(dist)

            distribution[c] = dist

        return distribution

    def _earth_movers_distance(self, dist1: torch.Tensor, dist2: torch.Tensor) -> float:
        """
        Compute Earth Mover's Distance (EMD) between two distributions.

        Args:
            dist1: First distribution of shape [H, W].
            dist2: Second distribution of shape [H, W].

        Returns:
            EMD as a float.
        """
        # Convert to numpy for scipy solver
        dist1_np = dist1.detach().cpu().numpy()
        dist2_np = dist2.detach().cpu().numpy()

        # Flatten distributions
        dist1_flat = dist1_np.flatten()
        dist2_flat = dist2_np.flatten()

        # Ensure distributions sum to 1
        if np.sum(dist1_flat) > 0:
            dist1_flat = dist1_flat / np.sum(dist1_flat)
        if np.sum(dist2_flat) > 0:
            dist2_flat = dist2_flat / np.sum(dist2_flat)

        # Compute cost matrix
        n = len(dist1_flat)
        cost_matrix = np.zeros((n, n))

        h, w = dist1.shape
        for i in range(h):
            for j in range(w):
                for k in range(h):
                    for l in range(w):
                        i_idx = i * w + j
                        j_idx = k * w + l
                        # Euclidean distance in grid space
                        cost_matrix[i_idx, j_idx] = np.sqrt((i - k) ** 2 + (j - l) ** 2) / np.sqrt(h ** 2 + w ** 2)

        # Solve EMD using linear_sum_assignment
        # This is a simplified version of EMD, assuming equal weights
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        emd = cost_matrix[row_idx, col_idx].sum() / len(row_idx)

        return float(emd)

    def build_graph(self):
        """
        Build the spatial relationship graph.
        """
        if not self.graph_built:
            # Compute multi-scale distributions
            for scale in self.scales:
                distribution = self._compute_spatial_distribution(scale)
                self.register_buffer(f"distribution_{scale}", distribution)

            # Compute edge weights based on Earth Mover's Distance at multiple scales
            edge_weights = torch.zeros(self.num_classes, self.num_classes, device=self.positions.device)

            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i != j:
                        emd_sum = 0.0

                        # Compute EMD at each scale
                        for scale_idx, scale in enumerate(self.scales):
                            dist_i = getattr(self, f"distribution_{scale}")[i]
                            dist_j = getattr(self, f"distribution_{scale}")[j]

                            # Compute EMD
                            emd = self._earth_movers_distance(dist_i, dist_j)

                            # Lower EMD means higher similarity
                            similarity = 1.0 - emd

                            # Apply scale weight
                            emd_sum += self.scale_weights[scale_idx] * similarity

                        # Add positional relationship
                        pos_i, pos_j = self.positions[i], self.positions[j]
                        size_i, size_j = self.sizes[i], self.sizes[j]

                        # Compute relative position
                        rel_x = 1 if pos_i[0] < pos_j[0] else (2 if pos_i[0] > pos_j[0] else 0)
                        rel_y = 1 if pos_i[1] < pos_j[1] else (2 if pos_i[1] > pos_j[1] else 0)

                        # Get embedding for relative position
                        pos_emb = self.relative_pos_embedding[rel_y, rel_x]

                        # Get embedding for overlap
                        overlap = self.overlaps[i, j]
                        overlap_emb = self.overlap_embedding * overlap

                        # Combine embeddings
                        combined_emb = pos_emb + overlap_emb

                        # Project to edge weight adjustment
                        pos_adjustment = torch.sigmoid(self.pos_proj(combined_emb))

                        # Apply adjustment
                        edge_weights[i, j] = emd_sum * (1.0 + self.positional_adjustment * pos_adjustment)

            # Normalize edge weights
            max_weight = torch.max(edge_weights)
            if max_weight > 0:
                edge_weights = edge_weights / max_weight

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
            # Update position (running average)
            old_count = self.sizes[cls_idx, 0] > 0
            if old_count:
                # Update with running average
                self.positions[cls_idx] = (self.positions[cls_idx] + batch_positions[i]) / 2
                self.sizes[cls_idx] = (self.sizes[cls_idx] + batch_sizes[i]) / 2
            else:
                # Initialize
                self.positions[cls_idx] = batch_positions[i]
                self.sizes[cls_idx] = batch_sizes[i]

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
                    self.overlaps[cls_i, cls_j] = (self.overlaps[cls_i, cls_j] + iou) / 2
                    self.overlaps[cls_j, cls_i] = self.overlaps[cls_i, cls_j]

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


def create_spatial_relationship_graph(
        num_classes: int,
        scales: List[int] = [5, 15, 25],
        scale_weights: List[float] = [0.2, 0.3, 0.5],
        positional_adjustment: float = 0.2,
        spatial_statistics: Optional[Dict[str, torch.Tensor]] = None
) -> SpatialRelationshipGraph:
    """
    Create a Spatial Relationship Graph module.

    Args:
        num_classes: Number of classes/labels.
        scales: List of grid scales for multi-scale representation.
        scale_weights: Weights for each scale.
        positional_adjustment: Factor for adjusting positional relationships.
        spatial_statistics: Pre-computed spatial statistics.

    Returns:
        SpatialRelationshipGraph module.
    """
    return SpatialRelationshipGraph(
        num_classes=num_classes,
        scales=scales,
        scale_weights=scale_weights,
        positional_adjustment=positional_adjustment,
        spatial_statistics=spatial_statistics
    )

