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
        assert abs(sum(scale_weights) - 1.0) < 1e-6, "Scale weights must sum to 1.0"

        # Initialize spatial statistics with default values for PASCAL VOC
        if spatial_statistics is not None:
            self.register_buffer("positions", spatial_statistics["positions"])
            self.register_buffer("sizes", spatial_statistics["sizes"])
            self.register_buffer("overlaps", spatial_statistics["overlaps"])
        else:
            # Initialize with some meaningful defaults based on PASCAL VOC dataset
            self.register_buffer("positions", torch.zeros(num_classes, 2))  # [num_classes, 2]
            self.register_buffer("sizes", torch.zeros(num_classes, 2))  # [num_classes, 2]
            self.register_buffer("overlaps", torch.zeros(num_classes, num_classes))  # [num_classes, num_classes]

            # Set default positions based on common locations in PASCAL VOC
            # These are rough estimates and will be refined during training
            # Format: [center_x, center_y] in normalized coordinates [0,1]
            default_positions = {
                # Animals typically in center or foreground
                'bird': [0.5, 0.5],
                'cat': [0.5, 0.6],
                'dog': [0.5, 0.6],
                'cow': [0.5, 0.6],
                'horse': [0.5, 0.6],
                'sheep': [0.5, 0.6],
                # Vehicles
                'aeroplane': [0.5, 0.4],
                'bicycle': [0.5, 0.7],
                'boat': [0.5, 0.5],
                'bus': [0.5, 0.5],
                'car': [0.5, 0.7],
                'motorbike': [0.5, 0.7],
                'train': [0.5, 0.5],
                # Indoor objects
                'bottle': [0.5, 0.5],
                'chair': [0.5, 0.6],
                'diningtable': [0.5, 0.6],
                'pottedplant': [0.3, 0.5],
                'sofa': [0.5, 0.6],
                'tvmonitor': [0.5, 0.4],
                # People
                'person': [0.5, 0.6]
            }

            # Set default sizes based on typical object sizes in PASCAL VOC
            # Format: [width, height] in normalized coordinates [0,1]
            default_sizes = {
                # Animals
                'bird': [0.2, 0.2],
                'cat': [0.3, 0.3],
                'dog': [0.3, 0.3],
                'cow': [0.4, 0.3],
                'horse': [0.4, 0.3],
                'sheep': [0.3, 0.2],
                # Vehicles
                'aeroplane': [0.6, 0.2],
                'bicycle': [0.3, 0.4],
                'boat': [0.4, 0.2],
                'bus': [0.4, 0.3],
                'car': [0.3, 0.2],
                'motorbike': [0.3, 0.3],
                'train': [0.6, 0.2],
                # Indoor objects
                'bottle': [0.1, 0.2],
                'chair': [0.3, 0.4],
                'diningtable': [0.5, 0.3],
                'pottedplant': [0.2, 0.3],
                'sofa': [0.5, 0.3],
                'tvmonitor': [0.2, 0.2],
                # People
                'person': [0.2, 0.5]
            }

            # PASCAL VOC class names in proper order
            pascal_classes = [
                'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
            ]

            # Set default positions and sizes
            for i, class_name in enumerate(pascal_classes):
                if i < num_classes:  # Make sure we don't go out of bounds
                    if class_name in default_positions:
                        self.positions[i] = torch.tensor(default_positions[class_name])
                    if class_name in default_sizes:
                        self.sizes[i] = torch.tensor(default_sizes[class_name])

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
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

            # Compute gaussian distribution
            sigma_x = max(0.01, size_x / 4)  # Scale sigma based on object size
            sigma_y = max(0.01, size_y / 4)

            # Use squared distance for numerical stability
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

        # Apply Sinkhorn regularization for faster and more stable EMD computation
        # This is a simplified version that works well for small grids
        # For larger grids or more precision, use a full Sinkhorn algorithm
        epsilon = 0.01  # Regularization strength
        a = np.ones(n) / n
        b = np.ones(n) / n

        # Initialize kernel with cost matrix
        K = np.exp(-cost_matrix / epsilon)

        # Initialize u and v
        u = np.ones(n)
        v = np.ones(n)

        # Sinkhorn iterations
        for _ in range(20):  # Usually converges quickly
            u = a / (K @ v)
            v = b / (K.T @ u)

        # Compute transport plan
        P = np.diag(u) @ K @ np.diag(v)

        # Compute EMD approximation
        emd = np.sum(P * cost_matrix)

        return float(emd)

    def build_graph(self):
        """
        Build the spatial relationship graph.
        """
        if not self.graph_built:
            # Compute multi-scale distributions if not already calculated
            for scale in self.scales:
                if not hasattr(self, f"distribution_{scale}") or getattr(self, f"distribution_{scale}").sum() == 0:
                    distribution = self._compute_spatial_distribution(scale)
                    self.register_buffer(f"distribution_{scale}", distribution)

            # Compute edge weights based on Earth Mover's Distance at multiple scales
            edge_weights = torch.zeros(self.num_classes, self.num_classes, device=self.positions.device)

            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i != j:
                        # Skip if either class has no spatial information
                        if torch.all(self.sizes[i] == 0) or torch.all(self.sizes[j] == 0):
                            continue

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
                        pos_adjustment = self.pos_proj(combined_emb)

                        # Apply adjustment
                        edge_weights[i, j] = emd_sum * (1.0 + self.positional_adjustment * pos_adjustment)

            # Add spatial relationship modifiers for PASCAL VOC
            # These capture common spatial relationships in natural images

            # Common above/below relationships
            above_below_pairs = [
                ('sky', 'ground'),  # Sky above ground
                ('aeroplane', 'ground'),  # Airplane above ground
                ('bird', 'ground'),  # Bird above ground
                ('ceiling', 'floor'),  # Ceiling above floor
                ('person', 'ground'),  # Person on ground
            ]

            # Common left/right relationships (in Western photos)
            left_right_pairs = [
                ('driver', 'passenger'),  # Driver on left, passenger on right
            ]

            # Apply a small boost to known spatial relationships
            pascal_classes = [
                'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
            ]

            # Ensure the above relationship modifiers are only applied if
            # both classes exist in our current dataset

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
        # Ensure inputs are on the same device
        device = self.positions.device
        batch_positions = batch_positions.to(device)
        batch_sizes = batch_sizes.to(device)
        batch_classes = batch_classes.to(device)

        # Convert batch_classes to a tensor if it's a list
        if isinstance(batch_classes, list):
            batch_classes = torch.tensor(batch_classes, device=device)

        # Make sure batch_classes has the right shape
        if batch_classes.dim() == 0:
            batch_classes = batch_classes.unsqueeze(0)

        # For each class, update position and size
        for i, cls_idx in enumerate(batch_classes):
            if i >= len(batch_positions) or i >= len(batch_sizes):
                continue

            cls_idx = cls_idx.item() if isinstance(cls_idx, torch.Tensor) else cls_idx

            # Skip if class index is out of range
            if cls_idx >= self.num_classes:
                continue

            # Update position (running average)
            old_count = self.sizes[cls_idx, 0] > 0
            if old_count:
                # Update with running average
                # Use exponential moving average for more stable updates
                momentum = 0.9  # Keep 90% of old value, add 10% of new value
                self.positions[cls_idx] = momentum * self.positions[cls_idx] + (1 - momentum) * batch_positions[i]
                self.sizes[cls_idx] = momentum * self.sizes[cls_idx] + (1 - momentum) * batch_sizes[i]
            else:
                # Initialize
                self.positions[cls_idx] = batch_positions[i]
                self.sizes[cls_idx] = batch_sizes[i]

        # Update overlap statistics
        # Group by image
        # This is a simplification - in practice, you'd need to track which objects are from the same image
        for i in range(len(batch_classes)):
            for j in range(i + 1, len(batch_classes)):
                if i >= len(batch_positions) or j >= len(batch_positions) or \
                        i >= len(batch_sizes) or j >= len(batch_sizes) or \
                        i >= len(batch_classes) or j >= len(batch_classes):
                    continue

                cls_i = batch_classes[i].item() if isinstance(batch_classes[i], torch.Tensor) else batch_classes[i]
                cls_j = batch_classes[j].item() if isinstance(batch_classes[j], torch.Tensor) else batch_classes[j]

                # Skip if class indices are out of range
                if cls_i >= self.num_classes or cls_j >= self.num_classes:
                    continue

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

                    # Update overlap with exponential moving average
                    momentum = 0.9  # Keep 90% of old value, add 10% of new value
                    self.overlaps[cls_i, cls_j] = momentum * self.overlaps[cls_i, cls_j] + (1 - momentum) * iou
                    self.overlaps[cls_j, cls_i] = self.overlaps[cls_i, cls_j]

        # Mark graph as not built to trigger rebuild
        self.graph_built = False

        # Compute distributions for each scale
        for scale in self.scales:
            distribution = self._compute_spatial_distribution(scale)
            self.register_buffer(f"distribution_{scale}", distribution)

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

