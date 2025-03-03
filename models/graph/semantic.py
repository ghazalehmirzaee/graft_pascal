"""
Semantic relationship graph module for GRAFT framework.

This module implements the semantic relationship graph component
that captures label relationships based on their taxonomic,
functional, and scene-type similarities.
"""
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SemanticRelationshipGraph(nn.Module):
    """
    Semantic Relationship Graph module.

    This graph captures the semantic relationships between labels
    based on their taxonomic hierarchy, functional similarity,
    and scene-type co-occurrence patterns.
    """

    def __init__(
            self,
            num_classes: int,
            class_names: List[str],
            dimension_weights: List[float] = [0.3, 0.4, 0.3],
            adaptation_factor: float = 0.7
    ):
        """
        Initialize Semantic Relationship Graph.

        Args:
            num_classes: Number of classes/labels.
            class_names: List of class names.
            dimension_weights: Weights for taxonomic, functional, and scene dimensions.
            adaptation_factor: Factor for adapting semantic relationships with observed data.
        """
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names
        self.dimension_weights = dimension_weights
        self.adaptation_factor = adaptation_factor

        assert len(dimension_weights) == 3, "Must provide weights for all 3 semantic dimensions"
        assert sum(dimension_weights) == 1.0, "Dimension weights must sum to 1.0"

        # Initialize semantic relationship matrices
        self.register_buffer("taxonomic_sim", self._compute_taxonomic_similarity())
        self.register_buffer("functional_sim", self._compute_functional_similarity())
        self.register_buffer("scene_sim", self._compute_scene_similarity())

        # Initialize edge weights
        self.register_buffer("edge_weights", torch.zeros(num_classes, num_classes))

        # Initialize observed co-occurrence (will be updated during training)
        self.register_buffer("observed_co_occurrence", torch.zeros(num_classes, num_classes))

        # Flag to track whether the graph has been built
        self.graph_built = False

        # Build initial graph
        self.build_graph()

    def _compute_taxonomic_similarity(self) -> torch.Tensor:
        """
        Compute taxonomic similarity between classes based on WordNet hierarchy.

        For PASCAL VOC, we use a predefined taxonomy:
        - vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
        - animal: bird, cat, cow, dog, horse, sheep
        - person: person
        - indoor: bottle, chair, diningtable, pottedplant, sofa, tvmonitor

        Returns:
            Taxonomic similarity matrix.
        """
        taxonomy = {
            'vehicle': ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train'],
            'animal': ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep'],
            'person': ['person'],
            'indoor': ['bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
        }

        # Initialize similarity matrix
        tax_sim = torch.zeros(self.num_classes, self.num_classes)

        # For each pair of classes, compute similarity
        for i, name_i in enumerate(self.class_names):
            for j, name_j in enumerate(self.class_names):
                if i == j:
                    tax_sim[i, j] = 1.0
                else:
                    # Check if classes belong to the same taxonomic category
                    same_category = False
                    for category, members in taxonomy.items():
                        if name_i in members and name_j in members:
                            same_category = True
                            break

                    tax_sim[i, j] = 0.7 if same_category else 0.1

        return tax_sim

    def _compute_functional_similarity(self) -> torch.Tensor:
        """
        Compute functional similarity between classes based on affordances.

        Classes with similar functional roles (e.g., "sittable", "container")
        have higher similarity.

        Returns:
            Functional similarity matrix.
        """
        # Define functional attributes
        functional_attributes = {
            'sittable': ['chair', 'sofa', 'diningtable'],
            'vehicle': ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train'],
            'container': ['bottle'],
            'animal': ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep'],
            'display': ['tvmonitor'],
            'plant': ['pottedplant']
        }

        # Create attribute-to-class mapping
        class_to_attributes = {class_name: [] for class_name in self.class_names}
        for attr, classes in functional_attributes.items():
            for class_name in classes:
                if class_name in class_to_attributes:
                    class_to_attributes[class_name].append(attr)

        # Initialize similarity matrix
        func_sim = torch.zeros(self.num_classes, self.num_classes)

        # For each pair of classes, compute similarity
        for i, name_i in enumerate(self.class_names):
            for j, name_j in enumerate(self.class_names):
                if i == j:
                    func_sim[i, j] = 1.0
                else:
                    # Get attributes for each class
                    attrs_i = set(class_to_attributes[name_i])
                    attrs_j = set(class_to_attributes[name_j])

                    # Compute Jaccard similarity
                    if attrs_i and attrs_j:
                        intersection = len(attrs_i.intersection(attrs_j))
                        union = len(attrs_i.union(attrs_j))
                        func_sim[i, j] = intersection / union if union > 0 else 0.0
                    else:
                        func_sim[i, j] = 0.0

        return func_sim

    def _compute_scene_similarity(self) -> torch.Tensor:
        """
        Compute scene-type similarity between classes.

        Classes that commonly appear in similar scene types (indoor, outdoor, urban, natural)
        have higher similarity.

        Returns:
            Scene-type similarity matrix.
        """
        # Define scene types
        scene_types = {
            'indoor_home': ['bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'],
            'indoor_office': ['chair', 'pottedplant', 'tvmonitor'],
            'outdoor_urban': ['bicycle', 'bus', 'car', 'motorbike', 'person', 'train'],
            'outdoor_natural': ['aeroplane', 'bird', 'boat', 'cow', 'dog', 'cat', 'horse', 'sheep']
        }

        # Create class-to-scene mapping
        class_to_scenes = {class_name: [] for class_name in self.class_names}
        for scene, classes in scene_types.items():
            for class_name in classes:
                if class_name in class_to_scenes:
                    class_to_scenes[class_name].append(scene)

        # Initialize similarity matrix
        scene_sim = torch.zeros(self.num_classes, self.num_classes)

        # For each pair of classes, compute similarity
        for i, name_i in enumerate(self.class_names):
            for j, name_j in enumerate(self.class_names):
                if i == j:
                    scene_sim[i, j] = 1.0
                else:
                    # Get scene types for each class
                    scenes_i = set(class_to_scenes[name_i])
                    scenes_j = set(class_to_scenes[name_j])

                    # Compute Jaccard similarity
                    if scenes_i and scenes_j:
                        intersection = len(scenes_i.intersection(scenes_j))
                        union = len(scenes_i.union(scenes_j))
                        scene_sim[i, j] = intersection / union if union > 0 else 0.0
                    else:
                        scene_sim[i, j] = 0.0

        return scene_sim

    def update_co_occurrence(self, labels: torch.Tensor):
        """
        Update observed co-occurrence matrix from batch of labels.

        Args:
            labels: Multi-hot encoded labels of shape [batch_size, num_classes].
        """
        batch_size = labels.shape[0]

        # Update co-occurrence matrix
        for i in range(batch_size):
            # Get positive classes
            pos_indices = torch.nonzero(labels[i]).squeeze(-1)

            # Update co-occurrence for each pair of labels
            for idx1 in pos_indices:
                for idx2 in pos_indices:
                    if idx1 != idx2:
                        self.observed_co_occurrence[idx1, idx2] += 1

        # Mark graph as not built to trigger rebuild
        self.graph_built = False

    def build_graph(self):
        """
        Build the semantic relationship graph.
        """
        # Compute weighted combination of semantic similarities
        semantic_sim = (
                self.dimension_weights[0] * self.taxonomic_sim +
                self.dimension_weights[1] * self.functional_sim +
                self.dimension_weights[2] * self.scene_sim
        )

        # Normalize observed co-occurrence
        observed_sim = torch.zeros_like(self.observed_co_occurrence)

        if torch.sum(self.observed_co_occurrence) > 0:
            # Compute row-wise sum
            row_sum = torch.sum(self.observed_co_occurrence, dim=1, keepdim=True)

            # Normalize by row sum to get conditional probabilities
            mask = row_sum > 0
            observed_sim[mask.repeat(1, self.num_classes)] = (
                    self.observed_co_occurrence[mask.repeat(1, self.num_classes)] /
                    row_sum.repeat(1, self.num_classes)[mask.repeat(1, self.num_classes)]
            )

            # Symmetrize
            observed_sim = 0.5 * (observed_sim + observed_sim.t())

        # Combine semantic and observed similarities
        if torch.sum(self.observed_co_occurrence) > 0:
            # Adapt semantic similarity with observed co-occurrence
            edge_weights = (
                    (1 - self.adaptation_factor) * semantic_sim +
                    self.adaptation_factor * observed_sim
            )
        else:
            # Use only semantic similarity
            edge_weights = semantic_sim

        # Normalize
        max_weight = torch.max(edge_weights)
        if max_weight > 0:
            edge_weights = edge_weights / max_weight

        self.edge_weights = edge_weights
        self.graph_built = True

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


def create_semantic_relationship_graph(
        num_classes: int,
        class_names: List[str],
        dimension_weights: List[float] = [0.3, 0.4, 0.3],
        adaptation_factor: float = 0.7
) -> SemanticRelationshipGraph:
    """
    Create a Semantic Relationship Graph module.

    Args:
        num_classes: Number of classes/labels.
        class_names: List of class names.
        dimension_weights: Weights for taxonomic, functional, and scene dimensions.
        adaptation_factor: Factor for adapting semantic relationships with observed data.

    Returns:
        SemanticRelationshipGraph module.
    """
    return SemanticRelationshipGraph(
        num_classes=num_classes,
        class_names=class_names,
        dimension_weights=dimension_weights,
        adaptation_factor=adaptation_factor
    )

