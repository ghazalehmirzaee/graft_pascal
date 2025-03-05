"""
Visual feature relationship graph module for GRAFT framework.

This module implements the visual feature relationship graph
component that captures label relationships based on their visual
similarities in the feature space.
"""
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VisualFeatureGraph(nn.Module):
    """
    Visual Feature Relationship Graph module.

    This graph captures the relationships between labels based on
    their visual feature similarities extracted from the vision
    transformer backbone.
    """

    def __init__(
            self,
            num_classes: int,
            feature_dim: int,
            similarity_balance: float = 0.7,
            tier1_threshold: int = 50,
            tier2_threshold: int = 10
    ):
        """
        Initialize Visual Feature Graph.

        Args:
            num_classes: Number of classes/labels.
            feature_dim: Dimension of visual features.
            similarity_balance: Balance factor between visual and contextual similarity.
            tier1_threshold: Threshold for Tier 1 relationships (frequent co-occurrences).
            tier2_threshold: Threshold for Tier 2 relationships (moderate co-occurrences).
        """
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.similarity_balance = similarity_balance
        self.tier1_threshold = tier1_threshold
        self.tier2_threshold = tier2_threshold

        # Initialize class-specific feature embeddings
        self.class_features = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.class_features)

        # Initialize context embeddings
        self.context_features = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.context_features)

        # Initialize co-occurrence counter
        self.register_buffer("co_occurrence", torch.zeros(num_classes, num_classes))

        # Initialize edge weights
        self.register_buffer("edge_weights", torch.zeros(num_classes, num_classes))

        # Instance count buffer
        self.register_buffer("instance_counts", torch.zeros(num_classes))

        # Learnable feature similarity projection
        self.sim_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )

        # Whether the graph has been built
        self.graph_built = False

    def update_features(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Update class-specific visual features from batch.

        Args:
            features: Visual features of shape [batch_size, feature_dim].
            labels: Multi-hot encoded labels of shape [batch_size, num_classes].
        """
        batch_size = features.shape[0]

        # For each sample, update features for positive classes
        for i in range(batch_size):
            # Get positive classes
            pos_indices = torch.nonzero(labels[i]).squeeze(-1)

            # Update co-occurrence for each pair of labels
            for idx1 in pos_indices:
                # Increment instance count
                self.instance_counts[idx1] += 1

                # Update class feature with moving average
                if self.instance_counts[idx1] > 1:
                    alpha = 1.0 / self.instance_counts[idx1]
                    self.class_features.data[idx1] = (1 - alpha) * self.class_features.data[idx1] + alpha * features[i]
                else:
                    self.class_features.data[idx1] = features[i]

                # Update co-occurrence
                for idx2 in pos_indices:
                    if idx1 != idx2:
                        self.co_occurrence[idx1, idx2] += 1

        # Mark graph as not built to trigger rebuild
        self.graph_built = False

    def build_graph(self):
        """
        Build the visual feature relationship graph using a tiered approach.
        """
        if not self.graph_built:
            edge_weights = torch.zeros(self.num_classes, self.num_classes, device=self.class_features.device)

            # Compute visual feature similarity
            # Normalize feature embeddings
            norm_features = F.normalize(self.class_features, p=2, dim=1)

            # Compute cosine similarity
            visual_sim = torch.mm(norm_features, norm_features.t())

            # Compute context similarity
            # Normalize context embeddings
            norm_context = F.normalize(self.context_features, p=2, dim=1)

            # Compute cosine similarity
            context_sim = torch.mm(norm_context, norm_context.t())

            # For each pair of classes, determine relationship tier and compute edge weight
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i != j:
                        # Determine tier based on co-occurrence frequency
                        co_occur_count = self.co_occurrence[i, j]

                        if co_occur_count > self.tier1_threshold:
                            # Tier 1: Frequent co-occurrences
                            # Directly use visual similarity with area weighting

                            # Compute feature difference (complementary features)
                            feat_diff = self.class_features[i] - self.class_features[j]

                            # Project difference to similarity score
                            sim_weight = self.sim_proj(feat_diff)

                            # Combine visual and context similarity
                            combined_sim = self.similarity_balance * visual_sim[i, j] + \
                                           (1 - self.similarity_balance) * context_sim[i, j]

                            # Weight by instance ratio for class balance
                            instance_ratio = min(self.instance_counts[i], self.instance_counts[j]) / \
                                             max(self.instance_counts[i], self.instance_counts[j])

                            edge_weights[i, j] = combined_sim * sim_weight * instance_ratio

                        elif co_occur_count > self.tier2_threshold:
                            # Tier 2: Moderate co-occurrences
                            # Interpolate between visual similarity and average relationships

                            # Compute direct visual similarity
                            direct_sim = self.similarity_balance * visual_sim[i, j] + \
                                         (1 - self.similarity_balance) * context_sim[i, j]

                            # Compute average relationship with shared neighbors
                            shared_neighbors = []
                            for k in range(self.num_classes):
                                if k != i and k != j and \
                                        self.co_occurrence[i, k] > self.tier1_threshold and \
                                        self.co_occurrence[j, k] > self.tier1_threshold:
                                    shared_neighbors.append(k)

                            # If there are shared neighbors, use their average relationship
                            avg_sim = 0.0
                            if shared_neighbors:
                                # Compute average visual similarity through shared neighbors
                                for k in shared_neighbors:
                                    avg_sim += (visual_sim[i, k] * visual_sim[j, k])
                                avg_sim /= len(shared_neighbors)

                                # Confidence factor based on co-occurrence
                                confidence = co_occur_count / self.tier1_threshold

                                # Interpolate between direct and average similarity
                                edge_weights[i, j] = confidence * direct_sim + (1 - confidence) * avg_sim
                            else:
                                # No shared neighbors, use direct similarity with lower confidence
                                confidence = co_occur_count / self.tier1_threshold
                                edge_weights[i, j] = confidence * direct_sim

                        else:
                            # Tier 3: Rare or no co-occurrences
                            # Infer relationships from network structure

                            # Find neighbors of i and j with strong connections
                            i_neighbors = []
                            j_neighbors = []

                            for k in range(self.num_classes):
                                if k != i and k != j:
                                    if self.co_occurrence[i, k] > self.tier2_threshold:
                                        i_neighbors.append((k, self.co_occurrence[i, k]))
                                    if self.co_occurrence[j, k] > self.tier2_threshold:
                                        j_neighbors.append((k, self.co_occurrence[j, k]))

                            # Sort neighbors by co-occurrence strength
                            i_neighbors.sort(key=lambda x: x[1], reverse=True)
                            j_neighbors.sort(key=lambda x: x[1], reverse=True)

                            # Take top neighbors (limit to 5 for efficiency)
                            i_top = i_neighbors[:5]
                            j_top = j_neighbors[:5]

                            # Compute transitive similarity
                            trans_sim = 0.0
                            total_weight = 0.0

                            for i_neigh, i_co in i_top:
                                for j_neigh, j_co in j_top:
                                    # Check for connections between neighbors
                                    if self.co_occurrence[i_neigh, j_neigh] > 0:
                                        # Transitive connection exists
                                        # Weight by connection strengths
                                        weight = (i_co * j_co) / self.tier1_threshold
                                        sim = (visual_sim[i, i_neigh] * visual_sim[j, j_neigh] *
                                               visual_sim[i_neigh, j_neigh])

                                        trans_sim += weight * sim
                                        total_weight += weight

                            if total_weight > 0:
                                edge_weights[i, j] = trans_sim / total_weight
                            else:
                                # Fallback to context similarity with low confidence
                                edge_weights[i, j] = 0.1 * context_sim[i, j]

            # Normalize edge weights
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


def create_visual_feature_graph(
        num_classes: int,
        feature_dim: int,
        similarity_balance: float = 0.7,
        tier1_threshold: int = 50,
        tier2_threshold: int = 10
) -> VisualFeatureGraph:
    """
    Create a Visual Feature Graph module.

    Args:
        num_classes: Number of classes/labels.
        feature_dim: Dimension of visual features.
        similarity_balance: Balance factor between visual and contextual similarity.
        tier1_threshold: Threshold for Tier 1 relationships (frequent co-occurrences).
        tier2_threshold: Threshold for Tier 2 relationships (moderate co-occurrences).

    Returns:
        VisualFeatureGraph module.
    """
    return VisualFeatureGraph(
        num_classes=num_classes,
        feature_dim=feature_dim,
        similarity_balance=similarity_balance,
        tier1_threshold=tier1_threshold,
        tier2_threshold=tier2_threshold
    )

