"""
Graph fusion network module for GRAFT framework.

This module implements the contextual graph fusion network
that integrates multiple graph components with adaptive,
context-aware fusion.
"""
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer based on the GAT architecture.

    This layer applies attention mechanisms to learn node representations
    by aggregating features from its neighbors with attention weights.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            dropout: float = 0.2,
            alpha: float = 0.2,
            concat: bool = True
    ):
        """
        Initialize Graph Attention Layer.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            dropout: Dropout probability.
            alpha: LeakyReLU negative slope.
            concat: Whether to concatenate or average attention heads.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # Linear transformation
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention weights
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # Leaky ReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(
            self,
            x: torch.Tensor,
            adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input node features of shape [batch_size, num_nodes, in_features].
            adj: Adjacency matrix of shape [batch_size, num_nodes, num_nodes].

        Returns:
            Updated node features of shape [batch_size, num_nodes, out_features].
        """
        batch_size, num_nodes, _ = x.size()

        # Linear transformation
        h = torch.bmm(x, self.W.unsqueeze(0).expand(batch_size, -1, -1))  # [batch_size, num_nodes, out_features]

        # Prepare inputs for attention
        # Create all possible node pairs
        a_input = torch.cat([
            h.repeat(1, 1, num_nodes).view(batch_size, num_nodes * num_nodes, self.out_features),
            h.repeat(1, num_nodes, 1)
        ], dim=2).view(batch_size, num_nodes, num_nodes, 2 * self.out_features)

        # Calculate attention coefficients
        e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(-1)  # [batch_size, num_nodes, num_nodes]

        # Mask attention coefficients using adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # [batch_size, num_nodes, num_nodes]

        # Apply softmax to normalize attention weights
        attention = F.softmax(attention, dim=2)  # [batch_size, num_nodes, num_nodes]
        attention = self.dropout_layer(attention)  # [batch_size, num_nodes, num_nodes]

        # Apply attention to node features
        h_prime = torch.bmm(attention, h)  # [batch_size, num_nodes, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class ContextualGraphFusion(nn.Module):
    """
    Contextual Graph Fusion Network.

    This module fuses multiple graph components with adaptive,
    context-aware fusion based on graph attention networks.
    """

    def __init__(
            self,
            num_classes: int,
            feature_dim: int,
            num_graphs: int = 3,  # Changed from 4 to 3 since semantic graph was removed
            hidden_dim: int = 128,
            dropout: float = 0.2,
            initial_uncertainties: Optional[List[float]] = None,
            multi_head: bool = True
    ):
        """
        Initialize Graph Fusion Network.

        Args:
            num_classes: Number of classes/labels.
            feature_dim: Feature dimension.
            num_graphs: Number of graph components to fuse.
            hidden_dim: Hidden layer dimension.
            dropout: Dropout probability.
            initial_uncertainties: Initial uncertainty estimates for each graph.
            multi_head: Whether to use multiple attention heads.
        """
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_graphs = num_graphs
        self.hidden_dim = hidden_dim
        self.multi_head = multi_head

        # Initialize graph-specific attention layers
        if multi_head:
            # Multi-head attention with 2 heads
            self.attention_heads = nn.ModuleList([
                nn.ModuleList([
                    GraphAttentionLayer(
                        in_features=feature_dim,
                        out_features=hidden_dim // 2,  # Split dimension between heads
                        dropout=dropout,
                        concat=True
                    ) for _ in range(2)  # 2 attention heads
                ]) for _ in range(num_graphs)
            ])

            # Output projection after concatenating heads
            self.head_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ) for _ in range(num_graphs)
            ])
        else:
            # Single-head attention
            self.attention_layers = nn.ModuleList([
                GraphAttentionLayer(
                    in_features=feature_dim,
                    out_features=hidden_dim,
                    dropout=dropout,
                    concat=True
                ) for _ in range(num_graphs)
            ])

        # Initialize graph uncertainty estimates
        if initial_uncertainties is not None:
            assert len(initial_uncertainties) == num_graphs, "Must provide uncertainty for each graph"
            self.register_buffer("graph_uncertainties", torch.tensor(initial_uncertainties))
        else:
            self.register_buffer("graph_uncertainties", torch.ones(num_graphs))

        # Make uncertainties learnable with constrained optimization
        self.log_uncertainties = nn.Parameter(torch.log(self.graph_uncertainties))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim)
        )

        # Layer normalization for stabilizing training
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(
            self,
            x: torch.Tensor,
            adj_matrices: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input node features of shape [batch_size, num_classes, feature_dim].
            adj_matrices: List of adjacency matrices, each of shape [num_classes, num_classes].

        Returns:
            Fused node features of shape [batch_size, num_classes, feature_dim].
        """
        assert len(adj_matrices) == self.num_graphs, "Number of adjacency matrices must match num_graphs"

        batch_size = x.size(0)

        # Compute uncertainty weights
        uncertainties = torch.exp(self.log_uncertainties)
        attention_weights = 1.0 / (uncertainties + 1e-10)  # Add epsilon to prevent division by zero
        attention_weights = F.softmax(attention_weights, dim=0)

        # Apply graph-specific attention layers
        graph_outputs = []

        for i in range(self.num_graphs):
            # Expand adjacency matrix to batch size
            batch_adj = adj_matrices[i].unsqueeze(0).expand(batch_size, -1, -1)

            if self.multi_head:
                # Apply multi-head attention
                head_outputs = []
                for head in self.attention_heads[i]:
                    head_output = head(x, batch_adj)
                    head_outputs.append(head_output)

                # Concatenate heads
                multi_head_output = torch.cat(head_outputs, dim=2)

                # Apply projection
                graph_output = self.head_projections[i](multi_head_output)
            else:
                # Apply single-head attention
                graph_output = self.attention_layers[i](x, batch_adj)

            graph_outputs.append(graph_output)

        # Weight and combine graph outputs
        fused_output = torch.zeros_like(graph_outputs[0])
        for i, graph_output in enumerate(graph_outputs):
            fused_output += attention_weights[i] * graph_output

        # Apply output projection
        fused_output = self.output_proj(fused_output)

        # Layer normalization
        fused_output = self.layer_norm(fused_output)

        # Skip connection
        x_out = x + fused_output

        return x_out


def create_graph_fusion_network(
        num_classes: int,
        feature_dim: int,
        num_graphs: int = 3,  # Changed from 4 to 3 since semantic graph was removed
        hidden_dim: int = 128,
        dropout: float = 0.2,
        initial_uncertainties: Optional[List[float]] = None,
        multi_head: bool = True
) -> ContextualGraphFusion:
    """
    Create a Graph Fusion Network.

    Args:
        num_classes: Number of classes/labels.
        feature_dim: Feature dimension.
        num_graphs: Number of graph components to fuse.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout probability.
        initial_uncertainties: Initial uncertainty estimates for each graph.
        multi_head: Whether to use multiple attention heads.

    Returns:
        ContextualGraphFusion module.
    """
    return ContextualGraphFusion(
        num_classes=num_classes,
        feature_dim=feature_dim,
        num_graphs=num_graphs,
        hidden_dim=hidden_dim,
        dropout=dropout,
        initial_uncertainties=initial_uncertainties,
        multi_head=multi_head
    )

