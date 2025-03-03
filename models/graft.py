"""
GRAFT: Graph-Augmented Framework with Vision Transformers for Multi-Label Classification.

This module implements the complete GRAFT framework, combining the Vision Transformer
backbone with multiple graph components for enhanced multi-label classification.
"""
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.vit import create_vit_model
from models.graph.co_occurrence import create_co_occurrence_graph
from models.graph.spatial import create_spatial_relationship_graph
from models.graph.visual import create_visual_feature_graph
from models.graph.fusion import create_graph_fusion_network


class GRAFT(nn.Module):
    """
    Graph-Augmented Framework with Vision Transformers for Multi-Label Classification.

    This model combines a Vision Transformer backbone with graph components
    to capture complex relationships between labels for improved multi-label classification.
    """

    def __init__(
            self,
            num_classes: int,
            class_names: List[str],
            img_size: int = 224,
            vit_variant: str = "base",
            pretrained: bool = True,
            pretrained_weights: Optional[str] = None,
            feature_dim: int = 768,
            graphs_enabled: bool = True,
            co_occurrence_enabled: bool = True,
            spatial_enabled: bool = True,
            visual_enabled: bool = True,
            context_types: int = 4,
            context_similarity_threshold: float = 0.5,
            scales: List[int] = [5, 15, 25],
            scale_weights: List[float] = [0.2, 0.3, 0.5],
            positional_adjustment: float = 0.2,
            similarity_balance: float = 0.7,
            tier1_threshold: int = 50,
            tier2_threshold: int = 10,
            initial_uncertainties: Optional[List[float]] = None,
            use_checkpointing: bool = False
    ):
        """
        Initialize GRAFT model.

        Args:
            num_classes: Number of classes for multi-label classification.
            class_names: List of class names.
            img_size: Input image size.
            vit_variant: Vision Transformer variant ("base" or "large").
            pretrained: Whether to use pretrained weights for the backbone.
            pretrained_weights: Path to pretrained weights.
            feature_dim: Feature dimension.
            graphs_enabled: Whether to enable graph components.
            co_occurrence_enabled: Whether to enable co-occurrence graph.
            spatial_enabled: Whether to enable spatial relationship graph.
            visual_enabled: Whether to enable visual feature graph.
            context_types: Number of context types for co-occurrence graph.
            context_similarity_threshold: Threshold for context similarity.
            scales: List of grid scales for spatial relationship graph.
            scale_weights: Weights for each scale.
            positional_adjustment: Factor for adjusting positional relationships.
            similarity_balance: Balance factor between visual and contextual similarity.
            tier1_threshold: Threshold for Tier 1 relationships.
            tier2_threshold: Threshold for Tier 2 relationships.
            initial_uncertainties: Initial uncertainty estimates for each graph.
            use_checkpointing: Whether to use gradient checkpointing to save memory.
        """
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names
        self.img_size = img_size
        self.feature_dim = feature_dim
        self.graphs_enabled = graphs_enabled
        self.use_checkpointing = use_checkpointing

        # Create Vision Transformer backbone
        self.backbone = create_vit_model(
            variant=vit_variant,
            img_size=img_size,
            num_classes=num_classes,
            pretrained=pretrained,
            pretrained_weights=pretrained_weights,
            use_checkpointing=use_checkpointing
        )

        # Extract features from backbone (remove classification head)
        self.backbone_feature_dim = feature_dim  # Usually 768 for ViT-Base

        # Node feature initialization - lightweight projection
        self.node_init = nn.Linear(self.backbone_feature_dim, feature_dim)

        # Graph components
        if graphs_enabled:
            # Initialize graph components
            self.graph_components = nn.ModuleDict()
            self.enabled_graphs = []

            if co_occurrence_enabled:
                self.graph_components["co_occurrence"] = create_co_occurrence_graph(
                    num_classes=num_classes,
                    context_types=context_types,
                    context_similarity_threshold=context_similarity_threshold
                )
                self.enabled_graphs.append("co_occurrence")

            if spatial_enabled:
                self.graph_components["spatial"] = create_spatial_relationship_graph(
                    num_classes=num_classes,
                    scales=scales,
                    scale_weights=scale_weights,
                    positional_adjustment=positional_adjustment
                )
                self.enabled_graphs.append("spatial")

            if visual_enabled:
                self.graph_components["visual"] = create_visual_feature_graph(
                    num_classes=num_classes,
                    feature_dim=feature_dim,
                    similarity_balance=similarity_balance,
                    tier1_threshold=tier1_threshold,
                    tier2_threshold=tier2_threshold
                )
                self.enabled_graphs.append("visual")

            # Prepare default uncertainties if needed
            if initial_uncertainties is None:
                initial_uncertainties = [1.0] * len(self.enabled_graphs)

            # Ensure we have the right number of uncertainty values
            if len(initial_uncertainties) != len(self.enabled_graphs):
                initial_uncertainties = [1.0] * len(self.enabled_graphs)

            # Graph fusion network
            if len(self.enabled_graphs) > 0:
                self.fusion_network = create_graph_fusion_network(
                    num_classes=num_classes,
                    feature_dim=feature_dim,
                    num_graphs=len(self.enabled_graphs),
                    hidden_dim=feature_dim // 2,
                    dropout=0.1,
                    initial_uncertainties=initial_uncertainties
                )

        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input images of shape [batch_size, 3, img_size, img_size].

        Returns:
            Dictionary containing:
                - logits: Classification logits [batch_size, num_classes].
                - features: Visual features [batch_size, feature_dim].
                - node_features: Node features after graph processing [batch_size, num_classes, feature_dim].
        """
        batch_size = x.shape[0]

        # Extract visual features from backbone
        backbone_features = self.backbone.forward_features(x)  # [batch_size, feature_dim]

        # Initial node features (replicated for each class)
        node_features = self.node_init(backbone_features)  # [batch_size, feature_dim]
        node_features = node_features.unsqueeze(1).expand(-1, self.num_classes,
                                                          -1)  # [batch_size, num_classes, feature_dim]

        # Apply graph components
        if self.graphs_enabled and len(self.enabled_graphs) > 0:
            # Get adjacency matrices for all enabled graphs
            adj_matrices = []

            for graph_name in self.enabled_graphs:
                graph = self.graph_components[graph_name]
                # Use custom get_adjacency_matrix method if available, otherwise use the module's built-in one
                adj_matrices.append(graph.get_adjacency_matrix())

            # Apply graph fusion
            node_features = self.fusion_network(node_features, adj_matrices)

        # Apply classifier to get logits
        logits = self.classifier(node_features.mean(dim=1))  # [batch_size, num_classes]

        return {
            "logits": logits,
            "features": backbone_features,
            "node_features": node_features
        }

    def update_graph_statistics(self, batch_data: Dict[str, Any]):
        """
        Update statistics for all graph components.

        Args:
            batch_data: Dictionary containing batch data, including:
                - labels: Multi-hot encoded labels [batch_size, num_classes].
                - features: Visual features [batch_size, feature_dim].
                - boxes: Bounding boxes [batch_size, num_boxes, 4].
                - box_labels: Class indices for boxes [batch_size, num_boxes].
        """
        if not self.graphs_enabled:
            return

        # Update co-occurrence graph
        if "co_occurrence" in self.graph_components and "labels" in batch_data:
            labels = batch_data["labels"]
            if labels is not None:
                self.graph_components["co_occurrence"].update_statistics(labels)

        # Update spatial relationship graph
        if "spatial" in self.graph_components and "boxes" in batch_data and "box_labels" in batch_data:
            # Extract box positions and sizes
            boxes = batch_data["boxes"]
            box_labels = batch_data["box_labels"]

            if boxes is not None and box_labels is not None:
                # Convert boxes to positions and sizes for spatial graph
                positions = []
                sizes = []

                for box in boxes:
                    x_min, y_min, x_max, y_max = box
                    positions.append([(x_min + x_max) / 2, (y_min + y_max) / 2])
                    sizes.append([x_max - x_min, y_max - y_min])

                # Convert to tensors if they're not already
                if not isinstance(positions, torch.Tensor):
                    positions = torch.tensor(positions,
                                             device=self.spatial.device if hasattr(self, 'spatial') else 'cpu')
                if not isinstance(sizes, torch.Tensor):
                    sizes = torch.tensor(sizes, device=self.spatial.device if hasattr(self, 'spatial') else 'cpu')

                self.graph_components["spatial"].update_statistics(positions, sizes, box_labels)

        # Update visual feature graph
        if "visual" in self.graph_components and "features" in batch_data and "labels" in batch_data:
            features = batch_data["features"]
            labels = batch_data["labels"]

            if features is not None and labels is not None:
                self.graph_components["visual"].update_features(features, labels)


def create_graft_model(
        num_classes: int,
        class_names: List[str],
        config: Dict[str, Any]
) -> GRAFT:
    """
    Create a GRAFT model with specified configuration.

    Args:
        num_classes: Number of classes for multi-label classification.
        class_names: List of class names.
        config: Model configuration dictionary.

    Returns:
        GRAFT model.
    """
    # Extract configuration parameters
    img_size = config.get("img_size", 224)
    vit_variant = config.get("vit_variant", "base")
    pretrained = config.get("backbone", {}).get("pretrained", True)
    pretrained_weights = config.get("backbone", {}).get("pretrained_weights", None)
    feature_dim = config.get("feature_dim", 768)
    use_checkpointing = config.get("use_checkpointing", False)

    # Graph configuration
    graphs_config = config.get("graphs", {})
    graphs_enabled = graphs_config.get("enabled", True)
    co_occurrence_enabled = graphs_config.get("co_occurrence", {}).get("enabled", True)
    spatial_enabled = graphs_config.get("spatial", {}).get("enabled", True)
    visual_enabled = graphs_config.get("visual", {}).get("enabled", True)

    # Co-occurrence graph parameters
    context_types = graphs_config.get("co_occurrence", {}).get("context_types", 4)
    context_similarity_threshold = graphs_config.get("co_occurrence", {}).get("context_similarity_threshold", 0.5)

    # Spatial graph parameters
    scales = graphs_config.get("spatial", {}).get("scales", [5, 15, 25])
    scale_weights = graphs_config.get("spatial", {}).get("scale_weights", [0.2, 0.3, 0.5])
    positional_adjustment = graphs_config.get("spatial", {}).get("positional_adjustment", 0.2)

    # Visual graph parameters
    similarity_balance = graphs_config.get("visual", {}).get("similarity_balance", 0.7)
    tier1_threshold = graphs_config.get("visual", {}).get("tier1_threshold", 50)
    tier2_threshold = graphs_config.get("visual", {}).get("tier2_threshold", 10)

    # Fusion parameters
    initial_uncertainties = graphs_config.get("fusion", {}).get("initial_uncertainties", None)

    # Create GRAFT model
    model = GRAFT(
        num_classes=num_classes,
        class_names=class_names,
        img_size=img_size,
        vit_variant=vit_variant,
        pretrained=pretrained,
        pretrained_weights=pretrained_weights,
        feature_dim=feature_dim,
        graphs_enabled=graphs_enabled,
        co_occurrence_enabled=co_occurrence_enabled,
        spatial_enabled=spatial_enabled,
        visual_enabled=visual_enabled,
        context_types=context_types,
        context_similarity_threshold=context_similarity_threshold,
        scales=scales,
        scale_weights=scale_weights,
        positional_adjustment=positional_adjustment,
        similarity_balance=similarity_balance,
        tier1_threshold=tier1_threshold,
        tier2_threshold=tier2_threshold,
        initial_uncertainties=initial_uncertainties,
        use_checkpointing=use_checkpointing
    )

    return model

