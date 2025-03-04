"""
Visualization utilities for GRAFT framework.

This module provides functions for visualizing model predictions,
graph structures, and attention weights.
"""
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score


def plot_adjacency_matrix(
        adjacency_matrix: Union[np.ndarray, torch.Tensor],
        class_names: List[str],
        title: str = "Adjacency Matrix",
        cmap: str = "Blues",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot adjacency matrix as a heatmap.

    Args:
        adjacency_matrix: Adjacency matrix of shape [num_classes, num_classes].
        class_names: List of class names.
        title: Plot title.
        cmap: Colormap for heatmap.
        figsize: Figure size.
        save_path: Path to save the figure.

    Returns:
        Matplotlib figure.
    """
    # Convert to numpy if tensor
    if isinstance(adjacency_matrix, torch.Tensor):
        adjacency_matrix = adjacency_matrix.detach().cpu().numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        adjacency_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Class")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Tight layout
    plt.tight_layout()

    # Save figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_graph_network(
        adjacency_matrix: Union[np.ndarray, torch.Tensor],
        class_names: List[str],
        threshold: float = 0.2,
        title: str = "Graph Network",
        figsize: Tuple[int, int] = (12, 10),
        node_size: int = 1000,
        save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot graph network using NetworkX.

    Args:
        adjacency_matrix: Adjacency matrix of shape [num_classes, num_classes].
        class_names: List of class names.
        threshold: Threshold for edge visibility.
        title: Plot title.
        figsize: Figure size.
        node_size: Size of nodes in the plot.
        save_path: Path to save the figure.

    Returns:
        Matplotlib figure.
    """
    # Convert to numpy if tensor
    if isinstance(adjacency_matrix, torch.Tensor):
        adjacency_matrix = adjacency_matrix.detach().cpu().numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create graph
    G = nx.Graph()

    # Add nodes
    for i, name in enumerate(class_names):
        G.add_node(i, label=name)

    # Add edges with weights above threshold
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            weight = adjacency_matrix[i, j]
            if weight > threshold:
                G.add_edge(i, j, weight=weight)

    # Get node positions using spring layout
    pos = nx.spring_layout(G, seed=42)

    # Get edge weights for width
    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]

    # Draw graph
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color="skyblue", alpha=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_weights, edge_color="gray", alpha=0.6)
    nx.draw_networkx_labels(G, pos, ax=ax, labels={i: data['label'] for i, data in G.nodes(data=True)})

    # Set title
    ax.set_title(title)

    # Remove axis
    ax.axis("off")

    # Tight layout
    plt.tight_layout()

    # Save figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        class_names: List[str],
        title: str = "Confusion Matrix",
        cmap: str = "Blues",
        figsize: Tuple[int, int] = (12, 10),
        normalize: bool = True,
        save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix for each class.

    Args:
        y_true: True binary labels of shape [n_samples, n_classes].
        y_pred: Predicted binary labels of shape [n_samples, n_classes].
        class_names: List of class names.
        title: Plot title.
        cmap: Colormap for heatmap.
        figsize: Figure size.
        normalize: Whether to normalize confusion matrix.
        save_path: Path to save the figure.

    Returns:
        Matplotlib figure.
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Number of classes
    n_classes = len(class_names)

    # Create subplots grid
    n_cols = 4
    n_rows = (n_classes + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    # Plot confusion matrix for each class
    for i, class_name in enumerate(class_names):
        if i < len(axes):
            ax = axes[i]

            # Compute confusion matrix
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])

            # Normalize if requested
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm = np.nan_to_num(cm)

            # Plot heatmap
            sns.heatmap(
                cm,
                annot=True,
                fmt=".2f" if normalize else "d",
                cmap=cmap,
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
                ax=ax
            )

            # Set title and labels
            ax.set_title(class_name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

    # Hide empty subplots
    for i in range(n_classes, len(axes)):
        axes[i].axis("off")

    # Set overall title
    fig.suptitle(title, fontsize=16)

    # Tight layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_precision_recall_curves(
        y_true: Union[np.ndarray, torch.Tensor],
        y_score: Union[np.ndarray, torch.Tensor],
        class_names: List[str],
        title: str = "Precision-Recall Curves",
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot precision-recall curves for each class.

    Args:
        y_true: True binary labels of shape [n_samples, n_classes].
        y_score: Target scores of shape [n_samples, n_classes].
        class_names: List of class names.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the figure.

    Returns:
        Matplotlib figure.
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_score, torch.Tensor):
        y_score = y_score.detach().cpu().numpy()

    # Number of classes
    n_classes = len(class_names)

    # Create subplots grid
    n_cols = 4
    n_rows = (n_classes + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    # Plot precision-recall curve for each class
    for i, class_name in enumerate(class_names):
        if i < len(axes):
            ax = axes[i]

            # Compute precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_score[:, i])

            # Compute average precision
            ap = average_precision_score(y_true[:, i], y_score[:, i])

            # Plot curve
            ax.plot(recall, precision, lw=2)

            # Add random baseline
            baseline = np.sum(y_true[:, i]) / len(y_true[:, i])
            ax.plot([0, 1], [baseline, baseline], 'k--', lw=1)

            # Set title and labels
            ax.set_title(f"{class_name} (AP={ap:.2f})")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")

            # Set limits
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])

    # Hide empty subplots
    for i in range(n_classes, len(axes)):
        axes[i].axis("off")

    # Set overall title
    fig.suptitle(title, fontsize=16)

    # Tight layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_attention_weights(
        attention_weights: Union[np.ndarray, torch.Tensor],
        class_names: List[str],
        title: str = "Attention Weights",
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot attention weights as a heatmap.

    Args:
        attention_weights: Attention weights of shape [num_classes, num_classes].
        class_names: List of class names.
        title: Plot title.
        cmap: Colormap for heatmap.
        figsize: Figure size.
        save_path: Path to save the figure.

    Returns:
        Matplotlib figure.
    """
    # Convert to numpy if tensor
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        attention_weights,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Class")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Tight layout
    plt.tight_layout()

    # Save figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_graph_contribution(
        graph_weights: Union[np.ndarray, torch.Tensor],
        graph_names: List[str],
        title: str = "Graph Contribution",
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot contribution of each graph component.

    Args:
        graph_weights: Weights of each graph component.
        graph_names: Names of graph components.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the figure.

    Returns:
        Matplotlib figure.
    """
    # Convert to numpy if tensor
    if isinstance(graph_weights, torch.Tensor):
        graph_weights = graph_weights.detach().cpu().numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot bar chart
    bars = ax.bar(graph_names, graph_weights)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom"
        )

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel("Graph Component")
    ax.set_ylabel("Weight")

    # Set y-axis limits
    ax.set_ylim([0, max(graph_weights) * 1.2])

    # Tight layout
    plt.tight_layout()

    # Save figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_predictions(
        images: torch.Tensor,
        true_labels: torch.Tensor,
        pred_scores: torch.Tensor,
        class_names: List[str],
        threshold: float = 0.5,
        num_samples: int = 5,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot input images with true and predicted labels.

    Args:
        images: Input images of shape [batch_size, 3, H, W].
        true_labels: True binary labels of shape [batch_size, num_classes].
        pred_scores: Predicted scores of shape [batch_size, num_classes].
        class_names: List of class names.
        threshold: Threshold for converting scores to binary predictions.
        num_samples: Number of samples to plot.
        figsize: Figure size.
        save_path: Path to save the figure.

    Returns:
        Matplotlib figure.
    """
    # Convert to numpy
    images = images.detach().cpu()
    true_labels = true_labels.detach().cpu().numpy()
    pred_scores = pred_scores.detach().cpu().numpy()

    # Convert scores to binary predictions
    pred_labels = (pred_scores >= threshold).astype(int)

    # Create figure
    fig, axes = plt.subplots(num_samples, 1, figsize=figsize)

    # Ensure axes is iterable
    if num_samples == 1:
        axes = [axes]

    # Plot samples
    for i, ax in enumerate(axes):
        if i < min(num_samples, len(images)):
            # Get image and labels
            img = images[i].permute(1, 2, 0).numpy()
            true = true_labels[i]
            pred = pred_labels[i]
            scores = pred_scores[i]

            # Normalize image
            img = (img - img.min()) / (img.max() - img.min())

            # Plot image
            ax.imshow(img)

            # Get true and predicted classes
            true_classes = [class_names[j] for j in range(len(class_names)) if true[j] == 1]
            pred_classes = [f"{class_names[j]} ({scores[j]:.2f})" for j in range(len(class_names)) if pred[j] == 1]

            # Create label text
            true_text = "True: " + ", ".join(true_classes)
            pred_text = "Pred: " + ", ".join(pred_classes)

            # Add labels to plot
            ax.set_title(true_text + "\n" + pred_text)

            # Remove axis
            ax.axis("off")

    # Tight layout
    plt.tight_layout()

    # Save figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig

