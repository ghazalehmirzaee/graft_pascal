"""
Visualization utilities for GRAFT framework.

This module provides functions for visualizing model predictions,
graph structures, and attention weights.
"""
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
from sklearn.metrics import precision_recall_curve, average_precision_score


def plot_adjacency_matrix(
        adjacency_matrix: Union[np.ndarray, torch.Tensor],
        class_names: List[str],
        title: str = "Adjacency Matrix",
        cmap: str = "Blues",
        figsize: Tuple[int, int] = (12, 10),
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
    im = sns.heatmap(
        adjacency_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0.0,  # Ensure uniform scale
        vmax=max(1.0, np.max(adjacency_matrix))  # Cap at 1.0 or higher if needed
    )

    # Set title and labels
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Class", fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

    # Tight layout
    plt.tight_layout()

    # Save figure if save_path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_graph_network(
        adjacency_matrix: Union[np.ndarray, torch.Tensor],
        class_names: List[str],
        threshold: float = 0.2,
        title: str = "Graph Network",
        figsize: Tuple[int, int] = (14, 12),
        node_size_factor: int = 1000,
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
        node_size_factor: Base size for nodes.
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

    # Add nodes with sizes based on their average connections
    node_strengths = np.sum(adjacency_matrix, axis=1)
    node_strengths = node_strengths / np.max(node_strengths) if np.max(node_strengths) > 0 else node_strengths

    # Add nodes
    for i, name in enumerate(class_names):
        G.add_node(i, label=name, size=node_strengths[i] * node_size_factor + node_size_factor / 2)

    # Add edges with weights above threshold
    edge_count = 0
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            weight = adjacency_matrix[i, j]
            if weight > threshold:
                G.add_edge(i, j, weight=weight)
                edge_count += 1

    # If too few edges, lower the threshold adaptively
    if edge_count < len(class_names) // 2:
        new_threshold = np.percentile(adjacency_matrix[adjacency_matrix > 0], 50)
        for i in range(len(class_names)):
            for j in range(i + 1, len(class_names)):
                weight = adjacency_matrix[i, j]
                if threshold >= weight > new_threshold:
                    G.add_edge(i, j, weight=weight)

    # Get node positions using spring layout with seed for reproducibility
    pos = nx.spring_layout(G, seed=42, k=0.3)

    # Get node sizes
    node_sizes = [G.nodes[i].get('size', node_size_factor) for i in G.nodes()]

    # Get edge weights for width
    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]

    # Create color map based on node connectivity
    node_connectivity = dict(nx.degree(G))
    norm = plt.Normalize(min(node_connectivity.values()), max(node_connectivity.values()))
    cmap = plt.cm.viridis
    node_colors = [cmap(norm(node_connectivity[node])) for node in G.nodes()]

    # Draw graph
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(
        G, pos, ax=ax, width=edge_weights, edge_color="gray", alpha=0.6,
        connectionstyle='arc3,rad=0.1'  # Slightly curved edges for better visibility
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels={i: data['label'] for i, data in G.nodes(data=True)},
        font_size=10, font_weight='bold'
    )

    # Set title with edge count info
    ax.set_title(f"{title}\n({edge_count} connections above threshold {threshold:.2f})", fontsize=14, fontweight='bold')

    # Remove axis
    ax.axis("off")

    # Tight layout
    plt.tight_layout()

    # Save figure if save_path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_precision_recall_curves(
        y_true: Union[np.ndarray, torch.Tensor],
        y_score: Union[np.ndarray, torch.Tensor],
        class_names: List[str],
        title: str = "Precision-Recall Curves",
        figsize: Tuple[int, int] = (16, 14),
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

    # Calculate grid dimensions
    n_cols = min(5, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Make sure axes is 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Flatten axes for easy iteration
    axes_flat = axes.flatten()

    # Calculate average precision and sort classes by it
    ap_scores = []
    for i in range(n_classes):
        ap = average_precision_score(y_true[:, i], y_score[:, i])
        ap_scores.append((i, ap))

    # Sort by AP score descending
    ap_scores.sort(key=lambda x: x[1], reverse=True)

    # Plot precision-recall curve for each class in order of AP score
    for idx, (i, ap) in enumerate(ap_scores):
        if idx < len(axes_flat):
            ax = axes_flat[idx]

            # Get class name
            class_name = class_names[i]

            # Calculate class frequency
            class_freq = np.mean(y_true[:, i])

            # Compute precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_score[:, i])

            # Plot curve
            ax.plot(recall, precision, 'b-', lw=2)

            # Add random baseline
            ax.plot([0, 1], [class_freq, class_freq], 'k--', lw=1)

            # Set title and labels
            ax.set_title(f"{class_name} (AP={ap:.2f}, freq={class_freq:.3f})")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")

            # Set limits
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.6)

    # Hide empty subplots
    for i in range(len(ap_scores), len(axes_flat)):
        axes_flat[i].axis("off")

    # Set overall title
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Tight layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure if save_path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_attention_weights(
        attention_weights: Union[np.ndarray, torch.Tensor],
        class_names: List[str],
        title: str = "Attention Weights",
        cmap: str = "viridis",
        figsize: Tuple[int, int] = (12, 10),
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
    im = sns.heatmap(
        attention_weights,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0,
        vmax=max(1.0, np.max(attention_weights))
    )

    # Set title and labels
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Class", fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

    # Tight layout
    plt.tight_layout()

    # Save figure if save_path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_graph_contribution(
        graph_weights: Union[np.ndarray, torch.Tensor],
        graph_names: List[str],
        title: str = "Graph Contribution",
        figsize: Tuple[int, int] = (10, 8),
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

    # Define colors for bars
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(graph_names)))

    # Plot bar chart
    bars = ax.bar(graph_names, graph_weights, color=colors, alpha=0.8)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontweight='bold'
        )

    # Set title and labels
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Graph Component", fontsize=12)
    ax.set_ylabel("Weight", fontsize=12)

    # Set y-axis limits
    ax.set_ylim([0, max(graph_weights) * 1.2])

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Add dark borders to bars
    for bar in bars:
        bar.set_edgecolor('black')
        bar.set_linewidth(1)

    # Tight layout
    plt.tight_layout()

    # Save figure if save_path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_prediction_examples(
        images: torch.Tensor,
        true_labels: torch.Tensor,
        pred_scores: torch.Tensor,
        class_names: List[str],
        threshold: float = 0.5,
        num_samples: int = 6,
        num_cols: int = 3,
        figsize: Tuple[int, int] = (18, 12),
        save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot example predictions with true and predicted labels.

    Args:
        images: Input images of shape [batch_size, 3, H, W].
        true_labels: True binary labels of shape [batch_size, num_classes].
        pred_scores: Predicted scores of shape [batch_size, num_classes].
        class_names: List of class names.
        threshold: Threshold for converting scores to binary predictions.
        num_samples: Number of samples to plot.
        num_cols: Number of columns in the grid.
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
    pred_labels = (pred_scores >= threshold).astype(np.int32)

    # Determine number of rows
    num_rows = (min(num_samples, len(images)) + num_cols - 1) // num_cols

    # Create figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Make sure axes is a 2D array
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)

    # Determine normalization stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Plot samples
    for i in range(num_rows):
        for j in range(num_cols):
            idx = i * num_cols + j
            ax = axes[i, j]

            if idx < min(num_samples, len(images)):
                # Get image and labels
                img = images[idx]

                # Denormalize image
                img = img * std + mean
                img = img.permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)

                # Get true and predicted classes
                true = true_labels[idx]
                pred = pred_labels[idx]
                scores = pred_scores[idx]

                # Get true and predicted class names
                true_classes = [class_names[k] for k in range(len(class_names)) if true[k] == 1]
                pred_classes_with_scores = [f"{class_names[k]} ({scores[k]:.2f})" for k in range(len(class_names)) if
                                            pred[k] == 1]

                # If no predicted classes, show top 3 scores
                if not pred_classes_with_scores:
                    top_indices = np.argsort(-scores)[:3]
                    pred_classes_with_scores = [f"{class_names[k]} ({scores[k]:.2f})" for k in top_indices]

                # Limit to top 5 for display
                if len(true_classes) > 5:
                    true_classes = true_classes[:5] + [f"... and {len(true_classes) - 5} more"]
                if len(pred_classes_with_scores) > 5:
                    pred_classes_with_scores = pred_classes_with_scores[:5] + [
                        f"... and {len(pred_classes_with_scores) - 5} more"]

                # Create label text
                true_text = "True: " + ", ".join(true_classes)
                pred_text = "Pred: " + ", ".join(pred_classes_with_scores)

                # Determine prediction quality for title color
                correct_preds = sum((true[k] == 1 and pred[k] == 1) for k in range(len(true)))
                incorrect_preds = sum(
                    (true[k] == 0 and pred[k] == 1) or (true[k] == 1 and pred[k] == 0) for k in range(len(true)))
                title_color = 'green' if correct_preds > 0 and incorrect_preds == 0 else 'red' if correct_preds == 0 else 'orange'

                # Plot image
                ax.imshow(img)

                # Add labels to plot
                ax.set_title(f"Sample {idx}", color=title_color, fontweight='bold')
                y_pos = 0.02
                ax.text(0.02, y_pos, true_text, transform=ax.transAxes, fontsize=9,
                        verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.8))
                ax.text(0.02, y_pos + 0.15, pred_text, transform=ax.transAxes, fontsize=9,
                        verticalalignment='bottom', bbox=dict(facecolor='wheat', alpha=0.8))

                # Remove axis
                ax.axis("off")
            else:
                ax.axis("off")

    # Set overall title
    fig.suptitle("Example Predictions", fontsize=16, fontweight='bold')

    # Tight layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure if save_path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_class_distribution(
        y_true: Union[np.ndarray, torch.Tensor],
        class_names: List[str],
        title: str = "Class Distribution",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot class distribution in the dataset.

    Args:
        y_true: True binary labels of shape [n_samples, n_classes].
        class_names: List of class names.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the figure.

    Returns:
        Matplotlib figure.
    """
    # Convert to numpy if tensor
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Calculate class counts
    class_counts = np.sum(y_true, axis=0)

    # Calculate class percentages
    class_percentages = 100 * class_counts / len(y_true)

    # Sort classes by frequency
    sorted_indices = np.argsort(-class_counts)
    sorted_names = [class_names[i] for i in sorted_indices]
    sorted_counts = class_counts[sorted_indices]
    sorted_percentages = class_percentages[sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Define colors for bars
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(sorted_names)))

    # Plot bar chart
    bars = ax.bar(sorted_names, sorted_counts, color=colors, alpha=0.8)

    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, sorted_percentages)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 5,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=0 if pct > 5 else 90,
            fontweight='bold'
        )

    # Set title and labels
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=10)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Add dark borders to bars
    for bar in bars:
        bar.set_edgecolor('black')
        bar.set_linewidth(1)

    # Tight layout
    plt.tight_layout()

    # Save figure if save_path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_training_curves(
        train_metrics: List[Dict[str, float]],
        val_metrics: List[Dict[str, float]],
        metric_names: List[str] = ["loss", "mAP"],
        figsize: Tuple[int, int] = (16, 6),
        save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation metrics.

    Args:
        train_metrics: List of training metrics dictionaries.
        val_metrics: List of validation metrics dictionaries.
        metric_names: List of metric names to plot.
        figsize: Figure size.
        save_path: Path to save the figure.

    Returns:
        Matplotlib figure.
    """
    # Create figure with shared x-axis
    fig, axes = plt.subplots(1, len(metric_names), figsize=figsize, sharey=False)

    # Make sure axes is a list
    if len(metric_names) == 1:
        axes = [axes]

    # Extract epochs
    train_epochs = [m.get("step", i) for i, m in enumerate(train_metrics)]
    val_epochs = [m.get("step", i) for i, m in enumerate(val_metrics)]

    # Plot each metric
    for i, metric_name in enumerate(metric_names):
        ax = axes[i]

        # Extract metric values
        train_values = [m.get(metric_name, 0.0) for m in train_metrics]
        val_values = [m.get(metric_name, 0.0) for m in val_metrics]

        # Plot training curve
        ax.plot(train_epochs, train_values, 'b-', label="Train", linewidth=2)

        # Plot validation curve
        ax.plot(val_epochs, val_values, 'r-', label="Validation", linewidth=2)

        # Find best validation score
        if metric_name != "loss":
            best_idx = np.argmax(val_values)
            best_val = val_values[best_idx]
            best_epoch = val_epochs[best_idx]
        else:
            best_idx = np.argmin(val_values)
            best_val = val_values[best_idx]
            best_epoch = val_epochs[best_idx]

        # Mark best point
        ax.plot(best_epoch, best_val, 'go', markersize=8,
                label=f"Best: {best_val:.4f} (Epoch {best_epoch})")

        # Add horizontal line at best value
        ax.axhline(y=best_val, color='g', linestyle='--', alpha=0.5)

        # Add vertical line at best epoch
        ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)

        # Set title and labels
        title_metric = metric_name.upper() if metric_name == "map" else metric_name.capitalize()
        ax.set_title(f"{title_metric}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(title_metric, fontsize=12)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)

        # Add legend
        ax.legend(loc='best')

    # Set overall title
    fig.suptitle("Training and Validation Metrics", fontsize=16, fontweight='bold')

    # Tight layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure if save_path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig

