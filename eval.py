"""
Evaluation script for GRAFT framework on PASCAL VOC dataset.

This script evaluates a trained GRAFT model on the validation set
of PASCAL VOC dataset and generates visualizations of the results.
"""
import os
import argparse
import yaml
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data.pascal_voc import create_pascal_voc_dataloaders, get_class_names
from models.graft import create_graft_model
from utils.metrics import compute_metrics
from utils.logger import create_logger
from utils.visualization import (
    plot_adjacency_matrix,
    plot_graph_network,
    plot_precision_recall_curves,
    plot_confusion_matrix,
    plot_predictions,
    plot_graph_contribution
)

# Set a fixed seed for reproducibility
SEED = 42


def set_seed(seed: int):
    """
    Set seed for reproducibility.

    Args:
        seed: Random seed.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        class_names: List[str],
        output_dir: str
) -> Dict[str, Any]:
    """
    Evaluate model.

    Args:
        model: GRAFT model.
        dataloader: Evaluation dataloader.
        device: Device for evaluation.
        class_names: List of class names.
        output_dir: Output directory for visualizations.

    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()

    # Evaluation metrics
    all_targets = []
    all_outputs = []
    all_images = []

    # Evaluation loop
    with torch.no_grad():
        for i, (images, targets, _) in enumerate(dataloader):
            # Move data to device
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)
            logits = outputs["logits"]

            # Store targets and outputs for computing metrics
            all_targets.append(targets.cpu())
            all_outputs.append(torch.sigmoid(logits).detach().cpu())

            # Store a few images for visualization
            if i < 5:  # Store only 5 batches to avoid memory issues
                all_images.append(images.cpu())

    # Concatenate all targets and outputs
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_outputs = torch.cat(all_outputs, dim=0).numpy()

    # Concatenate images (only a subset)
    if all_images:
        all_images = torch.cat(all_images, dim=0)

    # Compute evaluation metrics
    metrics = compute_metrics(all_targets, all_outputs)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate visualizations

    # 1. Precision-Recall Curves
    pr_curves_fig = plot_precision_recall_curves(
        all_targets,
        all_outputs,
        class_names,
        title="Precision-Recall Curves"
    )
    pr_curves_fig.savefig(os.path.join(output_dir, "precision_recall_curves.png"), dpi=300, bbox_inches="tight")

    # 2. Confusion Matrix (binarized outputs)
    pred_labels = (all_outputs >= 0.5).astype(int)
    conf_matrix_fig = plot_confusion_matrix(
        all_targets,
        pred_labels,
        class_names,
        title="Confusion Matrix"
    )
    conf_matrix_fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")

    # 3. Sample Predictions
    if all_images is not None and len(all_images) > 0:
        num_samples = min(10, len(all_images))  # Show at most 10 examples
        sample_idx = np.random.choice(len(all_images), num_samples, replace=False)

        sample_images = all_images[sample_idx]
        sample_targets = torch.from_numpy(all_targets[sample_idx])
        sample_outputs = torch.from_numpy(all_outputs[sample_idx])

        pred_fig = plot_predictions(
            sample_images,
            sample_targets,
            sample_outputs,
            class_names,
            threshold=0.5
        )
        pred_fig.savefig(os.path.join(output_dir, "sample_predictions.png"), dpi=300, bbox_inches="tight")

    # 4. Graph Visualizations
    if hasattr(model, "graph_components") and model.graphs_enabled:
        # 4.1. Adjacency Matrices
        for graph_name, graph in model.graph_components.items():
            if hasattr(graph, "get_adjacency_matrix"):
                adj_matrix = graph.get_adjacency_matrix()

                adj_fig = plot_adjacency_matrix(
                    adj_matrix,
                    class_names,
                    title=f"{graph_name} Adjacency Matrix"
                )
                adj_fig.savefig(os.path.join(output_dir, f"{graph_name}_adjacency.png"), dpi=300, bbox_inches="tight")

                # Also create graph network visualization
                graph_fig = plot_graph_network(
                    adj_matrix,
                    class_names,
                    title=f"{graph_name} Network"
                )
                graph_fig.savefig(os.path.join(output_dir, f"{graph_name}_network.png"), dpi=300, bbox_inches="tight")

        # 4.2. Graph Contribution
        if hasattr(model, "fusion_network") and hasattr(model.fusion_network, "log_uncertainties"):
            graph_weights = torch.exp(-model.fusion_network.log_uncertainties).detach().cpu().numpy()
            graph_weights = graph_weights / graph_weights.sum()  # Normalize

            contrib_fig = plot_graph_contribution(
                graph_weights,
                model.enabled_graphs,
                title="Graph Component Contribution"
            )
            contrib_fig.savefig(os.path.join(output_dir, "graph_contribution.png"), dpi=300, bbox_inches="tight")

    # Save metrics to JSON
    metrics_file = os.path.join(output_dir, "metrics.json")

    # Convert numpy arrays to lists for JSON serialization
    serializable_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            serializable_metrics[k] = v.tolist()
        elif isinstance(v, np.number):
            serializable_metrics[k] = float(v)
        else:
            serializable_metrics[k] = v

    with open(metrics_file, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)

    # Print summary metrics
    print("\nEvaluation Results:")
    print(f"Mean Average Precision (mAP): {metrics['mAP']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Average Sensitivity: {metrics['avg_sensitivity']:.4f}")
    print(f"Average Specificity: {metrics['avg_specificity']:.4f}")
    print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"Subset Accuracy: {metrics['subset_accuracy']:.4f}")

    # Per-class metrics
    print("\nPer-class Average Precision:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {metrics['AP'][i]:.4f}")

    return metrics


def main(args):
    """
    Main evaluation function.

    Args:
        args: Command-line arguments.
    """
    # Set fixed seed for reproducibility
    set_seed(SEED)

    # Load config
    config = load_config(args.config)

    # Set device
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Create dataloaders
    dataset_config = config["dataset"]

    _, val_loader, _, _ = create_pascal_voc_dataloaders(
        root=dataset_config["root"],
        img_size=dataset_config["img_size"],
        batch_size=dataset_config["batch_size"],
        num_workers=dataset_config["num_workers"],
        mean=dataset_config["normalization"]["mean"],
        std=dataset_config["normalization"]["std"]
    )

    # Get class names
    class_names = get_class_names()

    # Create model
    num_classes = len(class_names)
    model = create_graft_model(num_classes, class_names, config["model"])
    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print(f"Loaded model from {args.checkpoint}")

    # Create output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(config.get("output_dir", "./outputs"), "evaluation")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Evaluate model
    metrics = evaluate(
        model=model,
        dataloader=val_loader,
        device=device,
        class_names=class_names,
        output_dir=output_dir
    )

    print(f"\nEvaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GRAFT framework on PASCAL VOC dataset")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for visualizations")
    args = parser.parse_args()

    main(args)

