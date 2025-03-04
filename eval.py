"""
Evaluation script for GRAFT framework on PASCAL VOC dataset.

This script evaluates a trained GRAFT model on the validation set
of PASCAL VOC dataset and generates visualizations of the results.
"""
import os
import argparse
import yaml
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.pascal_voc import create_pascal_voc_dataloaders, get_class_names
from models.graft import create_graft_model
from utils.metrics import compute_metrics
from utils.logger import create_logger
from utils.visualization import (
    plot_adjacency_matrix,
    plot_graph_network,
    plot_precision_recall_curves,
    plot_class_distribution,
    plot_prediction_examples
)


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
        output_dir: str,
        batch_size: int = 32,
        num_visualization_samples: int = 10
) -> Dict[str, Any]:
    """
    Evaluate model.

    Args:
        model: GRAFT model.
        dataloader: Evaluation dataloader.
        device: Device for evaluation.
        class_names: List of class names.
        output_dir: Output directory for visualizations.
        batch_size: Batch size for processing.
        num_visualization_samples: Number of samples to visualize.

    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()

    # Create visualization directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Initialize arrays for metrics
    all_targets = []
    all_outputs = []
    visualization_images = []
    visualization_targets = []
    visualization_outputs = []

    # Evaluation loop with progress bar
    print(f"Evaluating model on {len(dataloader)} batches...")
    with torch.no_grad():
        for i, (images, targets, metadata) in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move data to device
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)
            logits = outputs["logits"]

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Store targets and outputs for computing metrics
            all_targets.append(targets.cpu())
            all_outputs.append(probs.cpu())

            # Store a subset of images for visualization
            if i < num_visualization_samples // batch_size + 1:
                visualization_images.append(images.cpu())
                visualization_targets.append(targets.cpu())
                visualization_outputs.append(probs.cpu())

    # Concatenate all targets and outputs
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_outputs = torch.cat(all_outputs, dim=0).numpy()

    # Compute evaluation metrics
    metrics = compute_metrics(all_targets, all_outputs)

    # Print summary metrics
    print("\nEvaluation Results:")
    print(f"Mean Average Precision (mAP): {metrics['mAP']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"Subset Accuracy: {metrics['subset_accuracy']:.4f}")

    # Save metrics to JSON
    metrics_file = os.path.join(output_dir, "metrics.json")

    # Convert numpy arrays to lists for JSON serialization
    serializable_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            serializable_metrics[k] = v.tolist()
        else:
            serializable_metrics[k] = v

    with open(metrics_file, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)

    # Generate visualizations

    # 1. Precision-Recall Curves
    if visualization_outputs:
        # Concatenate visualization data
        vis_images = torch.cat(visualization_images, dim=0)[:num_visualization_samples]
        vis_targets = torch.cat(visualization_targets, dim=0)[:num_visualization_samples]
        vis_outputs = torch.cat(visualization_outputs, dim=0)[:num_visualization_samples]

        # Plot example predictions
        pred_fig = plot_prediction_examples(
            vis_images,
            vis_targets,
            vis_outputs,
            class_names,
            threshold=0.5,
            num_samples=min(num_visualization_samples, len(vis_images)),
            save_path=os.path.join(vis_dir, "example_predictions.png")
        )

    # 2. PR Curves
    pr_curves_fig = plot_precision_recall_curves(
        all_targets,
        all_outputs,
        class_names,
        title="Precision-Recall Curves",
        save_path=os.path.join(vis_dir, "precision_recall_curves.png")
    )

    # 3. Class Distribution
    dist_fig = plot_class_distribution(
        all_targets,
        class_names,
        title="Class Distribution in Validation Set",
        save_path=os.path.join(vis_dir, "class_distribution.png")
    )

    # 4. Graph Visualizations (if available)
    if hasattr(model, "graph_components"):
        # Graph directory
        graph_dir = os.path.join(vis_dir, "graphs")
        os.makedirs(graph_dir, exist_ok=True)

        # Visualize each graph component
        for graph_name, graph in model.graph_components.items():
            if hasattr(graph, "get_adjacency_matrix"):
                adj_matrix = graph.get_adjacency_matrix()

                # Adjacency matrix heatmap
                adj_fig = plot_adjacency_matrix(
                    adj_matrix,
                    class_names,
                    title=f"{graph_name.capitalize()} Graph Adjacency Matrix",
                    save_path=os.path.join(graph_dir, f"{graph_name}_adjacency.png")
                )

                # Network visualization
                graph_fig = plot_graph_network(
                    adj_matrix,
                    class_names,
                    title=f"{graph_name.capitalize()} Graph Network",
                    save_path=os.path.join(graph_dir, f"{graph_name}_network.png")
                )

    print(f"\nEvaluation complete. Results saved to {output_dir}")
    return metrics


def main(args):
    """
    Main evaluation function.

    Args:
        args: Command-line arguments.
    """
    # Load config
    config = load_config(args.config)

    # Set device
    device_name = args.device if args.device else config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"Using device: {device}")

    # Create output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(config.get("output_dir", "./outputs"), "evaluation")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    dataset_config = config["dataset"]
    batch_size = args.batch_size if args.batch_size else dataset_config.get("batch_size", 32)
    num_workers = dataset_config.get("num_workers", 4)

    _, val_loader, _, _ = create_pascal_voc_dataloaders(
        root=dataset_config["root"],
        img_size=dataset_config["img_size"],
        batch_size=batch_size,
        num_workers=num_workers,
        mean=dataset_config["normalization"]["mean"],
        std=dataset_config["normalization"]["std"]
    )

    # Get class names
    class_names = get_class_names()

    # Create model
    print("Creating model...")
    num_classes = len(class_names)
    model = create_graft_model(num_classes, class_names, config["model"])
    model = model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Evaluate model
    start_time = time.time()
    metrics = evaluate(
        model=model,
        dataloader=val_loader,
        device=device,
        class_names=class_names,
        output_dir=output_dir,
        batch_size=batch_size,
        num_visualization_samples=args.num_vis_samples
    )
    elapsed_time = time.time() - start_time

    print(f"Evaluation completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GRAFT framework on PASCAL VOC dataset")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for visualizations")
    parser.add_argument("--device", type=str, default=None, help="Device to use (defaults to config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (defaults to config)")
    parser.add_argument("--num_vis_samples", type=int, default=10, help="Number of samples to visualize")
    args = parser.parse_args()

    main(args)

