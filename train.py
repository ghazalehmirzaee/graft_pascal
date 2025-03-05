"""
Training script for GRAFT framework on PASCAL VOC dataset with fixed seed.

This script implements the progressive training approach for the GRAFT framework,
with separate phases for backbone initialization, fine-tuning, graph construction,
progressive graph integration, and model refinement.
"""
import os
import argparse
import yaml
import random
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from data.pascal_voc import create_pascal_voc_dataloaders, get_class_names
from models.graft import create_graft_model
from utils.loss import create_loss_function
from utils.metrics import compute_metrics
from utils.logger import create_logger
from utils.visualization import (
    plot_adjacency_matrix,
    plot_graph_network,
    plot_precision_recall_curves
)

# Set a fixed seed for reproducibility
SEED = 42


def set_seed(seed: int):
    """
    Set seed for reproducibility.

    Args:
        seed: Random seed.
    """
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


def load_phase_config(config: Dict[str, Any], phase: str) -> Dict[str, Any]:
    """
    Load phase-specific configuration.

    Args:
        config: Base configuration.
        phase: Phase name.

    Returns:
        Phase-specific configuration.
    """
    # Get phase-specific config
    phase_config = config.get("phases", {}).get(phase, {})

    # Update base config with phase-specific config
    merged_config = config.copy()

    # Update nested dictionaries
    for key, value in phase_config.items():
        if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
            merged_config[key].update(value)
        else:
            merged_config[key] = value

    return merged_config


def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        epoch: int,
        logger: Any,
        update_graph: bool = True,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: GRAFT model.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device for training.
        epoch: Current epoch.
        logger: Logger.
        update_graph: Whether to update graph statistics.

    Returns:
        Dictionary of training metrics.
    """
    model.train()

    # Training metrics
    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_outputs = []

    # Training loop
    for i, (images, targets, metadata_list) in enumerate(dataloader):
        # Move data to device
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(images)
        logits = outputs["logits"]

        # Compute loss
        loss = criterion(logits, targets)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update graph statistics
        if update_graph:
            # Only update on non-distributed model or on the main process
            batch_data = {
                "labels": targets,
                "features": outputs["features"],
            }

            # Process bounding boxes if available in metadata
            if any("boxes" in meta for meta in metadata_list):
                # Filter and combine boxes from all samples in the batch
                all_boxes = []
                all_box_labels = []

                for idx, meta in enumerate(metadata_list):
                    if "boxes" in meta and len(meta["boxes"]) > 0:
                        sample_boxes = meta["boxes"]
                        sample_labels = meta.get("labels", [])

                        all_boxes.extend(sample_boxes)
                        all_box_labels.extend(sample_labels)

                if all_boxes:
                    batch_data["boxes"] = all_boxes
                    batch_data["box_labels"] = all_box_labels

            # Update stats in the model
            model.update_graph_statistics(batch_data)

        # Track metrics
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Store targets and outputs for computing metrics
        all_targets.append(targets.cpu())
        all_outputs.append(torch.sigmoid(logits).detach().cpu())

        # Log batch metrics
        if (i + 1) % 10 == 0 and logger is not None:
            logger.log_metrics({"loss": loss.item()}, epoch * len(dataloader) + i, "train")

    # Compute epoch metrics
    epoch_loss = total_loss / total_samples

    # Concatenate all targets and outputs
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_outputs = torch.cat(all_outputs, dim=0).numpy()

    # Compute evaluation metrics
    metrics = compute_metrics(all_targets, all_outputs)
    metrics["loss"] = epoch_loss

    return metrics


def validate(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        class_names: List[str],
        epoch: int,
        logger: Any,
        visualize: bool = False
) -> Dict[str, float]:
    """
    Validate model.

    Args:
        model: GRAFT model.
        dataloader: Validation dataloader.
        criterion: Loss function.
        device: Device for validation.
        class_names: List of class names.
        epoch: Current epoch.
        logger: Logger.
        visualize: Whether to generate visualizations.

    Returns:
        Dictionary of validation metrics.
    """
    model.eval()

    # Validation metrics
    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_outputs = []

    # Validation loop
    with torch.no_grad():
        for images, targets, metadata in dataloader:
            # Move data to device
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)
            logits = outputs["logits"]

            # Compute loss
            loss = criterion(logits, targets)

            # Track metrics
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Store targets and outputs for computing metrics
            all_targets.append(targets.cpu())
            all_outputs.append(torch.sigmoid(logits).detach().cpu())

    # Compute epoch metrics
    epoch_loss = total_loss / total_samples

    # Concatenate all targets and outputs
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_outputs = torch.cat(all_outputs, dim=0).numpy()

    # Compute evaluation metrics
    metrics = compute_metrics(all_targets, all_outputs)
    metrics["loss"] = epoch_loss

    # Generate visualizations
    if visualize and logger is not None:
        # Create precision-recall curves
        pr_curves_fig = plot_precision_recall_curves(
            all_targets,
            all_outputs,
            class_names,
            title=f"Precision-Recall Curves (Epoch {epoch})"
        )

        # Get graph visualizations if available
        graph_data = {}

        if hasattr(model, "graph_components") and model.graphs_enabled:
            # Get adjacency matrices
            for graph_name, graph in model.graph_components.items():
                if hasattr(graph, "get_adjacency_matrix"):
                    graph_data[graph_name] = graph.get_adjacency_matrix()

            # Create graph visualization
            if len(graph_data) > 0:
                # Create adjacency matrix visualization for one graph
                sample_graph_name = list(graph_data.keys())[0]
                adj_matrix_fig = plot_adjacency_matrix(
                    graph_data[sample_graph_name],
                    class_names,
                    title=f"{sample_graph_name} Adjacency Matrix (Epoch {epoch})"
                )

                # Add to graph data
                graph_data["fig"] = adj_matrix_fig

                # Log graph visualizations
                logger.log_graph_visualizations(graph_data, epoch)

        # Log precision-recall curves
        if hasattr(logger, "use_wandb") and logger.use_wandb:
            import wandb
            wandb.log({"pr_curves": wandb.Image(pr_curves_fig)})

    return metrics


def train_phase(
        phase: str,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_names: List[str],
        device: torch.device,
        logger: Any,
        checkpoint_path: Optional[str] = None,
        output_dir: str = "./outputs"
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Train a specific phase.

    Args:
        phase: Phase name.
        config: Configuration dictionary.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        class_names: List of class names.
        device: Device for training.
        logger: Logger.
        checkpoint_path: Path to checkpoint for resuming training.
        output_dir: Directory for saving outputs.

    Returns:
        Tuple of (trained model, best metrics).
    """
    # Set fixed seed for reproducibility in this phase
    set_seed(SEED)

    # Create metrics directory if it doesn't exist
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Get phase-specific config
    phase_config = load_phase_config(config, phase)

    # Set logger phase
    if logger is not None:
        logger.set_phase(phase)
        # Log phase config
        logger.log_hyperparameters(phase_config)

    # Create model
    num_classes = len(class_names)
    model = create_graft_model(num_classes, class_names, phase_config["model"])
    model = model.to(device)

    # Log model summary
    if logger is not None:
        model_summary = str(model)
        logger.log_model_summary(model_summary)

    # Create loss function
    class_weights = None
    if hasattr(train_loader.dataset, "class_weights"):
        class_weights = train_loader.dataset.class_weights.to(device)

    criterion = create_loss_function(phase_config["loss"], num_classes, class_weights)

    # Create optimizer
    optimizer_config = phase_config["training"]["optimizer"]
    optimizer_name = optimizer_config["name"].lower()

    # Convert learning rate from string to float if needed
    lr = optimizer_config["lr"]
    if isinstance(lr, str):
        lr = float(lr)

    # Convert weight decay from string to float if needed
    weight_decay = optimizer_config.get("weight_decay", 0.0)
    if isinstance(weight_decay, str):
        weight_decay = float(weight_decay)

    # Initialize the appropriate optimizer with properly typed parameters
    if optimizer_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        # Convert momentum from string to float if needed
        momentum = optimizer_config.get("momentum", 0.9)
        if isinstance(momentum, str):
            momentum = float(momentum)

        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Create scheduler
    scheduler_config = phase_config["training"]["scheduler"]
    scheduler_name = scheduler_config["name"].lower()

    # Convert T_0 to int if it's a string
    T_0 = scheduler_config.get("T_0", 5)
    if isinstance(T_0, str):
        T_0 = int(T_0)

    # Convert T_mult to int or float if it's a string
    T_mult = scheduler_config.get("T_mult", 1)
    if isinstance(T_mult, str):
        # If it contains a decimal point, convert to float, otherwise to int
        T_mult = float(T_mult) if '.' in T_mult else int(T_mult)

    # Get min_lr (eta_min) and convert to float if it's a string
    eta_min = optimizer_config.get("min_lr", 0.0)
    if isinstance(eta_min, str):
        eta_min = float(eta_min)

    # Get factor for ReduceLROnPlateau and convert if needed
    factor = scheduler_config.get("factor", 0.1)
    if isinstance(factor, str):
        factor = float(factor)

    # Get patience for ReduceLROnPlateau and convert if needed
    patience = scheduler_config.get("patience", 10)
    if isinstance(patience, str):
        patience = int(patience)

    if scheduler_name == "cosine_annealing_warm_restarts":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )
    elif scheduler_name == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=factor,
            patience=patience,
            verbose=True
        )
    elif scheduler_name == "none" or not scheduler_name:
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    # Load checkpoint if provided
    start_epoch = 0
    best_metric = float("-inf")
    best_epoch = -1

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        if logger is not None:
            checkpoint = logger.load_checkpoint(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Set start epoch and best metric
        start_epoch = checkpoint["epoch"] + 1
        if "metrics" in checkpoint and "mAP" in checkpoint["metrics"]:
            best_metric = checkpoint["metrics"]["mAP"]
            best_epoch = checkpoint["epoch"]

    # Training
    epochs = phase_config["training"]["epochs"]

    if logger is not None:
        logger.logger.info(f"Starting {phase} training for {epochs} epochs")

    # Get early stopping parameters
    early_stopping_config = phase_config["training"]["early_stopping"]
    use_early_stopping = early_stopping_config.get("enabled", False)
    es_patience = early_stopping_config.get("patience", 10)
    monitor = early_stopping_config.get("monitor", "val_mAP")

    # Initialize early stopping counter
    early_stop_counter = 0

    # Initialize lists to store metrics
    train_metrics_history = []
    val_metrics_history = []

    # Check if training is enabled for this phase
    if not phase_config["training"].get("enabled", True):
        if logger is not None:
            logger.logger.info(f"Training disabled for {phase}, skipping")

        # If graph construction phase, build graphs
        if phase == "phase3_graphs" and hasattr(model, "graph_components"):
            if model.graphs_enabled:
                if logger is not None:
                    logger.logger.info("Building graphs")

                # Initialize graph components
                for graph_name, graph in model.graph_components.items():
                    if hasattr(graph, "build_graph"):
                        graph.build_graph()

                # Visualize graphs
                if logger is not None:
                    graph_data = {}
                    for graph_name, graph in model.graph_components.items():
                        if hasattr(graph, "get_adjacency_matrix"):
                            graph_data[graph_name] = graph.get_adjacency_matrix()

                    # Create graph visualization
                    if len(graph_data) > 0:
                        sample_graph_name = list(graph_data.keys())[0]
                        adj_matrix_fig = plot_adjacency_matrix(
                            graph_data[sample_graph_name],
                            class_names,
                            title=f"{sample_graph_name} Adjacency Matrix"
                        )

                        # Add to graph data
                        graph_data["fig"] = adj_matrix_fig

                        # Log graph visualizations
                        logger.log_graph_visualizations(graph_data, 0)

        # Save metrics to JSON files
        metrics_path = os.path.join(metrics_dir, f"{phase}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump({"phase": phase, "trained": False}, f, indent=4)

        return model, {"phase": phase, "trained": False}

    # Training loop
    for epoch in range(start_epoch, epochs):
        # Train for one epoch
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            logger=logger,
            update_graph=True
        )

        # Store train metrics history
        train_metrics_history.append(train_metrics)

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_metrics.get("mAP", 0))
            else:
                scheduler.step()

        # Validate
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
            epoch=epoch,
            logger=logger,
            visualize=(epoch % 5 == 0)  # Visualize every 5 epochs
        )

        # Store validation metrics history
        val_metrics_history.append(val_metrics)

        # Log metrics
        if logger is not None:
            logger.log_metrics(train_metrics, epoch, "train")
            logger.log_metrics(val_metrics, epoch, "val")

        # Check if this is the best model
        current_metric = val_metrics.get("mAP", 0.0)
        is_best = current_metric > best_metric

        if is_best:
            best_metric = current_metric
            best_epoch = epoch
            early_stop_counter = 0

            # Save best model checkpoint
            best_model_path = os.path.join(output_dir, f"{phase}_best_model.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": val_metrics,
            }, best_model_path)

            if logger is not None:
                logger.logger.info(f"New best model saved at epoch {epoch} with mAP: {best_metric:.4f}")
        else:
            early_stop_counter += 1

        # Early stopping
        if use_early_stopping and early_stop_counter >= es_patience:
            if logger is not None:
                logger.logger.info(f"Early stopping triggered after {es_patience} epochs without improvement")
            break

    # Save metrics history to JSON files
    metrics_path = os.path.join(metrics_dir, f"{phase}_train_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(train_metrics_history, f, indent=4)

    metrics_path = os.path.join(metrics_dir, f"{phase}_val_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(val_metrics_history, f, indent=4)

    # Load best model
    best_model_path = os.path.join(output_dir, f"{phase}_best_model.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_metrics = checkpoint["metrics"]
    else:
        best_metrics = val_metrics

    # Finish phase
    if logger is not None:
        logger.logger.info(f"Finished {phase} training with best mAP: {best_metric:.4f} at epoch {best_epoch}")
        # Plot training curves
        logger.plot_training_curves()

    return model, best_metrics


def main(args):
    """
    Main training function.

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

    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.get("output_dir", "./outputs"), f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Create logger
    logger = create_logger(config, output_dir, experiment_name=args.experiment_name)

    # Create dataloaders
    dataset_config = config["dataset"]

    train_loader, val_loader, train_dataset, val_dataset = create_pascal_voc_dataloaders(
        root=dataset_config["root"],
        img_size=dataset_config["img_size"],
        batch_size=dataset_config["batch_size"],
        num_workers=dataset_config["num_workers"],
        mean=dataset_config["normalization"]["mean"],
        std=dataset_config["normalization"]["std"]
    )

    logger.logger.info(f"Training set size: {len(train_dataset)}")
    logger.logger.info(f"Validation set size: {len(val_dataset)}")

    # Get class names
    class_names = get_class_names()
    logger.logger.info(f"Classes: {class_names}")

    # Train through phases
    phases = [
        "phase1_backbone",
        "phase2_finetune",
        "phase3_graphs",
        "phase4_integration",
        "phase5_refinement"
    ]

    # Skip phases if specified
    start_phase = args.start_phase
    if start_phase in phases:
        start_idx = phases.index(start_phase)
        phases = phases[start_idx:]

    best_model = None
    best_metrics = None

    for phase in phases:
        logger.logger.info(f"Starting phase: {phase}")

        # Set checkpoint path from previous phase if available
        checkpoint_path = None
        if best_model is not None:
            prev_phase = phases[phases.index(phase) - 1]
            checkpoint_path = os.path.join(output_dir, f"{prev_phase}_best_model.pth")

        # Train phase
        best_model, best_metrics = train_phase(
            phase=phase,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            class_names=class_names,
            device=device,
            logger=logger,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir
        )

    # Final evaluation
    if best_model is not None:
        logger.logger.info("Final evaluation")

        # Evaluate on validation set
        final_metrics = validate(
            model=best_model,
            dataloader=val_loader,
            criterion=create_loss_function(config["loss"], len(class_names)),
            device=device,
            class_names=class_names,
            epoch=0,  # Not relevant for final evaluation
            logger=logger,
            visualize=True
        )

        # Log final metrics
        logger.log_metrics(final_metrics, 0, "test")

        # Save final metrics to JSON
        metrics_path = os.path.join(output_dir, "metrics", "final_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)

        # Print summary
        logger.logger.info("Final metrics:")
        for k, v in final_metrics.items():
            if isinstance(v, (float, int)):
                logger.logger.info(f"  {k}: {v:.4f}")

        # Save final model
        final_model_path = os.path.join(output_dir, "final_model.pth")
        torch.save({
            "model_state_dict": best_model.state_dict(),
            "metrics": final_metrics,
        }, final_model_path)

        logger.logger.info(f"Final model saved to {final_model_path}")

    # Finish logging
    logger.finish()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRAFT framework on PASCAL VOC dataset")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    parser.add_argument("--start_phase", type=str, default="phase1_backbone", help="Starting phase")
    parser.add_argument("--experiment_name", type=str, default=None, help="Custom name for the experiment in W&B")
    args = parser.parse_args()

    main(args)

