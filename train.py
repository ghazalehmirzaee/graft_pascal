"""
Training script for GRAFT framework on PASCAL VOC dataset with multi-GPU support.

This script implements the progressive training approach for the GRAFT framework,
with separate phases for backbone initialization, fine-tuning, graph construction,
progressive graph integration, and model refinement.
"""
import os
import argparse
import yaml
import random
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms

from data.pascal_voc import create_pascal_voc_dataloaders, get_class_names, get_num_classes, get_class_counts
from models.graft import create_graft_model
from utils.loss import create_loss_function
from utils.metrics import compute_metrics
from utils.logger import create_logger
from utils.visualization import (
    plot_adjacency_matrix,
    plot_graph_network,
    plot_precision_recall_curves
)


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
    # Commented out for better performance, but can be enabled for strict reproducibility
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


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
        local_rank: int = 0,
        distributed: bool = False,
        grad_accumulation_steps: int = 1  # Added gradient accumulation steps
) -> Dict[str, float]:
    """
    Train for one epoch with improved stability.

    Args:
        model: GRAFT model.
        dataloader: Training dataloader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device for training.
        epoch: Current epoch.
        logger: Logger.
        update_graph: Whether to update graph statistics.
        local_rank: Local rank for distributed training.
        distributed: Whether using distributed training.
        grad_accumulation_steps: Number of steps to accumulate gradients.

    Returns:
        Dictionary of training metrics.
    """
    model.train()

    # Training metrics
    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_outputs = []

    # Reset gradients at the beginning of the epoch
    optimizer.zero_grad()

    # Set epoch for the sampler to reshuffle data
    if distributed and isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(epoch)

    # Training loop
    for i, (images, targets, metadata_list) in enumerate(dataloader):
        # Move data to device
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward pass
        outputs = model(images)

        # Get logits - handle both DDP and non-DDP models
        if isinstance(model, DDP):
            logits = outputs["logits"]
        else:
            logits = outputs["logits"]

        # Compute loss with AMP-friendly approach
        loss = criterion(logits, targets)

        # Scale loss by accumulation steps
        loss = loss / grad_accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights every grad_accumulation_steps
        if (i + 1) % grad_accumulation_steps == 0:
            # Apply gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Unscale loss for logging
        loss_item = loss.item() * grad_accumulation_steps

        # Update graph statistics after optimization step to avoid memory buildup
        if update_graph and (i + 1) % grad_accumulation_steps == 0:
            with torch.no_grad():
                # Only update on non-distributed model or on the main process
                if not distributed or local_rank == 0:
                    # Process batch data for graph updates
                    batch_data = {
                        "labels": targets,
                        "features": outputs["features"].detach(),
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

                    # Update stats in the base model
                    if isinstance(model, DDP):
                        model.module.update_graph_statistics(batch_data)
                    else:
                        model.update_graph_statistics(batch_data)

        # Track metrics
        batch_size = images.size(0)
        total_loss += loss_item * batch_size * grad_accumulation_steps
        total_samples += batch_size

        # Compute output probabilities for metrics (avoid large memory allocations)
        with torch.no_grad():
            batch_probs = torch.sigmoid(logits).detach().cpu()

        # Store targets and outputs for computing metrics
        all_targets.append(targets.cpu())
        all_outputs.append(batch_probs)

        # Log batch metrics (less frequently to reduce overhead)
        if (i + 1) % (10 * grad_accumulation_steps) == 0 and logger is not None and local_rank == 0:
            logger.log_metrics({"loss": loss_item}, epoch * len(dataloader) + i, "train")

        # Clear memory cache periodically
        if i % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Process any remaining gradients
    if len(dataloader) % grad_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    # Gather metrics from all processes if distributed
    if distributed:
        try:
            # Gather all targets and outputs
            world_size = dist.get_world_size()
            gathered_targets = [None for _ in range(world_size)]
            gathered_outputs = [None for _ in range(world_size)]
            gathered_loss = torch.tensor([total_loss], device=device)
            gathered_samples = torch.tensor([total_samples], device=device)

            # All-reduce loss and samples
            dist.all_reduce(gathered_loss)
            dist.all_reduce(gathered_samples)

            # Concatenate local values
            local_targets = torch.cat(all_targets, dim=0)
            local_outputs = torch.cat(all_outputs, dim=0)

            # Gather from all processes
            dist.all_gather_object(gathered_targets, local_targets)
            dist.all_gather_object(gathered_outputs, local_outputs)

            # Only process metrics on rank 0
            if local_rank == 0:
                all_targets = [tensor for tensor in gathered_targets if tensor is not None]
                all_outputs = [tensor for tensor in gathered_outputs if tensor is not None]

                if all_targets and all_outputs:
                    all_targets = torch.cat(all_targets, dim=0)
                    all_outputs = torch.cat(all_outputs, dim=0)

                # Compute epoch metrics
                epoch_loss = gathered_loss.item() / gathered_samples.item()

                # Convert to numpy for metric computation
                all_targets_np = all_targets.numpy()
                all_outputs_np = all_outputs.numpy()

                # Compute evaluation metrics
                metrics = compute_metrics(all_targets_np, all_outputs_np)
                metrics["loss"] = epoch_loss

                return metrics
            return {}  # Return empty dict for non-zero ranks
        except Exception as e:
            # Handle potential NCCL errors
            if local_rank == 0:
                print(f"Error in distributed metrics gathering: {e}")
            return {"loss": total_loss / total_samples}
    else:
        # Non-distributed case - proceed as before
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
        visualize: bool = False,
        local_rank: int = 0,
        distributed: bool = False
) -> Dict[str, float]:
    """
    Validate model with improved stability.

    Args:
        model: GRAFT model.
        dataloader: Validation dataloader.
        criterion: Loss function.
        device: Device for validation.
        class_names: List of class names.
        epoch: Current epoch.
        logger: Logger.
        visualize: Whether to generate visualizations.
        local_rank: Local rank for distributed training.
        distributed: Whether using distributed training.

    Returns:
        Dictionary of validation metrics.
    """
    model.eval()

    # Validation metrics
    total_loss = 0.0
    total_samples = 0

    # Process validation data in chunks to avoid OOM
    chunk_size = 32
    all_targets_chunks = []
    all_outputs_chunks = []

    # Validation loop
    with torch.no_grad():
        for images, targets, metadata in dataloader:
            # Move data to device
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)

            # Get logits - handle both DDP and non-DDP models
            if isinstance(model, DDP):
                logits = outputs["logits"]
            else:
                logits = outputs["logits"]

            # Compute loss
            loss = criterion(logits, targets)

            # Track metrics
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Store targets and outputs for computing metrics
            all_targets_chunks.append(targets.cpu())
            all_outputs_chunks.append(torch.sigmoid(logits).cpu())

            # Clear memory cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Gather metrics from all processes if distributed
    if distributed:
        try:
            # Gather all targets and outputs
            world_size = dist.get_world_size()
            gathered_loss = torch.tensor([total_loss], device=device)
            gathered_samples = torch.tensor([total_samples], device=device)

            # All-reduce loss and samples
            dist.all_reduce(gathered_loss)
            dist.all_reduce(gathered_samples)

            # Concatenate local chunks
            local_targets = torch.cat(all_targets_chunks, dim=0)
            local_outputs = torch.cat(all_outputs_chunks, dim=0)

            # Gather from all processes
            gathered_targets = [None for _ in range(world_size)]
            gathered_outputs = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_targets, local_targets)
            dist.all_gather_object(gathered_outputs, local_outputs)

            # Only process metrics on rank 0
            if local_rank == 0:
                # Combine gathered data
                all_targets = torch.cat([tensor for tensor in gathered_targets if tensor is not None], dim=0)
                all_outputs = torch.cat([tensor for tensor in gathered_outputs if tensor is not None], dim=0)

                # Compute metrics
                epoch_loss = gathered_loss.item() / gathered_samples.item()

                # Convert to numpy for metric computation
                all_targets_np = all_targets.numpy()
                all_outputs_np = all_outputs.numpy()

                # Compute evaluation metrics
                metrics = compute_metrics(all_targets_np, all_outputs_np)
                metrics["loss"] = epoch_loss

                # Generate visualizations (only on rank 0)
                if visualize and logger is not None:
                    try:
                        # Create precision-recall curves
                        pr_curves_fig = plot_precision_recall_curves(
                            all_targets_np,
                            all_outputs_np,
                            class_names,
                            title=f"Precision-Recall Curves (Epoch {epoch})"
                        )

                        # Get graph visualizations if available
                        graph_data = {}

                        if isinstance(model, DDP):
                            model_to_check = model.module
                        else:
                            model_to_check = model

                        if hasattr(model_to_check, "graph_components") and model_to_check.graphs_enabled:
                            # Get adjacency matrices
                            for graph_name, graph in model_to_check.graph_components.items():
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

                    except Exception as e:
                        print(f"Error generating visualizations: {e}")

                return metrics
            return {}  # Return empty dict for non-zero ranks
        except Exception as e:
            if local_rank == 0:
                print(f"Error in distributed validation metrics: {e}")
            return {"loss": total_loss / total_samples}
    else:
        # Non-distributed case
        # Concatenate all targets and outputs
        all_targets = torch.cat(all_targets_chunks, dim=0).numpy()
        all_outputs = torch.cat(all_outputs_chunks, dim=0).numpy()

        # Compute epoch metrics
        epoch_loss = total_loss / total_samples

        # Compute evaluation metrics
        metrics = compute_metrics(all_targets, all_outputs)
        metrics["loss"] = epoch_loss

        # Generate visualizations
        if visualize and logger is not None:
            try:
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
            except Exception as e:
                print(f"Error generating visualizations: {e}")

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
        local_rank: int = 0,
        distributed: bool = False
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Train a specific phase with improved stability and error handling.

    Args:
        phase: Phase name.
        config: Configuration dictionary.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        class_names: List of class names.
        device: Device for training.
        logger: Logger.
        checkpoint_path: Path to checkpoint for resuming training.
        local_rank: Local rank for distributed training.
        distributed: Whether using distributed training.

    Returns:
        Tuple of (trained model, best metrics).
    """
    # Get phase-specific config
    phase_config = load_phase_config(config, phase)

    # Set logger phase (only on main process)
    if logger is not None and local_rank == 0:
        logger.set_phase(phase)
        # Log phase config
        logger.log_hyperparameters(phase_config)

    # Get number of classes
    num_classes = get_num_classes()

    # Get class counts for improved loss weighting
    samples_per_class = get_class_counts()

    # Gracefully handle potential failures
    try:
        # Create model
        model = create_graft_model(num_classes, class_names, phase_config["model"])
        model = model.to(device)

        # Wrap with DDP if distributed
        if distributed:
            # Sync batch norm across all processes
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            ddp_find_unused_parameters = phase_config.get("training", {}).get("ddp_find_unused_parameters", False)
            model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=ddp_find_unused_parameters)

        # Log model summary (only on main process)
        if logger is not None and local_rank == 0:
            if distributed:
                model_summary = str(model.module)
            else:
                model_summary = str(model)
            logger.log_model_summary(model_summary)

        # Create class weights from sample counts
        class_weights = None
        if samples_per_class is not None:
            if sum(samples_per_class) > 0:
                # Inverse frequency weighting with smoothing factor
                beta = 0.99
                effective_samples = [(1 - beta ** n) / (1 - beta) for n in samples_per_class]
                weights = [1.0 / max(1.0, s) for s in effective_samples]
                # Normalize weights
                total_weight = sum(weights)
                weights = [w / total_weight * num_classes for w in weights]
                class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

        # Create loss function
        criterion = create_loss_function(
            phase_config["loss"],
            num_classes,
            class_weights,
            samples_per_class
        )

        # Create optimizer
        optimizer_config = phase_config["training"]["optimizer"]
        optimizer_name = optimizer_config["name"].lower()

        # Convert learning rate from string to float if needed
        lr = optimizer_config["lr"]
        if isinstance(lr, str):
            lr = float(lr)

        # Scale learning rate by world size in distributed setting
        if distributed:
            world_size = dist.get_world_size()
            lr = lr * world_size
            if local_rank == 0 and logger is not None:
                logger.logger.info(f"Scaling learning rate by {world_size}x to {lr}")

        # Convert weight decay from string to float if needed
        weight_decay = optimizer_config.get("weight_decay", 0.0)
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)

        # Get gradient accumulation steps
        grad_accumulation_steps = phase_config.get("training", {}).get("grad_accumulation_steps", 1)

        # Convert momentum from string to float if needed (for SGD)
        momentum = optimizer_config.get("momentum", 0.9)
        if isinstance(momentum, str):
            momentum = float(momentum)

        # Initialize the appropriate optimizer with properly typed parameters
        if distributed:
            model_params = model.module.parameters()
        else:
            model_params = model.parameters()

        if optimizer_name == "adam":
            optimizer = optim.Adam(
                model_params,
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == "adamw":
            optimizer = optim.AdamW(
                model_params,
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(
                model_params,
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
        elif scheduler_name == "cosine_annealing":
            epochs = phase_config["training"].get("epochs", 100)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=eta_min
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
            try:
                if logger is not None and local_rank == 0:
                    checkpoint = logger.load_checkpoint(checkpoint_path)
                else:
                    checkpoint = torch.load(checkpoint_path, map_location=device)

                # Load model weights
                if distributed:
                    model.module.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint["model_state_dict"])

                # Load optimizer state if present
                if "optimizer_state_dict" in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    except Exception as e:
                        if local_rank == 0:
                            print(f"Warning: Could not load optimizer state: {e}")

                # Set start epoch and best metric
                start_epoch = checkpoint.get("epoch", 0) + 1
                if "metrics" in checkpoint and "mAP" in checkpoint["metrics"]:
                    best_metric = checkpoint["metrics"]["mAP"]
                    best_epoch = checkpoint.get("epoch", 0)

                if local_rank == 0 and logger is not None:
                    logger.logger.info(f"Resumed from epoch {start_epoch - 1} with best mAP: {best_metric:.4f}")
            except Exception as e:
                if local_rank == 0:
                    print(f"Error loading checkpoint: {e}")
                    if logger is not None:
                        logger.logger.warning(f"Failed to load checkpoint: {e}")

        # Training
        epochs = phase_config["training"]["epochs"]

        if logger is not None and local_rank == 0:
            logger.logger.info(f"Starting {phase} training for {epochs} epochs")

        # Get early stopping parameters
        early_stopping_config = phase_config["training"].get("early_stopping", {})
        use_early_stopping = early_stopping_config.get("enabled", False)
        patience = early_stopping_config.get("patience", 10)
        monitor = early_stopping_config.get("monitor", "val_mAP")

        # Initialize early stopping counter
        early_stop_counter = 0

        # Check if training is enabled for this phase
        if not phase_config["training"].get("enabled", True):
            if logger is not None and local_rank == 0:
                logger.logger.info(f"Training disabled for {phase}, skipping")

            # If graph construction phase, build graphs
            if phase == "phase3_graphs" and hasattr(model, "graph_components"):
                if distributed:
                    model_to_check = model.module
                else:
                    model_to_check = model

                if model_to_check.graphs_enabled and local_rank == 0:
                    if logger is not None:
                        logger.logger.info("Building graphs")

                    # Initialize graph components
                    for graph_name, graph in model_to_check.graph_components.items():
                        if hasattr(graph, "build_graph"):
                            try:
                                graph.build_graph()
                                if logger is not None:
                                    logger.logger.info(f"Built graph: {graph_name}")
                            except Exception as e:
                                if logger is not None:
                                    logger.logger.warning(f"Error building graph {graph_name}: {e}")

                    # Visualize graphs
                    if logger is not None:
                        try:
                            graph_data = {}
                            for graph_name, graph in model_to_check.graph_components.items():
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
                        except Exception as e:
                            logger.logger.warning(f"Error visualizing graphs: {e}")

            return model, {"phase": phase, "trained": False}

        # Create validation function with epoch parameter
        def validate_model(epoch):
            return validate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                class_names=class_names,
                epoch=epoch,
                logger=logger,
                visualize=(epoch % 5 == 0 or epoch == epochs - 1),  # Visualize every 5 epochs and last epoch
                local_rank=local_rank,
                distributed=distributed
            )

        # Initial validation to establish baseline
        if start_epoch == 0:
            try:
                val_metrics = validate_model(0)

                # Only main process logs metrics
                if local_rank == 0 or not distributed:
                    if logger is not None and val_metrics:
                        logger.log_metrics(val_metrics, 0, "val")

                    # Initialize best metric
                    current_metric = val_metrics.get("mAP", 0.0)
                    if current_metric > best_metric:
                        best_metric = current_metric
                        best_epoch = 0
            except Exception as e:
                if local_rank == 0:
                    print(f"Error during initial validation: {e}")

        # Training loop
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()

            # Train for one epoch
            try:
                train_metrics = train_epoch(
                    model=model,
                    dataloader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    epoch=epoch,
                    logger=logger,
                    update_graph=True,
                    local_rank=local_rank,
                    distributed=distributed,
                    grad_accumulation_steps=grad_accumulation_steps
                )

                # Log epoch time
                epoch_time = time.time() - epoch_start_time
                if local_rank == 0 and logger is not None:
                    logger.logger.info(f"Epoch {epoch} training completed in {epoch_time:.2f} seconds")

                # Update scheduler
                if scheduler is not None:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        if local_rank == 0 or not distributed:
                            scheduler.step(train_metrics.get("mAP", 0))
                    else:
                        scheduler.step()
            except Exception as e:
                if local_rank == 0:
                    print(f"Error during training epoch {epoch}: {e}")
                continue

            # Validate
            try:
                val_metrics = validate_model(epoch)
            except Exception as e:
                if local_rank == 0:
                    print(f"Error during validation epoch {epoch}: {e}")
                continue

            # Only main process handles logging and checkpointing
            if local_rank == 0 or not distributed:
                # Log metrics
                if logger is not None:
                    if train_metrics:  # Skip if empty (non-zero ranks in distributed mode)
                        logger.log_metrics(train_metrics, epoch, "train")
                    if val_metrics:  # Skip if empty (non-zero ranks in distributed mode)
                        logger.log_metrics(val_metrics, epoch, "val")

                # Check if this is the best model
                current_metric = val_metrics.get("mAP", 0.0)
                is_best = current_metric > best_metric

                if is_best:
                    best_metric = current_metric
                    best_epoch = epoch
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                # Save checkpoint
                if logger is not None:
                    # Get the model state dict
                    if distributed:
                        model_state_dict = model.module.state_dict()
                    else:
                        model_state_dict = model.state_dict()

                    # Save checkpoint less frequently to reduce IO overhead
                    if epoch % 5 == 0 or is_best or epoch == epochs - 1:
                        logger.save_checkpoint(
                            model_state_dict=model_state_dict,
                            optimizer=optimizer,
                            epoch=epoch,
                            metrics=val_metrics,
                            is_best=is_best
                        )

                # Early stopping
                if use_early_stopping and early_stop_counter >= patience:
                    if logger is not None:
                        logger.logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                    break

        # Make sure all processes reach here
        if distributed:
            dist.barrier()

        # On main process, load best model and finish phase
        if local_rank == 0 or not distributed:
            # Load best model
            phase_prefix = f"{phase}_" if phase is not None else ""
            best_model_path = os.path.join(logger.checkpoint_dir if logger is not None else ".",
                                           f"{phase_prefix}best_model.pth")

            if os.path.exists(best_model_path):
                try:
                    if logger is not None:
                        checkpoint = logger.load_checkpoint(best_model_path)
                    else:
                        checkpoint = torch.load(best_model_path, map_location=device)

                    if distributed:
                        model.module.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
                    else:
                        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))

                    best_metrics = checkpoint.get("metrics", val_metrics)
                except Exception as e:
                    print(f"Error loading best model: {e}")
                    best_metrics = val_metrics
            else:
                best_metrics = val_metrics

            # Finish phase
            if logger is not None:
                logger.logger.info(f"Finished {phase} training with best mAP: {best_metric:.4f} at epoch {best_epoch}")
                # Plot training curves
                logger.plot_training_curves()

            return model, best_metrics

        # For non-main processes in distributed mode, return what we have
        return model, {"phase": phase}

    except Exception as e:
        # Handle any unexpected errors
        if local_rank == 0:
            print(f"Error in train_phase for {phase}: {e}")
            import traceback
            traceback.print_exc()
            if logger is not None:
                logger.logger.error(f"Training error in {phase}: {e}")

        if distributed:
            # Try to safely abort all processes
            try:
                dist.barrier()
            except:
                pass

        # Return partial model if available
        return model if 'model' in locals() else None, {"phase": phase, "error": str(e)}


def create_dataloaders(
        config: Dict[str, Any],
        distributed: bool = False,
        local_rank: int = 0
) -> Tuple[DataLoader, DataLoader, Any, Any]:
    """
    Create dataloaders with distributed support and improved performance.

    Args:
        config: Configuration dictionary.
        distributed: Whether using distributed training.
        local_rank: Local rank for distributed training.

    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset)
    """
    dataset_config = config["dataset"]

    # Create dataloaders with pin_memory and persistent_workers for better performance
    train_loader, val_loader, train_dataset, val_dataset = create_pascal_voc_dataloaders(
        root=dataset_config["root"],
        img_size=dataset_config["img_size"],
        batch_size=dataset_config["batch_size"],
        num_workers=dataset_config["num_workers"],
        mean=dataset_config["normalization"]["mean"],
        std=dataset_config["normalization"]["std"],
        distributed=distributed,
        pin_memory=True,
        persistent_workers=(dataset_config["num_workers"] > 0),
        prefetch_factor=2,  # Prefetch 2 batches per worker
        drop_last=True  # Drop last incomplete batch for more stable training
    )

    if local_rank == 0:
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")

        # Log class distribution
        class_counts = get_class_counts()
        class_names = get_class_names()
        if class_counts:
            print("Class distribution:")
            for i, (name, count) in enumerate(zip(class_names, class_counts)):
                print(f"  {name}: {count} samples")

    return train_loader, val_loader, train_dataset, val_dataset


def main(args):
    """
    Main training function with improved error handling and cleanup.

    Args:
        args: Command-line arguments.
    """
    # Initialize distributed training
    local_rank = args.local_rank
    distributed = torch.cuda.device_count() > 1

    # Initialize process group before other operations
    if distributed:
        try:
            # Initialize process group
            torch.distributed.init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=60))
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)
            world_size = torch.distributed.get_world_size()
            if local_rank == 0:
                print(f"Using distributed training with {world_size} GPUs, local rank: {local_rank}")
        except Exception as e:
            print(f"Failed to initialize distributed environment: {e}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            distributed = False
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config - make sure to do this AFTER initializing distributed
    config = load_config(args.config)

    # Now we can use config safely
    if not distributed:
        # Only override device from config if we're not in distributed mode
        device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Create output directory
    output_dir = config.get("output_dir", "./outputs")
    if local_rank == 0:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)

    # Create logger (only on main process)
    logger = create_logger(config, output_dir, experiment_name=args.experiment_name,
                           local_rank=local_rank) if local_rank == 0 else None

    # Set device logging
    if local_rank == 0 and logger is not None:
        logger.logger.info(f"Using device: {device}")
        if distributed:
            logger.logger.info(f"Distributed training enabled with {world_size} GPUs")

    # Create dataloaders with distributed support
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
        config,
        distributed=distributed,
        local_rank=local_rank
    )

    if local_rank == 0 and logger is not None:
        logger.logger.info(f"Training set size: {len(train_dataset)}")
        logger.logger.info(f"Validation set size: {len(val_dataset)}")

    # Get class names
    class_names = get_class_names()
    if local_rank == 0 and logger is not None:
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
        if local_rank == 0 and logger is not None:
            logger.logger.info(f"Starting phase: {phase}")

        # Set checkpoint path from previous phase if available
        checkpoint_path = None
        if best_model is not None and phase != phases[0]:
            prev_phase = phases[phases.index(phase) - 1]
            checkpoint_path = os.path.join(logger.checkpoint_dir if logger is not None else ".",
                                           f"{prev_phase}_best_model.pth")

        # Train phase with error handling
        try:
            best_model, best_metrics = train_phase(
                phase=phase,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                class_names=class_names,
                device=device,
                logger=logger,
                checkpoint_path=checkpoint_path,
                local_rank=local_rank,
                distributed=distributed
            )
        except Exception as e:
            if local_rank == 0:
                print(f"Error in phase {phase}: {e}")
                import traceback
                traceback.print_exc()
                if logger is not None:
                    logger.logger.error(f"Phase error in {phase}: {e}")

            # Try to continue with next phase
            continue

    # Final evaluation - only on main process
    if best_model is not None and local_rank == 0 and logger is not None:
        logger.logger.info("Final evaluation")

        # Evaluate on validation set
        try:
            final_metrics = validate(
                model=best_model,
                dataloader=val_loader,
                criterion=create_loss_function(config["loss"], len(class_names)),
                device=device,
                class_names=class_names,
                epoch=0,  # Not relevant for final evaluation
                logger=logger,
                visualize=True,
                local_rank=local_rank,
                distributed=distributed
            )

            # Log final metrics
            logger.log_metrics(final_metrics, 0, "test")

            # Print summary
            logger.logger.info("Final metrics:")
            for k, v in final_metrics.items():
                if isinstance(v, (float, int)):
                    logger.logger.info(f"  {k}: {v:.4f}")

            # Save final model
            final_model_path = os.path.join(logger.checkpoint_dir, "final_model.pth")

            # Get the model state dict
            if distributed:
                model_state_dict = best_model.module.state_dict()
            else:
                model_state_dict = best_model.state_dict()

            torch.save({
                "model_state_dict": model_state_dict,
                "metrics": final_metrics,
            }, final_model_path)

            logger.logger.info(f"Final model saved to {final_model_path}")
        except Exception as e:
            logger.logger.error(f"Error in final evaluation: {e}")

    # Clean up distributed process group
    if distributed:
        try:
            dist.barrier()
            dist.destroy_process_group()
        except:
            pass

    # Finish logging
    if logger is not None and local_rank == 0:
        logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRAFT framework on PASCAL VOC dataset")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    parser.add_argument("--start_phase", type=str, default="phase1_backbone", help="Starting phase")
    parser.add_argument("--experiment_name", type=str, default=None, help="Custom name for the experiment in W&B")
    # Add this line for distributed training
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank for distributed training")
    args = parser.parse_args()

    main(args)

