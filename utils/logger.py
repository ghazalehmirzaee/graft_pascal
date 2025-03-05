"""
Logger utility for GRAFT framework.

This module provides a logger class for tracking and visualizing
training progress with support for console logging, file logging,
and Weights & Biases integration.
"""
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import json
import logging
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class GRAFTLogger:
    """
    Logger for GRAFT framework.

    This class provides logging functionality for the GRAFT framework,
    with support for console logging, file logging, and Weights & Biases integration.
    """

    def __init__(
            self,
            output_dir: str,
            project_name: str,
            use_wandb: bool = True,
            wandb_entity: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None,
            log_frequency: int = 10,
            run_name: Optional[str] = None
    ):
        """
        Initialize logger.

        Args:
            output_dir: Directory for saving logs and artifacts.
            project_name: Name of the project.
            use_wandb: Whether to use Weights & Biases.
            wandb_entity: W&B entity name (username or team name).
            config: Configuration dictionary for W&B.
            log_frequency: Frequency of logging during training.
            run_name: Optional custom name for the W&B run.
        """
        self.output_dir = output_dir
        self.project_name = project_name
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_entity = wandb_entity
        self.config = config
        self.log_frequency = log_frequency
        self.run_name = run_name

        # Initialize current_phase
        self.current_phase = None

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create subdirectories
        self.log_dir = os.path.join(output_dir, "logs")
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        self.vis_dir = os.path.join(output_dir, "visualizations")

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        # Set up logging
        self.setup_logging()

        # Initialize W&B
        if self.use_wandb:
            self.init_wandb()

        # Initialize metrics storage
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = None

        # Initialize best metrics
        self.best_val_metric = float("-inf")
        self.best_epoch = -1

        # Log initialization
        self.logger.info(f"Initialized logger for project: {project_name}")
        self.logger.info(f"Output directory: {output_dir}")

    def setup_logging(self):
        """
        Set up logging configuration.
        """
        # Create logger
        self.logger = logging.getLogger("GRAFT")
        self.logger.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # Create file handler
        log_file = os.path.join(self.log_dir, f"{self.project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def init_wandb(self):
        """
        Initialize Weights & Biases.
        """
        if not WANDB_AVAILABLE:
            self.logger.warning("Weights & Biases not available. Proceeding without W&B logging.")
            self.use_wandb = False
            return

        # Generate a descriptive run name if not provided
        run_name = self.run_name
        if run_name is None:
            # Extract key information from config for the name
            phase_str = f"{self.current_phase}" if self.current_phase else "all_phases"

            # Default values if config is not complete
            model_type = "vit"
            lr_str = ""
            graph_str = ""

            # Extract model and training info if available
            if self.config is not None:
                # Get model type
                if "model" in self.config and "backbone" in self.config["model"] and "name" in self.config["model"]["backbone"]:
                    model_type = self.config["model"]["backbone"]["name"]

                # Get learning rate
                if ("training" in self.config and "optimizer" in self.config["training"] and
                        "lr" in self.config["training"]["optimizer"]):
                    lr = self.config["training"]["optimizer"]["lr"]
                    lr_str = f"_lr{lr}" if lr else ""

                # Get graph info
                if "model" in self.config and "graphs" in self.config["model"]:
                    graphs_enabled = self.config["model"]["graphs"].get("enabled", False)
                    graph_str = "_with_graphs" if graphs_enabled else "_vision_only"

            # Create a timestamp for uniqueness
            timestamp = datetime.now().strftime("%m%d_%H%M")

            # Combine elements into a descriptive name
            run_name = f"{phase_str}_{model_type}{graph_str}{lr_str}_{timestamp}"

        # Initialize W&B run
        wandb.init(
            project=self.project_name,
            entity=self.wandb_entity,
            config=self.config,
            dir=self.output_dir,
            name=run_name
        )

        self.logger.info(f"Initialized W&B run: {wandb.run.name}")

    def set_phase(self, phase: str):
        """
        Set current training phase.

        Args:
            phase: Current phase ("phase1_backbone", "phase2_finetune", etc.).
        """
        self.current_phase = phase
        self.logger.info(f"Starting phase: {phase}")

        # Reset best metrics for new phase
        self.best_val_metric = float("-inf")
        self.best_epoch = -1

        # Update W&B run name to include the current phase
        if self.use_wandb and wandb.run is not None:
            # Extract the existing name and add phase if not already there
            current_name = wandb.run.name
            if not current_name.startswith(phase):
                new_name = f"{phase}_{current_name}"
                wandb.run.name = new_name
                self.logger.info(f"Updated W&B run name to: {new_name}")

    def log_metrics(self, metrics: Dict[str, Any], step: int, mode: str = "train"):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics.
            step: Current step.
            mode: Mode ("train", "val", or "test").
        """
        # Log to console
        if step % self.log_frequency == 0 or mode != "train":
            metric_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, (float, int)) else f"{k}: {v}" for k, v in metrics.items()])
            self.logger.info(f"{mode.capitalize()} [{step}] - {metric_str}")

        # Log to W&B
        if self.use_wandb:
            # Add mode prefix to metrics
            wandb_metrics = {f"{mode}/{k}": v for k, v in metrics.items() if isinstance(v, (float, int, np.number))}

            # Add step
            wandb_metrics["epoch"] = step

            # Add phase if available
            if self.current_phase is not None:
                wandb_metrics["phase"] = self.current_phase

            # Log to W&B
            wandb.log(wandb_metrics, step=step)

        # Store metrics
        metrics_copy = metrics.copy()
        metrics_copy["step"] = step

        if mode == "train":
            self.train_metrics.append(metrics_copy)
        elif mode == "val":
            self.val_metrics.append(metrics_copy)
        elif mode == "test":
            self.test_metrics = metrics_copy

    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """
        Log hyperparameters.

        Args:
            hyperparams: Dictionary of hyperparameters.
        """
        # Log to console
        self.logger.info(f"Hyperparameters: {json.dumps(hyperparams, indent=2)}")

        # Log to W&B
        if self.use_wandb:
            wandb.config.update(hyperparams, allow_val_change=True)

    def log_model_summary(self, model_summary: str):
        """
        Log model summary.

        Args:
            model_summary: Model summary string.
        """
        # Log to console
        self.logger.info(f"Model Summary:\n{model_summary}")

        # Log to file
        summary_file = os.path.join(self.log_dir, f"model_summary_{self.current_phase}.txt")
        with open(summary_file, "w") as f:
            f.write(model_summary)

        # Log to W&B
        if self.use_wandb:
            wandb.run.summary["model_summary"] = model_summary

    def save_checkpoint(
            self,
            model_state_dict: Dict,
            optimizer: torch.optim.Optimizer,
            epoch: int,
            metrics: Dict[str, Any],
            is_best: bool = False
    ):
        """
        Save model checkpoint.

        Args:
            model_state_dict: PyTorch model state dictionary.
            optimizer: PyTorch optimizer.
            epoch: Current epoch.
            metrics: Dictionary of metrics.
            is_best: Whether this is the best model so far.
        """
        # Create checkpoint
        checkpoint = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "phase": self.current_phase
        }

        # Save checkpoint
        phase_prefix = f"{self.current_phase}_" if self.current_phase is not None else ""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{phase_prefix}checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f"{phase_prefix}best_model.pth")
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {epoch}")

            # Update best metrics
            self.best_val_metric = metrics.get("mAP", metrics.get("val_mAP", 0.0))
            self.best_epoch = epoch

        # Log to W&B
        if self.use_wandb:
            # Save checkpoint as W&B artifact
            artifact = wandb.Artifact(
                name=f"model-{phase_prefix}{epoch}",
                type="model",
                description=f"Model checkpoint at epoch {epoch}",
                metadata=metrics
            )
            artifact.add_file(checkpoint_path, name=f"checkpoint_epoch_{epoch}.pth")
            wandb.log_artifact(artifact)

            # Track best model
            if is_best:
                best_artifact = wandb.Artifact(
                    name=f"best-model-{phase_prefix}",
                    type="model",
                    description=f"Best model at epoch {epoch}",
                    metadata=metrics
                )
                best_artifact.add_file(best_path, name="best_model.pth")
                wandb.log_artifact(best_artifact)

    def log_graph_visualizations(self, graph_data: Dict[str, Any], epoch: int):
        """
        Log graph visualizations.

        Args:
            graph_data: Dictionary containing graph data.
            epoch: Current epoch.
        """
        # Save graph visualizations locally
        phase_prefix = f"{self.current_phase}_" if self.current_phase is not None else ""
        vis_path = os.path.join(self.vis_dir, f"{phase_prefix}graph_vis_epoch_{epoch}.png")

        # Save to file
        if "fig" in graph_data:
            graph_data["fig"].savefig(vis_path, dpi=300, bbox_inches="tight")

        # Log to W&B
        if self.use_wandb:
            if "fig" in graph_data:
                wandb.log({f"graph_visualization/epoch_{epoch}": wandb.Image(graph_data["fig"])})

            # Log adjacency matrices
            for graph_name, adj_matrix in graph_data.items():
                if graph_name != "fig" and isinstance(adj_matrix, (np.ndarray, torch.Tensor)):
                    if isinstance(adj_matrix, torch.Tensor):
                        adj_matrix = adj_matrix.detach().cpu().numpy()

                    # Create heatmap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(adj_matrix, cmap="Blues")
                    plt.colorbar(im, ax=ax)
                    ax.set_title(f"{graph_name} Adjacency Matrix")

                    # Log to W&B
                    wandb.log({f"graph_adjacency/{graph_name}_epoch_{epoch}": wandb.Image(fig)})

                    # Close figure
                    plt.close(fig)

    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        Plot training curves.

        Args:
            save_path: Path to save the plot.
        """
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Extract metrics
        train_epochs = [m["step"] for m in self.train_metrics]
        val_epochs = [m["step"] for m in self.val_metrics]

        train_loss = [m.get("loss", 0.0) for m in self.train_metrics]
        val_loss = [m.get("loss", 0.0) for m in self.val_metrics]

        train_map = [m.get("mAP", 0.0) for m in self.train_metrics]
        val_map = [m.get("mAP", 0.0) for m in self.val_metrics]

        # Plot loss
        axes[0].plot(train_epochs, train_loss, "b-", label="Train")
        axes[0].plot(val_epochs, val_loss, "r-", label="Validation")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True)
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

        # Plot mAP
        axes[1].plot(train_epochs, train_map, "b-", label="Train")
        axes[1].plot(val_epochs, val_map, "r-", label="Validation")
        axes[1].set_title("Mean Average Precision (mAP)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("mAP")
        axes[1].legend()
        axes[1].grid(True)
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

        # Add best epoch marker
        if self.best_epoch >= 0:
            best_idx = val_epochs.index(self.best_epoch) if self.best_epoch in val_epochs else -1
            if best_idx >= 0:
                axes[1].plot(self.best_epoch, val_map[best_idx], "ro", markersize=10, label=f"Best (Epoch {self.best_epoch})")
                axes[1].legend()

        # Set title
        phase_title = f" - {self.current_phase}" if self.current_phase is not None else ""
        fig.suptitle(f"Training Curves{phase_title}")

        # Tight layout
        plt.tight_layout()

        # Save plot if path is provided
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        # Save to project directory
        phase_prefix = f"{self.current_phase}_" if self.current_phase is not None else ""
        default_path = os.path.join(self.vis_dir, f"{phase_prefix}training_curves.png")
        plt.savefig(default_path, dpi=300, bbox_inches="tight")

        # Log to W&B
        if self.use_wandb:
            wandb.log({"training_curves": wandb.Image(fig)})

        return fig

    def finish(self):
        """
        Finish logging.
        """
        # Plot training curves
        self.plot_training_curves()

        # Save metrics to JSON
        phase_prefix = f"{self.current_phase}_" if self.current_phase is not None else ""
        metrics_path = os.path.join(self.log_dir, f"{phase_prefix}metrics.json")

        metrics_data = {
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
            "best_val_metric": self.best_val_metric,
            "best_epoch": self.best_epoch
        }

        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)

        # Log completion
        self.logger.info("Logging finished")

        # Finish W&B run
        if self.use_wandb:
            # Log summary metrics
            wandb.run.summary["best_val_metric"] = self.best_val_metric
            wandb.run.summary["best_epoch"] = self.best_epoch

            if self.test_metrics is not None:
                for k, v in self.test_metrics.items():
                    if isinstance(v, (float, int, np.number)):
                        wandb.run.summary[f"test_{k}"] = v

            # Finish run
            wandb.finish()

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            Checkpoint dictionary.
        """
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        return checkpoint


def create_logger(
        config: Dict[str, Any],
        output_dir: Optional[str] = None,
        experiment_name: Optional[str] = None
) -> GRAFTLogger:
    """
    Create a GRAFT logger.

    Args:
        config: Configuration dictionary.
        output_dir: Output directory (overrides config if provided).
        experiment_name: Optional custom name for the experiment.

    Returns:
        GRAFTLogger instance.
    """
    # Get logger parameters from config
    project_name = config.get("project_name", "GRAFT-PASCAL")
    use_wandb = config.get("wandb", {}).get("enabled", True)
    wandb_entity = config.get("wandb", {}).get("entity", None)

    # Use provided output directory if any
    if output_dir is None:
        output_dir = config.get("output_dir", "./outputs")

    # Create logger
    logger = GRAFTLogger(
        output_dir=output_dir,
        project_name=project_name,
        use_wandb=use_wandb,
        wandb_entity=wandb_entity,
        config=config,
        log_frequency=10,
        run_name=experiment_name
    )

    return logger

