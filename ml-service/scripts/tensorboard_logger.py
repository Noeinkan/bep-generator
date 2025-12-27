"""
Advanced TensorBoard Logger for Deep Learning Training

This module provides a comprehensive TensorBoard logging interface for monitoring
neural network training with rich visualizations and metrics.

Usage:
    from tensorboard_logger import TensorBoardLogger

    logger = TensorBoardLogger(log_dir='runs/experiment_1', auto_start=True)
    logger.log_hyperparameters(config_dict)
    logger.log_epoch_metrics(epoch, train_loss, val_loss, lr, ...)
    logger.close()
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import subprocess
import sys
import webbrowser
import time
import atexit


class TensorBoardLogger:
    """
    Advanced TensorBoard Logger with comprehensive monitoring capabilities.

    Features:
    - Automatic TensorBoard server management
    - Rich metric logging (scalars, histograms, distributions, images)
    - Model architecture visualization
    - Gradient and weight tracking
    - Performance monitoring (GPU/CPU usage, throughput)
    - Text generation samples
    - Custom matplotlib plots

    Args:
        log_dir: Directory to save TensorBoard logs (default: auto-generated)
        auto_start: Automatically start TensorBoard server (default: True)
        port: TensorBoard server port (default: 6006)
        comment: Optional comment to append to log directory name
    """

    def __init__(self, log_dir=None, auto_start=True, port=6006, comment=''):
        """Initialize TensorBoard logger with automatic server startup"""

        # Create log directory
        if log_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = Path('runs') / f'training_{timestamp}{comment}'
        else:
            log_dir = Path(log_dir)

        self.log_dir = log_dir
        self.port = port
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.tensorboard_process = None

        print(f"\n{'='*60}")
        print(f"[TENSORBOARD] Logger initialized")
        print(f"[TENSORBOARD] Log directory: {log_dir}")
        print(f"{'='*60}\n")

        # Auto-start TensorBoard if requested
        if auto_start:
            self.start_tensorboard()

    def start_tensorboard(self):
        """Start TensorBoard server in background and open browser"""
        try:
            print(f"[START] Starting TensorBoard server...")

            # Start TensorBoard in background
            tensorboard_cmd = [
                sys.executable, '-m', 'tensorboard.main',
                '--logdir', str(self.log_dir.parent),
                '--port', str(self.port)
            ]

            self.tensorboard_process = subprocess.Popen(
                tensorboard_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )

            # Register cleanup on exit
            def cleanup():
                if self.tensorboard_process and self.tensorboard_process.poll() is None:
                    self.tensorboard_process.terminate()
                    print("\n[STOPPED] TensorBoard server stopped.")

            atexit.register(cleanup)

            # Wait for server to start
            time.sleep(3)

            # Open browser
            url = f'http://localhost:{self.port}'
            print(f"[OK] TensorBoard started!")
            print(f"[CHART] Dashboard URL: {url}")
            webbrowser.open(url)
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"[WARNING] Could not start TensorBoard automatically: {e}")
            print(f"[TIP] Start manually: tensorboard --logdir={self.log_dir.parent} --port={self.port}")
            print(f"{'='*60}\n")

    def log_hyperparameters(self, config):
        """
        Log hyperparameters as formatted text

        Args:
            config: Dictionary of hyperparameters
        """
        hyperparam_text = "## Training Configuration\n\n"
        for key, value in config.items():
            hyperparam_text += f"- **{key}**: {value}\n"

        self.writer.add_text('Config/Hyperparameters', hyperparam_text, 0)
        print(f"[OK] Hyperparameters logged to TensorBoard")

    def log_model_graph(self, model, input_sample):
        """
        Log model computational graph

        Args:
            model: PyTorch model
            input_sample: Sample input tensor for tracing
        """
        try:
            self.writer.add_graph(model, input_sample)
            print(f"[OK] Model graph logged to TensorBoard")
        except Exception as e:
            print(f"[WARNING] Could not log model graph: {e}")

    def log_epoch_metrics(self, epoch, train_loss, val_loss, val_acc_top1, val_acc_top5,
                         learning_rate, gradient_norm, batch_losses,
                         epochs_without_improvement, best_val_loss):
        """
        Log comprehensive metrics for a training epoch

        Args:
            epoch: Current epoch number
            train_loss: Average training loss
            val_loss: Average validation loss
            val_acc_top1: Top-1 validation accuracy
            val_acc_top5: Top-5 validation accuracy
            learning_rate: Current learning rate
            gradient_norm: L2 norm of gradients
            batch_losses: List of batch losses for distribution analysis
            epochs_without_improvement: Epochs since last improvement
            best_val_loss: Best validation loss so far
        """
        # Loss metrics
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/validation', val_loss, epoch)
        self.writer.add_scalar('Loss/best_val', best_val_loss, epoch)
        self.writer.add_scalar('Loss/batch_std', np.std(batch_losses), epoch)
        self.writer.add_scalar('Loss/batch_mean', np.mean(batch_losses), epoch)
        self.writer.add_scalar('Loss/batch_min', np.min(batch_losses), epoch)
        self.writer.add_scalar('Loss/batch_max', np.max(batch_losses), epoch)

        # Accuracy metrics
        self.writer.add_scalar('Accuracy/val_top1', val_acc_top1, epoch)
        self.writer.add_scalar('Accuracy/val_top5', val_acc_top5, epoch)
        self.writer.add_scalar('Accuracy/top5_minus_top1', val_acc_top5 - val_acc_top1, epoch)

        # Training dynamics
        self.writer.add_scalar('Training/learning_rate', learning_rate, epoch)
        self.writer.add_scalar('Training/gradient_norm', gradient_norm, epoch)
        self.writer.add_scalar('Training/epochs_without_improvement', epochs_without_improvement, epoch)

        # Overfitting indicator
        overfitting_gap = val_loss - train_loss
        self.writer.add_scalar('Analysis/overfitting_gap', overfitting_gap, epoch)

        # Loss distribution histogram
        self.writer.add_histogram('Distribution/batch_losses', np.array(batch_losses), epoch)

    def log_model_weights(self, model, epoch):
        """
        Log model weights and biases as histograms

        Args:
            model: PyTorch model
            epoch: Current epoch number
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Log weight distributions
                self.writer.add_histogram(f'Weights/{name}', param.data.cpu().numpy(), epoch)

                # Log weight statistics
                self.writer.add_scalar(f'WeightStats/{name}_mean', param.data.mean().item(), epoch)
                self.writer.add_scalar(f'WeightStats/{name}_std', param.data.std().item(), epoch)
                self.writer.add_scalar(f'WeightStats/{name}_min', param.data.min().item(), epoch)
                self.writer.add_scalar(f'WeightStats/{name}_max', param.data.max().item(), epoch)

    def log_gradients(self, model, epoch):
        """
        Log gradient distributions and statistics

        Args:
            model: PyTorch model
            epoch: Current epoch number
        """
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Log gradient distributions
                self.writer.add_histogram(f'Gradients/{name}', param.grad.data.cpu().numpy(), epoch)

                # Log gradient statistics
                grad_norm = param.grad.data.norm(2).item()
                self.writer.add_scalar(f'GradientNorms/{name}', grad_norm, epoch)
                self.writer.add_scalar(f'GradientStats/{name}_mean', param.grad.data.mean().item(), epoch)
                self.writer.add_scalar(f'GradientStats/{name}_std', param.grad.data.std().item(), epoch)

    def log_text_sample(self, text, epoch, tag='Generated_Text'):
        """
        Log generated text samples

        Args:
            text: Generated text string
            epoch: Current epoch number
            tag: Tag for organizing samples
        """
        self.writer.add_text(f'Samples/{tag}', text, epoch)

    def log_confusion_matrix(self, predictions, targets, epoch, num_classes=10):
        """
        Log confusion matrix as an image

        Args:
            predictions: Model predictions (tensor)
            targets: Ground truth labels (tensor)
            epoch: Current epoch number
            num_classes: Number of classes for the confusion matrix
        """
        try:
            import seaborn as sns
            from sklearn.metrics import confusion_matrix

            # Compute confusion matrix
            cm = confusion_matrix(
                targets.cpu().numpy(),
                predictions.cpu().numpy(),
                labels=list(range(num_classes))
            )

            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Confusion Matrix - Epoch {epoch}')

            # Convert to image and log
            self.log_matplotlib_figure(fig, epoch, tag='confusion_matrix')
            plt.close(fig)

        except ImportError:
            print("[WARNING] seaborn or sklearn not available for confusion matrix")
        except Exception as e:
            print(f"[WARNING] Could not log confusion matrix: {e}")

    def log_matplotlib_figure(self, figure, epoch, tag='custom_plot'):
        """
        Log a matplotlib figure as an image

        Args:
            figure: Matplotlib figure object
            epoch: Current epoch number
            tag: Tag for organizing plots
        """
        # Convert matplotlib figure to image tensor
        buf = io.BytesIO()
        figure.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        # Convert to numpy array
        from PIL import Image
        image = Image.open(buf)
        image_array = np.array(image)

        # Log to TensorBoard (HWC -> CHW)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        self.writer.add_image(f'Plots/{tag}', image_tensor, epoch)

        buf.close()

    def log_learning_curve(self, train_losses, val_losses, epoch):
        """
        Create and log a custom learning curve plot

        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            epoch: Current epoch number
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs_range = range(1, len(train_losses) + 1)

        ax.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        self.log_matplotlib_figure(fig, epoch, tag='learning_curve')
        plt.close(fig)

    def log_performance_metrics(self, epoch, batch_time, samples_per_sec, gpu_memory_used=None):
        """
        Log performance and efficiency metrics

        Args:
            epoch: Current epoch number
            batch_time: Average time per batch (seconds)
            samples_per_sec: Training throughput
            gpu_memory_used: GPU memory usage in GB (optional)
        """
        self.writer.add_scalar('Performance/batch_time_ms', batch_time * 1000, epoch)
        self.writer.add_scalar('Performance/samples_per_second', samples_per_sec, epoch)

        if gpu_memory_used is not None:
            self.writer.add_scalar('Performance/gpu_memory_gb', gpu_memory_used, epoch)

    def log_perplexity(self, loss, epoch):
        """
        Log perplexity metric (common for language models)

        Args:
            loss: Cross-entropy loss
            epoch: Current epoch number
        """
        perplexity = np.exp(loss)
        self.writer.add_scalar('Metrics/perplexity', perplexity, epoch)

    def log_prediction_distribution(self, predictions, epoch, top_k=10):
        """
        Log distribution of top-k predictions

        Args:
            predictions: Prediction logits or probabilities
            epoch: Current epoch number
            top_k: Number of top predictions to visualize
        """
        # Get top-k prediction indices and their frequencies
        top_preds = torch.topk(predictions, k=top_k, dim=1)[1]
        unique, counts = torch.unique(top_preds, return_counts=True)

        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(unique.cpu().numpy(), counts.cpu().numpy())
        ax.set_xlabel('Character Index')
        ax.set_ylabel('Prediction Frequency')
        ax.set_title(f'Top-{top_k} Prediction Distribution - Epoch {epoch}')

        self.log_matplotlib_figure(fig, epoch, tag='prediction_distribution')
        plt.close(fig)

    def flush(self):
        """Flush all pending writes to disk"""
        self.writer.flush()

    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()
        print(f"\n[OK] TensorBoard logger closed")
        print(f"[CHART] Logs saved to: {self.log_dir}")
        print(f"[TIP] View with: tensorboard --logdir={self.log_dir.parent}\n")
