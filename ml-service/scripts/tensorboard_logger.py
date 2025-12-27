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
    - Beginner-friendly explanations for all metrics

    Args:
        log_dir: Directory to save TensorBoard logs (default: auto-generated)
        auto_start: Automatically start TensorBoard server (default: True)
        port: TensorBoard server port (default: 6006)
        comment: Optional comment to append to log directory name
        model: PyTorch model for graph visualization (optional)
        n_vocab: Vocabulary size for embedding projections (optional)
        hyperparameters: Dictionary of hyperparameters to log (optional)
        device: Device (CPU/GPU) for logging (optional)
        sample_input: Sample input tensor for model graph (optional)
    """

    def __init__(self, log_dir=None, auto_start=True, port=6006, comment='',
                 model=None, n_vocab=None, hyperparameters=None, device='cpu',
                 sample_input=None):
        """Initialize TensorBoard logger with automatic server startup and explanations"""

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
        self.embeddings_logged = False  # Track if embeddings have been logged
        self.n_vocab = n_vocab
        self.device = device

        print(f"\n{'='*60}")
        print(f"[TENSORBOARD] Logger initialized")
        print(f"[TENSORBOARD] Log directory: {log_dir}")
        print(f"{'='*60}\n")

        # Log hyperparameters if provided
        if hyperparameters is not None:
            self.log_hyperparameters(hyperparameters)

        # Add beginner-friendly explanations (logged once at epoch 0)
        self._add_explanations()

        # Log model graph if model and sample input are provided
        if model is not None and sample_input is not None:
            self.log_model_graph(model, sample_input)

        # Auto-start TensorBoard if requested
        if auto_start:
            self.start_tensorboard()

    def _add_explanations(self):
        """Add beginner-friendly explanations for all TensorBoard metrics"""

        explanations = """
# TensorBoard Guide for Beginners

Welcome to TensorBoard! This dashboard helps you understand how your neural network is learning.
Below are explanations for each metric and visualization you'll see.

---

## Loss (Error Score)

**What it is:** Loss measures how wrong the model's predictions are. Lower is better!

- **Training Loss**: Error on data the model is learning from
- **Validation Loss**: Error on data the model has never seen (most important!)
- **Best Val Loss**: The lowest validation loss achieved so far

**What to look for:**
- Both losses should decrease over time
- If validation loss goes up while training loss goes down = **overfitting** (model memorizing instead of learning)
- Goal: Get validation loss as low as possible

**Overfitting Gap**: The difference between validation and training loss. Smaller is better!

---

## Accuracy

**What it is:** Percentage of correct predictions the model makes.

- **Top-1 Accuracy**: Model's first guess is correct
- **Top-5 Accuracy**: Correct answer is in model's top 5 guesses

**What to look for:**
- Higher is better (100% = perfect, but unrealistic)
- Top-5 should always be higher than Top-1
- Focus on validation accuracy (not training)

---

## Learning Rate

**What it is:** How big of a "step" the model takes when learning. Too big = unstable, too small = slow.

**What to look for:**
- Usually starts high and decreases over time
- If loss stops improving, learning rate will automatically reduce (LR Scheduler)
- When LR drops, you might see sudden improvements in loss

---

## Gradients

**What it is:** Gradients tell the model which direction to adjust its weights.

- **Gradient Norm**: Overall magnitude of gradients
- **Gradient Distributions**: Shows how gradients flow through each layer

**What to look for:**
- Very small gradients (< 0.001) = **vanishing gradients** (model not learning)
- Very large gradients (> 100) = **exploding gradients** (training unstable)
- Healthy range: 0.01 - 10
- Histograms should show smooth distributions, not stuck at zero

---

## Weights (Model Parameters)

**What it is:** The numbers the model adjusts to learn patterns in your data.

**What to look for:**
- Distributions should be smooth bell curves
- If all weights are near zero = model not learning
- If weights explode to very large values = unstable training
- Weights should gradually stabilize as training progresses

---

## Generated Text Samples

**What it is:** Text the model generates at different training stages.

**What to look for:**
- Early on: Random gibberish or repeated characters
- Mid-training: Words start forming, some grammatical structure
- Late training: Coherent sentences relevant to your BEP documents
- Quality should improve each epoch

---

## Early Stopping

**What it is:** Automatic mechanism to stop training if the model stops improving.

- **Epochs Without Improvement**: How many epochs passed with no better validation loss
- **Patience**: How many epochs to wait before stopping (default: 15)

**What to look for:**
- Counter resets to 0 when model improves
- When counter reaches patience limit = training stops
- Prevents wasted time training a model that won't improve

---

## Model Graph (Architecture)

**What it is:** Visual representation of how data flows through your neural network.

**What to look for:**
- **Input**: Character sequences (text data)
- **Embedding Layer**: Converts characters to dense vectors
- **RNN Layers** (LSTM/GRU): Process sequences and remember patterns
- **Output**: Predictions for next character

**How to use:** Navigate to the "GRAPHS" tab to see this visualization.

---

## Embeddings Projection

**What it is:** 3D visualization showing how the model represents each character internally.

**What to look for:**
- Similar characters (like vowels) should cluster together
- Clear separation between different character types
- Better organization = better understanding by the model

**How to use:** Navigate to the "PROJECTOR" tab to explore interactively.

---

## Performance Metrics

**What it is:** How fast and efficiently your training is running.

- **Batch Time**: How long each batch takes to process
- **Samples/Second**: Training throughput (higher = faster)
- **GPU Memory**: How much VRAM is being used (avoid hitting limits)

**What to look for:**
- Consistent batch times = stable training
- High throughput = efficient use of hardware
- GPU memory should be stable, not growing

---

## Distribution Histograms

**What it is:** Shows the spread and distribution of values for losses, weights, gradients, etc.

**What to look for:**
- Smooth, bell-shaped curves = healthy distributions
- Spikes at extremes = potential issues
- Distributions should evolve smoothly over training

---

## Quick Tips

1. **Most Important Metric**: Watch Validation Loss - it tells you if the model is actually learning
2. **Check for Overfitting**: If training loss decreases but validation loss increases, you're overfitting
3. **Be Patient**: Deep learning takes time. Don't stop training too early!
4. **Use Multiple Tabs**: Compare scalars, view text samples, inspect distributions together
5. **Experiment**: Try different hyperparameters and compare runs side-by-side

---

## Common Issues & Solutions

**Loss is not decreasing:**
- Try increasing learning rate
- Train for more epochs
- Check if data is properly preprocessed

**Validation loss increasing (overfitting):**
- Add more dropout
- Use less complex model (fewer layers/units)
- Get more training data
- Enable early stopping (already done!)

**Training is slow:**
- Increase batch size (if GPU memory allows)
- Use GPU instead of CPU
- Reduce model size

**Gradients vanishing/exploding:**
- Already using gradient clipping (max norm = 5.0)
- Try different RNN type (GRU instead of LSTM)
- Reduce number of layers

---

**Happy Training!**

For more help, visit: https://www.tensorflow.org/tensorboard/get_started
"""

        # Log the comprehensive guide
        self.writer.add_text('Guide/Complete_Beginners_Guide', explanations, 0)

        # Log individual metric explanations
        self._add_metric_explanations()

        print("[OK] Beginner-friendly explanations added to TensorBoard")

    def _add_metric_explanations(self):
        """Add individual explanations for each metric category"""

        # Loss explanation
        loss_explanation = """
### Understanding Loss Metrics

**Training Loss** (`Loss/train`):
- Error measured on the data the model is currently learning from
- Should steadily decrease over time
- If stuck at a high value, learning rate might be too low

**Validation Loss** (`Loss/validation`):
- **MOST IMPORTANT METRIC** - measures true model performance
- Error on data the model has never seen before
- This is what you should optimize for!
- If increasing while training loss decreases → overfitting

**Best Validation Loss** (`Loss/best_val`):
- Tracks the lowest validation loss achieved so far
- Shown as a horizontal reference line
- Model checkpoint is saved when this improves

**Batch Loss Statistics**:
- `batch_std`: Variation in loss across batches (lower = more stable)
- `batch_mean`: Average batch loss
- High variance might indicate need for larger batch size
"""
        self.writer.add_text('Guide/Loss_Metrics', loss_explanation, 0)

        # Accuracy explanation
        accuracy_explanation = """
### Understanding Accuracy Metrics

**Top-1 Accuracy** (`Accuracy/val_top1`):
- Percentage of times the model's first guess is correct
- For character prediction, this is quite challenging!
- 30-50% is actually good for character-level models

**Top-5 Accuracy** (`Accuracy/val_top5`):
- Percentage of times the correct answer is in the top 5 predictions
- Should be significantly higher than Top-1
- Indicates the model has learned meaningful patterns

**What's a good accuracy?**
- Character-level: 40-60% Top-1, 70-90% Top-5
- Don't expect 99%+ like image classification
- Focus on validation loss as primary metric
"""
        self.writer.add_text('Guide/Accuracy_Metrics', accuracy_explanation, 0)

        # Gradient explanation
        gradient_explanation = """
### Understanding Gradients

**Gradient Norm** (`Gradients/norm`):
- Overall strength of the learning signal
- Too small (< 0.001): Vanishing gradients - model barely learning
- Too large (> 100): Exploding gradients - training unstable
- Sweet spot: 0.1 - 10

**Gradient Histograms** (`Gradients/*`):
- Shows distribution of gradients for each layer
- Should be centered around small non-zero values
- Stuck at zero? Model stopped learning
- Very wide spread? Training might be unstable

**Your model has gradient clipping enabled (max norm: 5.0)**
This prevents exploding gradients automatically!
"""
        self.writer.add_text('Guide/Gradient_Metrics', gradient_explanation, 0)

        # Training dynamics explanation
        training_explanation = """
### Understanding Training Dynamics

**Learning Rate** (`Training/learning_rate`):
- Controls how fast the model learns
- Starts at initial value (usually 0.001)
- **Automatically reduces** when validation loss plateaus
- When you see a drop, expect loss to improve shortly after

**Early Stopping** (`Training/epochs_without_improvement`):
- Counts epochs since last improvement
- Resets to 0 when validation loss improves
- Training stops if patience limit reached (prevents wasting time)

**Overfitting Gap** (`Analysis/overfitting_gap`):
- Difference between validation and training loss
- Small gap (< 0.5): Good generalization
- Large gap (> 1.0): Model is overfitting
- Negative gap: Unusual, might indicate issues
"""
        self.writer.add_text('Guide/Training_Dynamics', training_explanation, 0)

        # Generated samples explanation
        samples_explanation = """
### Understanding Generated Text

Generated samples appear every 5 epochs under `Samples/Generated_Samples`.

**What to expect at different stages:**

**Epochs 1-10 (Early Training):**
- Random characters or gibberish
- Example: "thj bxz qwpz..."
- Might repeat same character
- **This is normal!** Model is just starting

**Epochs 10-30 (Mid Training):**
- Real words start appearing
- Some grammatical structure
- Example: "the project will include the..."
- Still makes mistakes and nonsensical phrases

**Epochs 30-50+ (Late Training):**
- Coherent sentences
- Contextually relevant to BEP documents
- Proper punctuation and spacing
- Example: "the bep establishes the information requirements..."

**Quality Check:**
Compare samples across epochs - should steadily improve!
"""
        self.writer.add_text('Guide/Generated_Samples', samples_explanation, 0)

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

            # Add explanation for the graph
            graph_explanation = """
### Model Architecture Graph

This visualization shows the complete computational flow of your neural network.

**How to navigate:**
1. Go to the **GRAPHS** tab in TensorBoard
2. Double-click nodes to expand/collapse layers
3. Hover over connections to see tensor shapes

**What you're seeing:**

**Input Layer**:
- Takes character sequences (encoded as integers)
- Shape: (batch_size, sequence_length)

**Embedding Layer**:
- Converts each character index to a dense vector
- Learns semantic relationships between characters
- Output shape: (batch_size, sequence_length, embedding_dim)

**RNN Layers** (LSTM or GRU):
- Process sequences while maintaining memory
- Multiple layers stack for deeper understanding
- Hidden states flow between time steps

**Output Layer** (Fully Connected):
- Maps hidden states to vocabulary predictions
- Output shape: (batch_size, vocab_size)
- Softmax applied to get probabilities

**Color coding:**
- Blue boxes: Operations (layers, activations)
- Yellow boxes: Variables (weights, biases)
- Gray boxes: Constants and inputs

**Useful tips:**
- Look for bottlenecks (very thin connections)
- Verify layer dimensions match expectations
- Check if all layers are properly connected
"""
            self.writer.add_text('Guide/Model_Graph', graph_explanation, 0)

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
