# TODO: import all local files and functions for train.py
from configs.setup_env import device

# TODO: rename file to train.py
import logging
from typing import Optional
import matplotlib.pyplot as plt

# TODO: Add log dir (or file) to set up logging/logger
# TODO: Add logging setup
logging.basicConfig(
    level = logging.DEBUG, # Detailed info on bugs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("vit_debug.log", mode="w") # Save to file
    ]
)

from configs.model_args.model_args_large import ModelArgs
from configs.training_args import TrainingArgs
from save_load_checkpoints.save_checkpoint import save_checkpoint
from save_load_checkpoints.load_checkpoint import load_checkpoint
from src.vit import VisionTransformer
from training.training_components.setup_training_components import get_training_components
from training.train_val.training_loop import train
from training.train_val.validation_loop import validate
from training.setup_imagenet_data import setup_data_loaders

# Create logger object
# TODO: set up real logger
logger = logging.getLogger(__name__)

def main(resume_from_checkpoint: Optional[str] = None, patience: int = 3) -> None:
    """Main training loop.

    Args:
        resume_from (Optional[str]): Path to checkpoint to resume training from.
        patience (int): Number of epochs to wait for improvement before early stopping.
    """
    start_epoch = 0
    best_loss = float("inf")
    early_stop_counter = 0

    # Initialize lists for visualization
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    # Initialize model arguments and training arguments
    model_args = ModelArgs()
    training_args = TrainingArgs()

    # Initialize model
    model = VisionTransformer(model_args).to(device)

    # Get training components
    optimizer, scheduler, scaler = get_training_components(model, training_args)

    # Get dataloaders
    train_loader, val_loader = setup_data_loaders()

    # Resume from checkpoint if provided
    try:
        if resume_from_checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            checkpoint_info = load_checkpoint(
                resume_from_checkpoint, model, optimizer, scheduler, scaler, device
            )
            start_epoch = checkpoint_info['epoch'] + 1
            best_loss = checkpoint_info['loss']
    except Exception as e:
        logger.info(f"Failed to resume from {resume_from_checkpoint}: {e}")

    # Log parameters
    # TODO: change all print statements to log statements
    print(f"Starting training from epoch {start_epoch}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad()):,}")

    # Main training loop
    for epoch in range(start_epoch, training_args.epochs):
        print(f"\nEpoch {epoch + 1}/{training_args.epochs}")
        print("-" * 50)

        # Train
        train_loss, train_acc = train(
            model, train_loader, optimizer,
            training_args, scaler, epoch
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, training_args
        )

        # Update lr
        scheduler.step()

        # Track metrics for visualization
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}")

        # Save regular checkpoint
        if (epoch + 1) % 1 == 0:
            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                training_args, model_args, scaler, is_best=False
            )
            logger.info(f"saved checkpoint to {checkpoint_path}")

        # Save best checkpoint (based on validation loss)
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
            best_checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                training_args, model_args, scaler, is_best=True
            )
            logger.info(f"saved best checkpoint to {best_checkpoint_path}")
        else:
            early_stop_counter += 1

        # Early stopping
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

        return train_losses, train_accuracies, 

    plot_metrics()

def plot_metrics() -> None:
    """Plot training loss and validation accuracy over epochs."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()