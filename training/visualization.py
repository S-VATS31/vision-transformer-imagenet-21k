from typing import List

import matplotlib.pyplot as plt

def plot_metrics(
    train_losses: List[float], 
    train_accuracies: List[float], 
    val_losses: List[float], 
    val_accuracies: List[float],
) -> None:
    """Plot training loss and validation accuracy over epochs.
    
    Args:
        train_lossses (List[float]): List containing all train losses.
        train_accuracies (List[float]): List containing all train accuracies.
        val_losses (List[float]): List containing all validation losses.
        val_accuracies (List[float]): List containing all validation accuracies.
    """
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