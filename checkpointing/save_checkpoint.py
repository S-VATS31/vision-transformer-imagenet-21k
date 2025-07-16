from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler

from configs.training_args import TrainingArgs
from configs.model_args.model_args_large import ModelArgs

# TODO: set up logger and add here
# TODO: set up logger name: ("checkpointing")

def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    loss: float,
    training_args: TrainingArgs,
    model_args: ModelArgs,
    scaler: Optional[GradScaler] = None,
    is_best: bool = False,
) -> str:
    """Save checkpoint to .pth file.
    
    Args:
        model (nn.Module): Transformer architecture.
        optimizer: PyTorch optimizer.
        scheduler: PyTorch scheduler.
        epoch (int): Current epoch to save checkpoint to.
        step (int): Current step to save checkpoint to.
        loss (float): Current loss to save checkpoint to.
        training_args (TrainingArgs): Training hyperparameters.
        model_args (ModelArgs): Model hyperparameters.
        scaler (Optional[GradScaler]): Save if GradScaler is not None.
        is_best (bool): Whether the current checkpoint contains the lowest validation loss or not.

    Returns:
        str: Returns path to save checkpoint so it can be loaded later.
    """
    try:
        # Create checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'training_args': training_args.__dict__,
            'model_args': model_args.__dict__,
        }

        # Add scaler state if using AMP
        if scaler is not None:
            checkpoint_data['scaler_state_dict'] = scaler.state_dict()
        
        # Create filename
        filename = "best_model.pt" if is_best else f"checkpoint_step_{step}_epoch{epoch}.pt"
        
        # Load checkpoint data to filename
        torch.save(checkpoint_data, filename)
        logger.info(f"Succesfully saved checkpoint to {filename}")
        
        return filename

    except Exception as e:
        logger.error(f"Failed to save checkpoint as {filename}: {e}")
        raise