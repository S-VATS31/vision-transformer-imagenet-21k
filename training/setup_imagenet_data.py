from typing import Tuple

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def setup_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Setup up training and validation data loaders.

        Returns:
            DataLoader: Data loader containing all training examples.
            DataLoader: Data loader containing all validation examples.
        """
        # Training transforms with data augmentation
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.model_args.img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=self.training_args.color_jitter,
                contrast=self.training_args.color_jitter,
                saturation=self.training_args.color_jitter,
                hue=0.1
            ),
            transforms.RandomRotation(degrees=15),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)
            if self.training_args.auto_augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=self.training_args.random_erasing_prob, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])

        # Validation transforms
        val_transform = transforms.Compose([
            transforms.Resize(int(self.model_args.img_size * 1.14)),
            transforms.CenterCrop(self.model_args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        # Get training dataset
        try:
            train_dataset = datasets.ImageFolder(
                root="/data/imagenet1k/train",
                transform=train_transform
            )
        except FileNotFoundError:
            raise FileNotFoundError("ImageNet training dataset cannot be found.") from None
        except RuntimeError as e:
            raise RuntimeError(f"Failure occured when trying to open ImageNet training dataset: {e}.") from None

        # Get validation dataset
        try:
            val_dataset = datasets.ImageFolder(
                root= "/data/imagenet1k/val",
                transform=val_transform
            )
        except FileNotFoundError:
            raise FileNotFoundError("ImageNet validation dataset cannot be found.") from None
        except RuntimeError as e:
            raise RuntimeError(f"Error occured when trying to open ImageNet validation dataset: {e}.") from None

        # Train loader
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.training_args.batch_size,
            shuffle=True,
            num_workers=self.training_args.num_workers,
            pin_memory=self.training_args.pin_memory,
            persistent_workers=self.training_args.persistent_workers
        )

        # Validation loader
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.training_args.batch_size,
            shuffle=False,
            num_workers=self.training_args.num_workers,
            pin_memory=self.training_args.pin_memory,
            persistent_workers=self.training_args.persistent_workers
        )

        # TODO: change all print statements to log statements
        # Dataset statistics
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Number of classes: {len(train_dataset.classes)}")

        return train_loader, val_loader
