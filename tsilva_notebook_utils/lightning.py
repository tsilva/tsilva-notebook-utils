import os
import tempfile
import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import ToPILImage, v2

from tsilva_notebook_utils.video import (create_video_from_frames,
                                         save_tensor_frames)

DATASET_SPECS = {
    "imagenet": {
        "image_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "cifar10": {
        "image_size": 32,
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2023, 0.1994, 0.2010]
    },
    "mnist": {
        "image_size": 28,
        "mean": [0.1307],
        "std": [0.3081]
    }
}


class ThresholdStoppingCallback(pl.Callback):
    def __init__(self, metric, threshold):
        super().__init__()
        self.metric = metric
        self.threshold = threshold
        
    def on_train_epoch_end(self, trainer, pl_module):
        value = trainer.callback_metrics.get(self.metric)
        if value >= self.threshold:
            print(f"Stopping training as {self.metric} reached {self.threshold}")
            trainer.should_stop = True


class EpochTimeLogger(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start_time
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log({"epoch_time_sec": epoch_time, "epoch": trainer.current_epoch})


class BackboneWarmupCallback(pl.Callback):
    def __init__(self, unfreeze_at: Union[float, int]):
        if isinstance(unfreeze_at, float):
            assert 0.0 < unfreeze_at < 1.0, "Percentage must be between 0 and 1."
        elif isinstance(unfreeze_at, int):
            assert unfreeze_at >= 0, "Epoch number must be non-negative."
        else:
            raise TypeError("`unfreeze_at` must be a float (percentage) or an int (epoch number).")
        
        self.unfreeze_at = unfreeze_at
        self.unfrozen = False
        self.unfreeze_epoch = None

    def on_train_start(self, trainer, pl_module):
        if isinstance(self.unfreeze_at, float):
            self.unfreeze_epoch = int(self.unfreeze_at * trainer.max_epochs)
        else:
            self.unfreeze_epoch = self.unfreeze_at

    def on_train_epoch_start(self, trainer, pl_module):
        if self.unfrozen:
            return

        if trainer.current_epoch == 0:
            print(f"[Epoch {trainer.current_epoch}] Training with frozen backbone until epoch {self.unfreeze_epoch}...")
            pl_module.freeze_backbone()
        elif trainer.current_epoch >= self.unfreeze_epoch:
            print(f"[Epoch {trainer.current_epoch}] Unfroze backbone... now training all layers.")
            pl_module.unfreeze_backbone()
            self.unfrozen = True


def create_transforms(transforms_spec, compose=True):
    _transforms = []
    for name, args, kwargs in transforms_spec:
        _kwargs = kwargs.copy()
        _class = getattr(v2, name)
        if name == "ToDtype": _kwargs["dtype"] = getattr(torch, _kwargs["dtype"])
        transform_fn = _class(*args, **_kwargs)
        _transforms.append(transform_fn)
    return v2.Compose(_transforms) if compose else _transforms


def assert_transforms_in_device(transforms, device):
    img = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8, device=device)
    for t in transforms.transforms:
        out = t(img)
        name = t.__class__.__name__
        print(f"{name:<25} | Output device: {out.device}")
        assert out.device == device, f"{name}: expected device {device}, but got {out.device}"
        img = out


def create_dataset_transforms(
    dataset_id: str, 
    augmentation_pipeline: list = [], 
    pretrained_dataset_id: str = None
) -> tuple:
    # Retrieve dataset specs
    assert dataset_id in DATASET_SPECS, f"Unknown dataset spec: {dataset_id}"
    dataset_spec = DATASET_SPECS[dataset_id]
    pretrained_dataset_spec = DATASET_SPECS.get(pretrained_dataset_id, dataset_spec)

    dataset_image_size = dataset_spec["image_size"]
    pretrained_image_dataset_size = pretrained_dataset_spec["image_size"]
    assert dataset_image_size <= pretrained_image_dataset_size, f"Dataset size {dataset_image_size} must be less than or equal to pretrained dataset size {pretrained_image_dataset_size}"

    # Ensure image is tensor
    transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True)
    ]

    # Resize image to match pretrained dataset size
    if dataset_image_size < pretrained_image_dataset_size:
        crop_fraction = 0.875
        resize_size = int(round(pretrained_image_dataset_size / crop_fraction))
        transforms.extend([
            v2.Resize(resize_size),
            v2.CenterCrop(pretrained_image_dataset_size)
        ])

    # Add augmentation pipeline
    transforms.extend(
        create_transforms(augmentation_pipeline, compose=False)
    )

    # Normalize tensor to dataset statistics
    transforms.extend([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=pretrained_dataset_spec["mean"], std=pretrained_dataset_spec["std"])
    ])

    # Return composed transforms
    transform = v2.Compose(transforms)
    return transform


class ImageDataLoader(DataLoader):

    def render_video(self, n_batches=1, n_images=None, **kwargs):
        images = []
        for _ in range(n_batches):
            batch = next(iter(self))
            x, _ = batch
            images.extend(x.unbind(0))
            if n_images and len(images) >= n_images:
                images = images[:n_images]
                break

        with tempfile.TemporaryDirectory() as temp_dir:
            save_tensor_frames(images, temp_dir)
            return create_video_from_frames(temp_dir, **kwargs)


class RepeatedImageDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, index, n_samples):
        self.base_dataset = base_dataset
        self.index = index
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        image = self.base_dataset[self.index]
        return image


class BaseDataModule(pl.LightningDataModule):
    DatasetClass = None  # To be set by subclasses

    def __init__(
        self, 
        batch_size, 
        train_size, 
        seed, 

        train_shuffle=True,
        train_pin_memory=True,
        train_n_workers=8,
        train_persistent_workers=True,
        
        val_shuffle=False,
        val_pin_memory=True,
        val_n_workers=4,
        val_persistent_workers=False,

        test_shuffle=False,
        test_pin_memory=False,
        test_n_workers=2,
        test_persistent_workers=False,

        augmentation_pipeline=[],
        pretrained_dataset_id=None
    ):
        super().__init__()
        assert self.DatasetClass is not None, "DatasetClass must be set in subclass"
    
        self.class_names = None
        
        self.batch_size = batch_size
        self.train_size = train_size
        self.seed = seed

        self.train_shuffle = train_shuffle
        self.train_pin_memory = train_pin_memory
        self.train_n_workers = train_n_workers
        self.train_persistent_workers = train_persistent_workers
        
        self.val_shuffle = val_shuffle
        self.val_pin_memory = val_pin_memory
        self.val_n_workers = val_n_workers
        self.val_persistent_workers = val_persistent_workers

        self.test_shuffle = test_shuffle
        self.test_pin_memory = test_pin_memory
        self.test_n_workers = test_n_workers
        self.test_persistent_workers = test_persistent_workers

        self.augmentation_pipeline = augmentation_pipeline
        self.pretrained_dataset_id = pretrained_dataset_id
        self.class_names = None
        
        self._train_transform = create_dataset_transforms(self.dataset_id, self.augmentation_pipeline, pretrained_dataset_id=self.pretrained_dataset_id)
        self._val_transform = create_dataset_transforms(self.dataset_id, [], pretrained_dataset_id=self.pretrained_dataset_id)
        self._test_transform = create_dataset_transforms(self.dataset_id, [], pretrained_dataset_id=self.pretrained_dataset_id)
        self.transforms = {
            "train": lambda x: self._train_transform(x),
            "val" : lambda x: self._val_transform(x),
            "test": lambda x: self._test_transform(x)
        }
    
    @property
    def dataset_id(self):
        return self.DatasetClass.__name__.lower()
    
    @property
    def download_path(self):
        return f"./temp/{self.dataset_id}"
    
    def prepare_data(self):
        self.DatasetClass(root=self.download_path, train=True, download=True)
        self.DatasetClass(root=self.download_path, train=False, download=True)

    def setup(self, stage=None):
        if stage is None or stage == "fit":
            self.full = self.DatasetClass(root=self.download_path, train=True, transform=self.transforms["train"])
            self.class_names = list(self.full.classes)

            total = len(self.full)
            train_size = int(total * self.train_size)
            val_size = total - train_size
            self.train_set, self.val_set = torch.utils.data.random_split(
                self.full, [train_size, val_size], generator=torch.Generator().manual_seed(self.seed)
            )

            self.train_set.transform = self.transforms["train"]
            self.val_set.transform = self.transforms["val"]

        if stage is None or stage == "test":
            self.test_set = self.DatasetClass(root=self.download_path, train=False, transform=self.transforms["test"])

    def train_dataloader(self, **kwargs):
        return ImageDataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.train_n_workers,
            persistent_workers=self.train_persistent_workers,
            pin_memory=self.train_pin_memory,
            **kwargs
        )

    def val_dataloader(self, **kwargs):
        return ImageDataLoader(
            self.val_set, 
            batch_size=self.batch_size, 
            shuffle=self.val_shuffle, 
            num_workers=self.val_n_workers,
            pin_memory=self.val_pin_memory,
            persistent_workers=self.val_persistent_workers,
            **kwargs
        )

    def test_dataloader(self, **kwargs):
        return ImageDataLoader(
            self.test_set, 
            batch_size=self.batch_size, 
            shuffle=self.test_shuffle, 
            num_workers=self.test_n_workers,
            pin_memory=self.test_pin_memory,
            persistent_workers=self.test_persistent_workers,
            **kwargs
        )
    
    def repeated_dataloader(self, dataloader, n_samples=10, **kwargs):
        return ImageDataLoader(
            RepeatedImageDataset(dataloader.dataset, 0, n_samples), 
            batch_size=n_samples,
            **kwargs
        )

    def _get_split_dataset(self, split):
        return getattr(self, f"{split}_set")
    
    def get_classwise_dataloader(self, n_samples_per_class=5, split='train', num_workers=2, **kwargs):
        """
        Fast version: Use underlying dataset targets to avoid loading images during sampling.
        """
        dataset = self._get_split_dataset(split)

        # Access raw dataset
        if isinstance(dataset, torch.utils.data.Subset):
            base_dataset = dataset.dataset
            subset_indices = dataset.indices
        else:
            base_dataset = dataset
            subset_indices = list(range(len(dataset)))

        # Try to use the raw targets without loading each item
        targets = (
            np.array(base_dataset.targets)
            if hasattr(base_dataset, 'targets')
            else np.array([base_dataset[i][1] for i in range(len(base_dataset))])  # fallback
        )

        class_to_indices = {i: [] for i in range(len(self.class_names))}
        shuffle_indices = np.random.permutation(len(subset_indices))
        for idx in subset_indices:
            label = targets[idx]
            if len(class_to_indices[label]) < n_samples_per_class:
                class_to_indices[label].append(idx)
            if all(len(v) >= n_samples_per_class for v in class_to_indices.values()):
                break

        final_indices = [i for indices in class_to_indices.values() for i in indices]
        subset = torch.utils.data.Subset(base_dataset, final_indices)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, **kwargs)

    def render_transforms(self, split="train", n_samples=10, shuffle=True, fps=1, scale=4):
        skip_normalize_fn=lambda x: x if not isinstance(x, v2.Normalize) else None
        with self.no_augmentations(filter=skip_normalize_fn):
            dataloader_fn = getattr(self, f"{split}_dataloader")
            dataloader = dataloader_fn()
            repeated_dataloader = self.repeated_dataloader(dataloader, n_samples=n_samples, shuffle=shuffle)
            return repeated_dataloader.render_video(fps=fps, scale=scale)


    class _NoAugmentations:
        def __init__(self, outer, filter=None):
            self.outer = outer
            self.filter = filter
            self.prev_train_transform = None
            self.prev_val_transform = None
            self.prev_test_transform = None

        def __enter__(self):
            self.prev_train_transform = self.outer._train_transform
            self.prev_val_transform = self.outer._val_transform
            self.prev_test_transform = self.outer._test_transform
            default_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float, scale=True)])
            self.outer._train_transform = v2.Compose([x for x in self.outer._train_transform.transforms if self.filter(x)]) if self.filter else default_transform
            self.outer._val_transform = v2.Compose([x for x in self.outer._val_transform.transforms if self.filter(x)]) if self.filter else default_transform
            self.outer._test_transform = v2.Compose([x for x in self.outer._test_transform.transforms if self.filter(x)]) if self.filter else default_transform

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.outer._train_transform = self.prev_train_transform
            self.outer._val_transform = self.prev_val_transform
            self.outer._test_transform = self.prev_test_transform
            self.prev_train_transform = None
            self.prev_val_transform = None
            self.prev_test_transform = None

    def no_augmentations(self, *args, **kwargs):
        return self._NoAugmentations(self, *args, **kwargs)


class MNISTDataModule(BaseDataModule):
    DatasetClass = MNIST


class CIFAR10DataModule(BaseDataModule):
    DatasetClass = CIFAR10


def create_data_module(config, **kwargs):
    dataset_id = config['dataset_id']
    dataset_modules = {
        "mnist": MNISTDataModule,
        "cifar10": CIFAR10DataModule
    }
    dataset_id = dataset_id.lower()
    datamodule_class = dataset_modules.get(dataset_id)
    assert datamodule_class is not None, f"Unsupported dataset: {dataset_id}"
    datamodule = datamodule_class(**{
        "seed": config['seed'],
        "batch_size": config['batch_size'],
        "train_size": config['train_size'],
        "augmentation_pipeline": config.get('augmentation_pipeline', []),
        "pretrained_dataset_id": config.get('pretrained_dataset_id', None),
        **kwargs
    })
    datamodule.prepare_data()
    return datamodule


def render_samples_per_class(dm, n_samples=5, split='train'):
    """
    Show `n_samples` random images for each class from a classification dataset,
    using class-wise sampling.

    Args:
        dm: A Lightning DataModule with train/val/test dataloaders and:
            - .class_names: list of class names
            - .mean and .std: for unnormalizing images (optional)
            - .get_classwise_dataloader(n_samples_per_class, split): custom method
        n_samples: Number of images per class to show
        split: 'train', 'val', or 'test'
    """
    # Get class names
    class_names = getattr(dm, 'class_names', None)
    if class_names is None:
        raise ValueError("DataModule must define `class_names` for this function to work.")
    n_classes = len(class_names)

    # Use classwise dataloader
    loader = dm.get_classwise_dataloader(n_samples_per_class=n_samples, split=split, num_workers=4)

    # Collect samples
    images_per_class = {i: [] for i in range(n_classes)}
    for batch in loader:
        x, y = batch
        for img, label in zip(x, y):
            label = int(label)
            if len(images_per_class[label]) < n_samples:
                images_per_class[label].append(img)

    # Plotting
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(n_samples * 2, n_classes * 2))

    for class_idx in range(n_classes):
        for j in range(n_samples):
            ax = axes[class_idx, j] if n_classes > 1 else axes[j]
            if len(images_per_class[class_idx]) <= j:
                ax.axis('off')
                continue

            img = images_per_class[class_idx][j]
            img = img.permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)

            ax.imshow(img, interpolation='nearest')

            # Dynamically compute tick locations based on image size
            height, width = img.shape[:2]
            tick_step = max(1, width // 8)
            xticks = np.arange(0, width + 1, tick_step)
            yticks = np.arange(0, height + 1, tick_step)

            ax.set_xticks(xticks)
            ax.set_yticks(yticks)

            # Y tick labels always
            ax.set_yticklabels([str(t) for t in yticks])

            # X tick labels only on last row
            if class_idx == n_classes - 1:
                ax.set_xticklabels([str(t) for t in xticks])

            ax.tick_params(labelsize=6)
            ax.grid(color='gray', linestyle='--', linewidth=0.5)

            # Left column: class name
            if j == 0:
                ax.set_ylabel(class_names[class_idx], fontsize=10, rotation=90, labelpad=10)
            else:
                ax.set_yticklabels([])

    plt.tight_layout()
    return plt


def seed_everything(*args, workers=True, **kwargs):
    return pl.seed_everything(*args, workers=workers, **kwargs)


def overfit_batches(model, datamodule, overfit_batches=1):
    trainer = pl.Trainer(
        overfit_batches=overfit_batches,
        max_epochs=100,
        enable_checkpointing=False,
        callbacks=[ThresholdStoppingCallback("train/acc", 1.0)]
    )
    return trainer.fit(model, datamodule=datamodule)


class StopOnLambda(pl.Callback):
    """
    Stop training when a user-defined lambda condition on metrics is True.
    Example:
        StopOnLambda(lambda metrics: metrics.get('reward_mean', -float('inf')) >= 475)
    """
    def __init__(self, condition, message="Stopping criterion met."):
        super().__init__()
        self.condition = condition
        self.message = message

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        metrics = trainer.callback_metrics
        if self.condition(metrics):
            print(self.message)
            trainer.should_stop = True
