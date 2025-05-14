import time
import torch
import numpy as np
from typing import Union
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

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


def build_dataset_transforms(dataset_id: str, augmentation_pipeline: list = []):
    assert dataset_id in DATASET_SPECS, f"Unknown dataset spec: {dataset_id}"
    spec = DATASET_SPECS[dataset_id]

    image_size = spec["image_size"]
    crop_fraction = 0.875
    resize_size = int(round(image_size / crop_fraction))

    preprocessing_pipeline = [
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size)
    ]
    
    _augmentation_pipeline = []
    for name, args, kwargs in augmentation_pipeline:
        _class = getattr(transforms, name, None)
        augmentation_fn = _class(*args, **kwargs)
        _augmentation_pipeline.append(augmentation_fn)

    normalization_pipeline = [
        transforms.ToTensor(),
        transforms.Normalize(mean=spec["mean"], std=spec["std"]),
    ]

    train_transform = transforms.Compose(
        preprocessing_pipeline + _augmentation_pipeline + normalization_pipeline
    )
    test_transform = transforms.Compose(
        preprocessing_pipeline + normalization_pipeline
    )

    return train_transform, test_transform


class BaseDataModule(pl.LightningDataModule):
    DatasetClass = None  # To be set by subclasses

    def __init__(
        self, 
        batch_size, 
        train_size, 
        seed, 
        train_shuffle=True,
        train_pin_memory=True,
        train_n_workers=2,
        val_shuffle=False,
        val_pin_memory=True,
        val_n_workers=2,
        test_shuffle=False,
        test_pin_memory=False,
        test_n_workers=2,
        augmentation_pipeline=[],
        pretrained_dataset_id=None
    ):
        super().__init__()
        assert self.DatasetClass is not None, "DatasetClass must be set in subclass"
        self.dataset_id = self.DatasetClass.__name__.lower().replace("dataset", "")
        self.download_path = f"./temp/{self.dataset_id}"
        self.batch_size = batch_size
        self.train_size = train_size
        self.seed = seed
        
        # Split-specific args (default to base if None)
        self.train_shuffle = train_shuffle
        self.train_pin_memory = train_pin_memory
        self.train_n_workers = train_n_workers
        # Set val/test shuffle default to False if not provided
        self.val_shuffle = False if val_shuffle is None else val_shuffle
        self.val_pin_memory = val_pin_memory
        self.val_n_workers = val_n_workers
        self.test_shuffle = False if test_shuffle is None else test_shuffle
        self.test_pin_memory = test_pin_memory
        self.test_n_workers = test_n_workers

        self.augmentation_pipeline = augmentation_pipeline
        self.pretrained_dataset_id = pretrained_dataset_id
        self.class_names = None

    def prepare_data(self):
        self.DatasetClass(root=self.download_path, train=True, download=True)
        self.DatasetClass(root=self.download_path, train=False, download=True)

    def setup(self, stage=None):
        dataset_id = self.pretrained_dataset_id if self.pretrained_dataset_id else self.dataset_id
        self.train_transform, self.test_transform = build_dataset_transforms(dataset_id, self.augmentation_pipeline)

        if stage is None or stage == "fit":
            full = self.DatasetClass(root=self.download_path, train=True, transform=self.train_transform)
            self.class_names = list(full.classes)
            total = len(full)
            train_size = int(total * self.train_size)
            val_size = total - train_size
            self.train_set, self.val_set = torch.utils.data.random_split(
                full, [train_size, val_size], generator=torch.Generator().manual_seed(self.seed)
            )
            self.val_set.dataset.transform = self.test_transform

        if stage is None or stage == "test":
            self.test_set = self.DatasetClass(root=self.download_path, train=False, transform=self.test_transform)

    def _get_split_arg(self, split, arg_name):
        base = getattr(self, f"base_{arg_name}")
        split_val = getattr(self, f"{split}_{arg_name}")
        return base if split_val is None else split_val

    def train_dataloader(self, **kwargs):
        return DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            shuffle=self._get_split_arg("train", "shuffle"), 
            num_workers=self._get_split_arg("train", "n_workers"),
            pin_memory=self._get_split_arg("train", "pin_memory"),
            **kwargs
        )

    def val_dataloader(self, **kwargs):
        return DataLoader(
            self.val_set, 
            batch_size=self.batch_size, 
            shuffle=self._get_split_arg("val", "shuffle"), 
            num_workers=self._get_split_arg("val", "n_workers"),
            pin_memory=self._get_split_arg("val", "pin_memory"),
            **kwargs
        )

    def test_dataloader(self, **kwargs):
        return DataLoader(
            self.test_set, 
            batch_size=self.batch_size, 
            shuffle=self._get_split_arg("test", "shuffle"), 
            num_workers=self._get_split_arg("test", "n_workers"),
            pin_memory=self._get_split_arg("test", "pin_memory"),
            **kwargs
        )

    def _get_split_dataset(self, split):
        if split == 'train':
            return self.train_set
        elif split == 'val':
            return self.val_set
        elif split == 'test':
            return self.test_set
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

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
        for idx in subset_indices:
            label = targets[idx]
            if len(class_to_indices[label]) < n_samples_per_class:
                class_to_indices[label].append(idx)
            if all(len(v) >= n_samples_per_class for v in class_to_indices.values()):
                break

        final_indices = [i for indices in class_to_indices.values() for i in indices]
        subset = torch.utils.data.Subset(base_dataset, final_indices)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, **kwargs)


class MNISTDataModule(BaseDataModule):
    DatasetClass = MNIST

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_id = "mnist"
        self.download_path = f"./temp/{self.dataset_id}"


class CIFAR10DataModule(BaseDataModule):
    DatasetClass = CIFAR10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_id = "cifar10"
        self.download_path = f"./temp/{self.dataset_id}"


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

