import time
import torch
from typing import Union
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
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


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size, 
        train_size, 
        seed, 
        num_workers=2, 
        train_shuffle=True,
        val_shuffle=False,
        test_shuffle=False,
        augmentation_pipeline=[],
        pretrained_dataset_id=None
    ):
        super().__init__()
        self.dataset_id = "mnist"
        self.download_path = f"./temp/{self.dataset_id}"
        self.batch_size = batch_size
        self.train_size = train_size
        self.seed = seed
        self.num_workers = num_workers
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.test_shuffle = test_shuffle
        self.n_classes = 10
        self.augmentation_pipeline = augmentation_pipeline
        self.pretrained_dataset_id = pretrained_dataset_id


    def prepare_data(self):
        MNIST(root=self.download_path, train=True, download=True)
        MNIST(root=self.download_path, train=False, download=True)

    def setup(self, stage=None):
        dataset_id = self.pretrained_dataset_id if self.pretrained_dataset_id else self.dataset_id
        self.train_transform, self.test_transform = build_dataset_transforms(dataset_id, self.augmentation_pipeline)

        full = MNIST(root=self.download_path, train=True, transform=self.train_transform)
        total = len(full)
        train_size = int(total * self.train_size)
        val_size = total - train_size
        self.train_set, self.val_set = torch.utils.data.random_split(
            full, [train_size, val_size], generator=torch.Generator().manual_seed(self.seed)
        )
        self.val_set.dataset.transform = self.test_transform
        self.test_set = MNIST(root=self.download_path, train=False, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            shuffle=self.train_shuffle, 
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, 
            batch_size=self.batch_size, 
            shuffle=self.val_shuffle, 
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, 
            batch_size=self.batch_size, 
            shuffle=self.test_shuffle, 
            num_workers=self.num_workers
        )
    

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size, 
        train_size, 
        seed, 
        num_workers=2, 
        train_shuffle=True,
        val_shuffle=False,
        test_shuffle=False,
        augmentation_pipeline=[],
        pretrained_dataset_id=None
    ):
        super().__init__()
        self.dataset_id = "cifar10"
        self.download_path = f"./temp/{self.dataset_id}"
        self.batch_size = batch_size
        self.train_size = train_size
        self.seed = seed
        self.num_workers = num_workers
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.test_shuffle = test_shuffle
        self.n_classes = 10
        self.augmentation_pipeline = augmentation_pipeline
        self.pretrained_dataset_id = pretrained_dataset_id

    def prepare_data(self):
        CIFAR10(root=self.download_path, train=True, download=True)
        CIFAR10(root=self.download_path, train=False, download=True)

    def setup(self, stage=None):
        dataset_id = self.pretrained_dataset_id if self.pretrained_dataset_id else self.dataset_id
        self.train_transform, self.test_transform = build_dataset_transforms(dataset_id, self.augmentation_pipeline)

        full = CIFAR10(root=self.download_path, train=True, transform=self.train_transform)
        total = len(full)
        train_size = int(total * self.train_size)
        val_size = total - train_size
        self.train_set, self.val_set = torch.utils.data.random_split(
            full, [train_size, val_size], generator=torch.Generator().manual_seed(self.seed)
        )
        self.val_set.dataset.transform = self.test_transform
        self.test_set = CIFAR10(root=self.download_path, train=False, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            shuffle=self.train_shuffle, 
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, 
            batch_size=self.batch_size, 
            shuffle=self.val_shuffle, 
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, 
            batch_size=self.batch_size, 
            shuffle=self.test_shuffle, 
            num_workers=self.num_workers
        )


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
        "augmentation_pipeline": config['augmentation_pipeline'],
        "pretrained_dataset_id": config['pretrained_dataset_id'],
        **kwargs
    })
    datamodule.prepare_data()
    return datamodule

