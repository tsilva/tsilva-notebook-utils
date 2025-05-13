import time
from typing import Union
import pytorch_lightning as pl

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

