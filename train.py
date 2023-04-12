import hydra
import pytorch_lightning as pl

from omegaconf import DictConfig
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar

from data.datamodule import SimpleDataModule
from modules.simple_ae import SimpleAEModule


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig):
    # create data loader and module
    data_module = SimpleDataModule(cfg)
    module = SimpleAEModule(cfg)

    # create callbacks
    model_checkpoint = ModelCheckpoint(dirpath=cfg.location.result_dir + "/checkpoints", filename="SimpleAETest", verbose=True, monitor=cfg.metric, mode=cfg.metric_target)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    progress_bar = TQDMProgressBar(refresh_rate=50)
    callbacks = [model_checkpoint, lr_monitor, progress_bar]
    
    # create logger
    logger = pl_loggers.WandbLogger(project="My Project Name", name="My Experiment Name")

    trainer = pl.Trainer(max_epochs=cfg.num_epochs, callbacks=callbacks, logger=logger, accelerator='gpu', devices=cfg.location.n_gpus, strategy="ddp")
    trainer.fit(module, data_module)

    # test model with the best metric
    trainer.test(module, data_module, ckpt_path="best")


if __name__ == "__main__":
    main()