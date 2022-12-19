import hydra
import pytorch_lightning as pl

from omegaconf import DictConfig
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar

from data.dataloader import SimpleDataloader
from modules.simple import SimpleModule


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig):
    # create flat dict with all parameters
    flat_dict = dict(cfg)
    flat_dict.update(dict(cfg.location))

    # create data loader and module
    data_loader = SimpleDataloader(**flat_dict)
    module = SimpleModule(**flat_dict)

    # create callbacks
    model_checkpoint = ModelCheckpoint(dirpath=flat_dict["result_dir"] + "/checkpoints", filename="SimpleTest", verbose=True, monitor=flat_dict["metric"], mode=flat_dict["metric_target"])
    lr_monitor = LearningRateMonitor(logging_interval='step')
    progress_bar = TQDMProgressBar(refresh_rate=50)
    callbacks = [model_checkpoint, lr_monitor, progress_bar]

    # create logger
    logger = pl_loggers.WandbLogger(project="My Project Name", name="My Experiment Name")

    trainer = pl.Trainer(max_epochs=flat_dict["num_epochs"], callbacks=callbacks, logger=logger, accelerator='gpu', devices=flat_dict["n_gpus"], strategy="ddp")
    trainer.fit(module, data_loader)



if __name__ == "__main__":
    main()