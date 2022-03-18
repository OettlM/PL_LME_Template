from argparse import ArgumentParser

from data.dataloader import SimpleDataloader
from modules.simple import SimpleModule

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def main():

    parser = ArgumentParser()
    # flag to indicate whether code is executed on pc or cluster
    parser.add_argument('-cluster', action='store_true')

    # add class specific arguments
    parser = SimpleDataloader.add_model_specific_args(parser)
    parser = SimpleModule.add_model_specific_args(parser)

    # parse arguments
    args = parser.parse_args()


    if args.cluster:
        # some cluster specific code
        # e.g. hard coded path, ...
        
        gpus=2
        strategy = 'ddp'

        data_dir = ""
        result_dir = ""
    else:
        # some pc specific code
        # e.g. hard coded path, ...
        
        gpus = 1
        strategy = 'none'

        data_dir = ""
        result_dir = ""


    dict_args = vars(args)

    data_loader = SimpleDataloader(data_dir, **dict_args)
    module = SimpleModule(**dict_args)

    model_checkpoint = ModelCheckpoint(dirpath=result_dir + "/checkpoints", filename="SimpleTest", verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [model_checkpoint, lr_monitor] # more callbacks can be added

    # tensorflow logger as an example
    tb_logger = pl_loggers.TensorBoardLogger(result_dir + "/tb_logs", name="SimpleTest")
    #tb_logger = pl_loggers.WandbLogger(...) #TODO some logging routines might have to be adjusted

    trainer = pl.Trainer(max_epochs=100, callbacks=callbacks, logger=tb_logger, progress_bar_refresh_rate=50, gpus=gpus, strategy=strategy)
    trainer.fit(module, data_loader)


if __name__ == "__main__":
    main()