import torch
import torchmetrics

import numpy as np
import pytorch_lightning as pl

from networks.ae import Autoencoder


class SimpleModule(pl.LightningModule):
    def __init__(self, lr, optim, scheduler, **kwargs): # simply add new parameters by name here and in the config file
        super().__init__()
        
        self._lr = lr
        self._optim = optim
        self._scheduler = scheduler

        # save all named parameters
        self.save_hyperparameters()

        # instantiate the model
        self._model = Autoencoder()

        # create loss and metric functions
        self.loss_f = torch.nn.MSELoss()
        self.metric_f = torch.nn.L1Loss()

        # create aggregators (required for ddp training)
        self._train_loss_agg = torchmetrics.MeanMetric()
        self._val_loss_agg = torchmetrics.MeanMetric()

        self._val_l1_agg = torchmetrics.MeanMetric()
        

    def forward(self, x):
        # the method used for inference
        output = self._model(x)
        return output

    def training_step(self, batch, batch_idx):
        output = self._model(batch)
        train_loss = self.loss_f(output, batch)
        
        self._train_loss_agg.update(train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        output = self._model(batch)
        val_loss = self.loss_f(output, batch)
        
        self._val_loss_agg.update(val_loss)

        val_metric = self.metric_f(output, batch)
        self._val_l1_agg.update(val_metric)
        return val_loss


    def training_epoch_end(self, outputs):
        # required if values returned in the training_steps have to be processed in a specific way
        self.log(self._train_loss_agg.compute(), "Training Loss")
        self._train_loss_agg.reset()


    def validation_epoch_end(self, outputs):
        # required if values returned in the validation_step have to be processed in a specific way
        self.log(self._val_loss_agg.compute(), "Val Loss")
        self._val_loss_agg.reset()

        self.log(self._val_l1_agg.compute(), "Val F1 Metric")
        self._val_l1_agg.reset()

        # we could also log example images to wandb here
        # img = some_example_img
        # out_img = self._model(img)
        # self.logger.log_image("Example Image", images=[out_img])


    def configure_optimizers(self):
        # configure optimizers based on command line parameters
        if self._optim == "sdg":
            optimizer = torch.optim.SGD(self._model.parameters(), lr=self._lr, momentum=0.95)
        elif self._optim == "adam":
            optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        # TODO add more options if required

        # configure shedulers based on command line parameters
        if self._scheduler == "none":
            return optimizer
        elif self._scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.1)
        # TODO add more options if required

        return [optimizer], [scheduler]