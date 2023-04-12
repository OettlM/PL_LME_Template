import torch
import torchmetrics

import numpy as np
import pytorch_lightning as pl

from networks.ae import Autoencoder


class SimpleAEModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        
        self._lr = cfg.lr
        self._optim = cfg.optim
        self._scheduler = cfg.scheduler

        self._log_imgs = cfg.log_images

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
        self.log(self._train_loss_agg.compute(), "Train Loss")
        self._train_loss_agg.reset()


    def validation_epoch_end(self, outputs):
        # required if values returned in the validation_step have to be processed in a specific way
        self.log(self._val_loss_agg.compute(), "Val Loss")
        self._val_loss_agg.reset()

        self.log(self._val_l1_agg.compute(), "Val L1")
        self._val_l1_agg.reset()

        # we could also log example images to wandb here
        if self._log_imgs:
            with torch.no_grad():
                vis_img = self.trainer.datamodule.get_vis_img()
                vis_out = self._model(vis_img.unsqueeze(0).cuda()).cpu()[0]
                
                vis_img = vis_img.permute(1,2,0).numpy()
                vis_out = vis_out.permute(1,2,0).numpy()

                self.logger.log_image("AE example img", images=[vis_img, vis_out])


    def configure_optimizers(self):
        # configure optimizers based on command line parameters
        if self._optim == "sgd":
            optimizer = torch.optim.SGD(self._model.parameters(), lr=self._lr, momentum=0.95)
        elif self._optim == "adam":
            optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        # TODO add more options if required

        # configure shedulers based on command line parameters
        if self._scheduler == "none":
            return optimizer
        elif self._scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.1)
        # TODO add more options if required

        return [optimizer], [scheduler]