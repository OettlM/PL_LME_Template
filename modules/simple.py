import torch

import numpy as np
import pytorch_lightning as pl

from networks.ae import Autoencoder


class SimpleModule(pl.LightningModule):
    def __init__(self, lr, optim, scheduler, **kwargs): #TODO add arguments included in parser
        super().__init__()
        
        self._lr = lr
        self._optim = optim
        self._scheduler = scheduler

        # save all parameters
        self.save_hyperparameters()

        # instantiate the model
        self._model = Autoencoder()

        # create loss and metric functions
        self.loss_f = torch.nn.MSELoss()
        self.metric_f = torch.nn.L1Loss()
        

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SimpleModule")
        parser.add_argument("--lr", type=float, default=0.01)
        parser.add_argument("--optim", type=str, default="sdg")
        parser.add_argument("--scheduler", type=str, default="none")
        # TODO add module specific command line arguments
        
        return parent_parser

    def forward(self, x):
        # the method used for inference
        output = self._model(x)
        return output

    def training_step(self, batch, batch_idx):
        model_in = batch[0].permute(0, 3, 1, 2)
        label = model_in

        model_out = self._model(model_in)

        train_loss = self.loss_f(model_out, label)
        self.log("train_loss", train_loss, on_epoch=True, on_step=False)

        train_metric = self.metric_f(model_out, label)
        self.log("train_metric", train_metric, on_epoch=True, on_step=False)

        #return {"loss": train_loss, "metrics": train_metric}
        return train_loss

    def validation_step(self, batch, batch_idx):
        # gradients are automatically deactivated here
        model_in = batch[0].permute(0, 3, 1, 2)
        label = model_in

        model_out = self._model(model_in)

        eval_loss = self.loss_f(model_out, label)
        self.log("eval_loss", eval_loss, on_epoch=True, on_step=False)

        eval_metric = self.metric_f(model_out, label)
        self.log("eval_metric", eval_metric, on_epoch=True, on_step=False)

        return {"loss": eval_loss, "metrics": eval_metric}
        #return eval_loss


    #def training_epoch_end(self, outputs):
        # required if values returned in the training_steps have to be processed in a specific way
        # e.g. if the mean of a metric is not sufficient, but the 5th percentile is required
        #pass

    def validation_epoch_end(self, outputs):
        # required if values returned in the validation_step have to be processed in a specific way
        # e.g. if the mean of a metric is not sufficient, but the 5th percentile is required
        metrics = []
        [metrics.append(s['metrics']) for s in outputs]
        metrics = np.array(metrics)

        if len(metrics) > 0:
            percs = np.percentile(metrics, [5])
            self.log("eval_metric_percentile_5", percs[0])

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