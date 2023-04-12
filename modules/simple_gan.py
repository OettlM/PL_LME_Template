import torch
import torchmetrics

import pytorch_lightning as pl


# example code, mainly taken from:
# https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/basic-gan.html

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
        self._hidden_dim = cfg.hidden_dim # dimension of noise vector
        self._generator = ... # TODO add your generator model here
        self._discriminator = ... # TODO add your discriminator model here

        # create loss and metric functions
        self.loss_f = torch.nn.BCELoss()

        # create aggregators (required for ddp training)
        self._train_loss_gen_agg = torchmetrics.MeanMetric()
        self._train_loss_disc_agg = torchmetrics.MeanMetric()
        self._val_loss_gen_agg = torchmetrics.MeanMetric()
        self._val_loss_disc_agg = torchmetrics.MeanMetric()
        

    def forward(self, z):
        return self._generator(z)
    

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs = batch

        # ground truth (tensors of ones and zeros) same shape as images
        valid = torch.ones(real_imgs.size(0), 1)
        fake = torch.zeros(real_imgs.size(0), 1)

        # generator training
        if optimizer_idx == 0:
            z = torch.randn(real_imgs.shape[0], self._hidden_dim)

            self.gen_imgs = self._generator(z)

            gen_loss = self.loss_f(self._discriminator(self.gen_imgs), valid)
            self._train_loss_gen_agg.update(gen_loss)
            return gen_loss
        
        # discriminator training
        if optimizer_idx == 1:
            real_loss = self.loss_f(real_imgs, valid)
            fake_loss = self.loss_f(self.gen_imgs.detach(), fake)
            disc_loss = (real_loss + fake_loss) / 2.0

            self._train_loss_disc_agg.update(disc_loss)
            return disc_loss

    def validation_step(self, batch, batch_idx, optimizer_idx):
        real_imgs = batch

        # ground truth (tensors of ones and zeros) same shape as images
        valid = torch.ones(real_imgs.size(0), 1)
        fake = torch.zeros(real_imgs.size(0), 1)

        # generator training
        if optimizer_idx == 0:
            z = torch.randn(real_imgs.shape[0], self._hidden_dim)

            self.gen_imgs = self._generator(z)

            gen_loss = self.loss_f(self._discriminator(self.gen_imgs), valid)
            self._train_loss_gen_agg.update(gen_loss)
            return gen_loss
        
        # discriminator training
        if optimizer_idx == 1:
            real_loss = self.loss_f(real_imgs, valid)
            fake_loss = self.loss_f(self.gen_imgs.detach(), fake)
            disc_loss = (real_loss + fake_loss) / 2.0

            self._train_loss_disc_agg.update(disc_loss)
            return disc_loss


    def training_epoch_end(self, outputs):
        # required if values returned in the training_steps have to be processed in a specific way
        self.log(self._train_loss_gen_agg.compute(), "Train Generator Loss")
        self._train_loss_gen_agg.reset()

        self.log(self._train_loss_disc_agg.compute(), "Train Discriminator Loss")
        self._train_loss_disc_agg.reset()


    def validation_epoch_end(self, outputs):
        # required if values returned in the validation_step have to be processed in a specific way
        self.log(self._val_loss_gen_agg.compute(), "Val Generator Loss")
        self._val_loss_gen_agg.reset()

        self.log(self._val_loss_disc_agg.compute(), "Val Discriminator Loss")
        self._val_loss_disc_agg.reset()

        # we could also log example images to wandb here
        if self._log_imgs:
            with torch.no_grad():
                z = ... # TODO some random vector as torch
                vis_out = self._generator(z.unsqueeze(0).cuda()).cpu()[0]
            
                vis_out = vis_out.permute(1,2,0).numpy()

                self.logger.log_image("GAN example img", images=[vis_out])


    def configure_optimizers(self):
        # configure optimizers based on command line parameters
        if self._optim == "sgd":
            optimizer_g = torch.optim.SGD(self._generator.parameters(), lr=self._lr, momentum=0.95)
            optimizer_d = torch.optim.SGD(self._discriminator.parameters(), lr=self._lr, momentum=0.95)
        elif self._optim == "adam":
            optimizer_g = torch.optim.Adam(self._generator.parameters(), lr=self._lr)
            optimizer_d = torch.optim.Adam(self._discriminator.parameters(), lr=self._lr)
        # TODO add more options if required

        # configure shedulers based on command line parameters
        if self._scheduler == "none":
            return [optimizer_g, optimizer_d]
        elif self._scheduler == "step":
            scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, 20, 0.1)
            scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, 20, 0.1)
        # TODO add more options if required

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]