import os
import shutil

import albumentations as A
import pytorch_lightning as pl

from os import listdir
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from albumentations.pytorch.transforms import ToTensorV2

from data.dataset import MapStyleDataset



class SimpleDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        
        self._data_dir = cfg.location.data_dir
        self._batch_size = cfg.location.batch_size
        self._location = cfg.location.name
        self._n_workers = cfg.location.n_workers

        self._folder_name = "example_img" # TODO adapt to your needs

    
    def prepare_data(self):
        # This method is called once to prepare the dataset

        if self._location == "local":
            # we are on a local pc, no need to copy and extract images
            pass
        elif self._location == "cluster":
            # copy zipped images to local node for faster access during training
            node_dir = os.path.join('/scratch', os.environ['SLURM_JOB_ID'])
            Path(node_dir).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self._data_dir + f"/{self._folder_name}.zip", node_dir + f"/{self._folder_name}.zip")

            # unpack zip
            shutil.unpack_archive(node_dir + f"/{self._folder_name}.zip", node_dir + f"/{self._folder_name}")

            # delete zip file (since no longer needed)
            os.remove(node_dir + f"/{self._folder_name}.zip")


    def setup(self, stage):
        # This method is calles by every process when DDP multi-GPU training is used

        if self._location == "local":
            data_folder = self._data_dir + f"/{self._folder_name}"
        elif self._location == "cluster":
            node_dir = os.path.join('/scratch', os.environ['SLURM_JOB_ID'])
            data_folder = node_dir + f"/{self._folder_name}"

        # TODO write your own dataset setup code starting from here

        # Example code for a simple autoencoder
        all_data_list = listdir(data_folder)

        split_point = int(len(all_data_list)*0.75)
        train_data_list = all_data_list[:split_point]
        val_data_list = all_data_list[split_point:]

        # setup transforms
        aug_train = transforms.Compose([A.HorizontalFlip(), A.VerticalFlip(), A.ToFloat(max_value=255.0), ToTensorV2()])
        aug_unmod = transforms.Compose([A.ToFloat(max_value=255.0), ToTensorV2()])

        # create datasets
        self._ds_train = MapStyleDataset(train_data_list, data_folder, aug_train)
        self._ds_eval = MapStyleDataset(val_data_list, data_folder, aug_unmod)
        # TODO add test data
        # self._ds_test = MapStyleDataset(test_data_list, data_folder, aug_unmod)

    def train_dataloader(self):
        return DataLoader(self._ds_train, batch_size=self._batch_size, num_workers=self._n_workers, pin_memory=True, prefetch_factor=2, shuffle=True)
   
    def val_dataloader(self):
        return DataLoader(self._ds_eval, batch_size=self._batch_size, num_workers=self._n_workers, pin_memory=True, prefetch_factor=2, shuffle=False)

    # TODO activate when test dataset exists
    #def test_dataloader(self):
    #    return DataLoader(self._ds_test, batch_size=self._batch_size, num_workers=self._n_workers, pin_memory=True, prefetch_factor=2, shuffle=False)

    def get_vis_img(self):
        return self._ds_eval[0]