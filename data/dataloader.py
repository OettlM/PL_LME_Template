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



class SimpleDataloader(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, location, n_workers, **kwargs): # simply add new parameters by name here and in the config file
        super().__init__()
        
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._location = location
        self._n_workers = n_workers

    
    def prepare_data(self):
        # This method is called once to prepare the dataset

        if self._location == "local":
            # we are on a local pc, no need to copy and extract images
            pass
        elif self._location == "cluster":
            # copy zipped images to local node for faster access during training
            node_dir = os.path.join('/scratch', os.environ['SLURM_JOB_ID'])
            Path(node_dir).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self._data_dir + "/example_img.zip", node_dir + "/example_img.zip")

            # unpack zip
            shutil.unpack_archive(node_dir + "/example_img.zip", node_dir + "/example_img")

            # delete zip file (since no longer needed)
            os.remove(node_dir + "/example_img.zip")


    def setup(self, stage):
        # This method is calles by every process when DDP multi-GPU training is used

        if self._location == "local":
            data_folder = self._data_dir + "/example_img"
        elif self._location == "cluster":
            node_dir = os.path.join('/scratch', os.environ['SLURM_JOB_ID'])
            data_folder = node_dir + "/example_img"

        all_data_list = listdir(data_folder)

        split_point = int(len(all_data_list)*0.75)
        train_data_list = all_data_list[:split_point]
        val_data_list = all_data_list[split_point:]

        # setup transforms
        trans_train = transforms.Compose([A.HorizontalFlip(), A.VerticalFlip(), A.ToFloat(max_value=255.0), ToTensorV2()])
        trans_eval = transforms.Compose([A.ToFloat(max_value=255.0), ToTensorV2()])

        # create datasets, in this case map-style datasets
        self._ds_train = MapStyleDataset(train_data_list, data_folder, trans_train)
        self._ds_eval = MapStyleDataset(val_data_list, data_folder, trans_eval)

    def train_dataloader(self):
        return DataLoader(self._ds_train, batch_size=self._batch_size, num_workers=8, pin_memory=True, prefetch_factor=8, shuffle=True)
   
    def val_dataloader(self):
        return DataLoader(self._ds_eval, batch_size=self._batch_size, num_workers=8, pin_memory=True, prefetch_factor=8, shuffle=True)