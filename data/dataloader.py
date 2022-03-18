from os import listdir
from torchvision import transforms
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import matplotlib.image as mpimg

from data.dataset import MapStyleDataset



class SimpleDataloader(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, **kwargs): #TODO add arguments included in parser
        super().__init__()
        
        self._data_dir = data_dir
        self._batch_size = batch_size

        self._train_f = None
        self._eval_f = None

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataloader")
        parser.add_argument("--batch_size", type=int, default=8)
        # TODO add dataloader specific command line arguments

        return parent_parser
    
    def prepare_data(self):
        # Use this method to do things that might write to disk or that need to be done only from a single process in distributed settings

        #TODO if not done in sbatch file, copy data from /cluster directory to executing node
        #from distutils.dir_util import copy_tree
        #copy_tree(from_directory, to_directory)

        # get list of all files/samples, ... 
        self._train_f = listdir(self._data_dir + "/train")
        self._eval_f = listdir(self._data_dir + "/eval")
        
        # if our data is small, we might wanna load it once here and keep it in RAM memory
        self._train_img = []
        for file in self._train_f:
            self._train_img.append(mpimg.imread(file))

        self._eval_img = []
        for file in self._eval_f:
            self._eval_img.append(mpimg.imread(file))

    def setup(self, stage):
        # stuff that is performed on every gpu, e.g.: count number of classes, build vocabulary, perform train/val/test splits

        # setup transforms
        trans_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ToTensor()])
        trans_eval = transforms.Compose([transforms.ToTensor()])

        # create datasets, in this case map-style datasets
        self._ds_train = MapStyleDataset(self._train_img, trans_train)
        self._ds_eval = MapStyleDataset(self._eval_img, trans_eval)

    def train_dataloader(self):
        return DataLoader(self._ds_train, batch_size=self._batch_size, num_workers=8, pin_memory=True, prefetch_factor=8, shuffle=True) #TODO adjust parameters if nessesary
   
    def val_dataloader(self):
        return DataLoader(self._ds_eval, batch_size=self._batch_size, num_workers=8, pin_memory=True, prefetch_factor=8, shuffle=True) #TODO adjust parameters if nessesary