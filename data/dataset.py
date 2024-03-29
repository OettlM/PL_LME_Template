import torch
import numpy as np

from PIL import Image



class MapStyleDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, data_folder, transforms=None):
        self._data_list = data_list
        self._data_folder = data_folder
        self._transforms = transforms


    def __len__(self):
        return len(self._data_list)


    def __getitem__(self, idx):
        # TODO adapt to your needs
        # load whatever is required
        # for Autoencoder/ simple GAN: image
        # for Classification: image, class_label
        # for Cycle-GAN: image, origin
        # for ...: ...
        img_name = self._data_list[idx]
        img = Image.open(self._data_folder + img_name)
        img = np.array(img)

        if self._transforms is not None:
            img = self._transforms(img)
        else:
            img = torch.from_numpy(img).permute(2,0,1).to(torch.float32)

        return img

