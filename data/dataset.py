import torch
import math



class MapStyleDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, transforms=None):
        self._img_list = img_list
        self._transforms = transforms

    def __len__(self):
        return len(self._img_list)

    def __getitem__(self, idx):
        img = self._img_list[idx]

        # if imgs are not pre-loaded, load them here

        if self._transforms is not None:
            img = self._transforms(img)

        return img



class IterStyleDataset(torch.utils.data.IterableDataset):
    def __init__(self, img_list, transforms=None):
        self._img_list = img_list
        self._transforms = transforms
    
    def __iter__(self):
        # create iterator function
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # we only have 1 thread
            local_imgs = self.imgs
        else:
            # we have multiple threads
            # split dataset between threads
            per_worker = int(math.ceil(len(self.imgs) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = 0 + worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.imgs))

            local_imgs = self.imgs[iter_start:iter_end]

        return iter(self.image_yielder(local_imgs))

    def image_yielder(self, imgs):
        for img in imgs:
            # if imgs are not pre-loaded, load them here

            if self._transforms is not None:
                img = self._transforms(img)

            yield img

