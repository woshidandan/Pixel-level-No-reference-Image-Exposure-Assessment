import os
from torchvision import transforms
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

global_best_pic='F:/IEA/label/global_best'

IMAGE_NET_MEAN = [0., 0., 0.]
IMAGE_NET_STD = [1., 1., 1.]
normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)



class IEA_Dataset(Dataset):
    def __init__(self, path_to_csv, images_path,if_train,npy_file_path):
        self.df = pd.read_csv(path_to_csv)
        self.images_path =  images_path
        self.npy_file_path=npy_file_path
        self.best_pic_path=global_best_pic

        self.grey_transform =  transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]
)

        if if_train:
            self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop((256, 256)),
            transforms.ToTensor(),
            normalize])
        else:
            self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            normalize])


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        scores_names = [f'score{i}' for i in range(2,12)]
        y = row['score']

        image_id = row['img_id']
        ref_id=row['ref_id']
        image_path = os.path.join(self.images_path, f'{image_id}')

        npy_path=os.path.join(self.npy_file_path,f'{image_id[:-4]}.npy')
        npy_map = torch.from_numpy(np.load(npy_path))/255

        ref_path=os.path.join(self.best_pic_path,f'{ref_id}')
        ref_image=self.grey_transform(default_loader(ref_path))
        x = self.transform(default_loader(image_path))

        return x, y,npy_map,ref_image
