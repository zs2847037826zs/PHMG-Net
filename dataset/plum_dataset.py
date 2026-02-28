import os.path
from os.path import join

import numpy as np
from skimage import io
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from dataset.metadata import DataTuple


class PlumDataset(Dataset):
    def __init__(self, path, transform=None, mode='train', fold=None, label_file=None):
        """SKin Lesion"""
        self.path = path
        self.transform = transform
        self.mode = mode
        self.pd_dir = join(self.path, "persistent_diagram")
        self.data = self.get_data(fold)
        self.classes_name = ['大果', '中果', '小果', '坏果']
        self.classes = list(range(len(self.classes_name)))

    def __getitem__(self, i):
        mete_data: DataTuple = self.data[i]

        target = mete_data.target
        img = io.imread(os.path.join(self.path, "data1", mete_data.imag + ".jpg"))

        if self.transform is not None:
            img = self.transform(img)

        return img, target, mete_data.pd, mete_data.weight, mete_data.sugar, mete_data.imag

    def __len__(self):
        return len(self.data)

    def get_target(self, path):
        if path[0] == 'd':
            return np.array(0)
        elif path[0] == 'z':
            return np.array(1)
        elif path[0] == 'x':
            return np.array(2)
        elif path[0] == 'h':
            return np.array(3)

    def get_data(self, fold):
        if self.mode in ['train', 'val', 'test']:
            xlsx = f'plum_dataset_{self.mode}.xlsx'
        else:
            raise ValueError("mode 只能为train、val、test")

        fn = os.path.join(self.path, xlsx)
        pdfile = pd.read_excel(fn, dtype={"weight": float, "sugar": float})
        data_list = []
        for index, row in pdfile.iterrows():
            target = self.get_target(row['path'])
            imag = row['path']
            sugar = np.array([row['sugar']])
            weight = np.array([row['weight']])

            pd_ = np.load(os.path.join(self.path, "persistent_diagram", row['path'], "pd_combined.npy"))
            result = process_your_pd_data(pd_)
            data = DataTuple(target, imag, sugar, weight, result)
            data_list.append(data)

        return data_list


def process_your_pd_data(pd_array, top_k=100, min_persistence=None, max_persistence=None):
    if len(pd_array) == 0:
        return pd_array

    persistence = pd_array[:, 1] - pd_array[:, 0]

    mask = np.ones(len(pd_array), dtype=bool)

    if min_persistence is not None:
        mask = mask & (persistence >= min_persistence)

    if max_persistence is not None:
        mask = mask & (persistence <= max_persistence)

    filtered_data = pd_array[mask]
    filtered_persistence = persistence[mask]

    if len(filtered_data) == 0:
        return np.zeros((0, 4))

    sorted_indices = np.argsort(filtered_persistence)[::-1]

    k = min(top_k, len(filtered_data))
    top_indices = sorted_indices[:k]
    result = filtered_data[top_indices]

    return result


if __name__ == "__main__":
    input_size = 256
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.ToPILImage(),
        # transforms.Resize(re_size),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.Resize(re_size),
        transforms.Resize((input_size, input_size), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
        transforms.RandomRotation([-180, 180]),
        transforms.RandomAffine([-180, 180], translate=[0.1, 0.1],
                                scale=[0.7, 1.3]),
        # transforms.RandomCrop(input_size),
    ])
    data = PlumDataset("../data", transform=train_transform)

    dataloader = DataLoader(data, batch_size=16)

    for item in dataloader:
        print(item)
