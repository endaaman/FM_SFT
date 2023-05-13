from glob import glob
from typing import NamedTuple

from PIL import Image
from PIL.Image import Image as ImageType
import torch
from torch.utils.data import Dataset
from pydantic import BaseModel, Field

from endaaman import grid_split

class Item(NamedTuple):
    path: str
    name: str
    label: str
    image: ImageType


class ImageDataset(Dataset):
    def __init__(self):
        items = self.load_data()

    def load_data(self):
        labels = ['FM', 'SFT']
        for label in labels:
            print(label)
            basedir = f'datasets/FM_vs_SFT/{label}'
            for path in sorted(glob(f'{basedir}/*.jpg')):
                i = Image.open(path)

        return []


if __name__ == '__main__':
    ds = ImageDataset()

