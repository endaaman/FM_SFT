import os
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
    index: int


class ImageDataset(Dataset):
    def __init__(self, image_size=512):
        self.image_size = image_size
        self.items = self.load_data()

    def load_data(self):
        labels = ['FM', 'SFT']
        items = {}
        for label in labels:
            basedir = f'datasets/FM_vs_SFT/{label}'
            for path in sorted(glob(f'{basedir}/*.jpg')):
                image = Image.open(path)
                name = os.path.splitext(os.path.basename(path))[0]
                ii = grid_split(image, self.image_size, overwrap=False, flattern=True)
                for idx, i in enumerate(ii):
                    item = Item(
                        path=path,
                        name=name,
                        label=label,
                        image=i,
                        index=idx,
                    )
                    if name in items:
                        items[name].append(item)
                    else:
                        items[name] = [item]
        return items


if __name__ == '__main__':
    ds = ImageDataset()
