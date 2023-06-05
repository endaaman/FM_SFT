import os
from glob import glob
from typing import NamedTuple
from itertools import groupby

from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Image as ImageType
from sklearn.model_selection import KFold, StratifiedKFold
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from pydantic import BaseModel, Field
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from endaaman import grid_split
from endaaman.ml import BaseMLCLI, get_global_seed


J = os.path.join
Image.MAX_IMAGE_PIXELS = 8_000_000_000

BASE_DIR = 'datasets/FM_vs_SFT'
LABELS = ['FM', 'SFT']
LABEL_TO_NUM = {k:i for i, k in enumerate(LABELS) }

class Item(NamedTuple):
    path: str
    patient: str
    label: str
    image: ImageType
    index: int
    test: bool



def aug_train(crop_size, input_size):
    return [
        A.RandomCrop(width=crop_size, height=crop_size),
        A.Resize(width=input_size, height=input_size),
        A.RandomRotate90(p=1),
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=5, p=0.5),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ]

def aug_test(crop_size, input_size):
    return [
        A.CenterCrop(width=crop_size, height=crop_size),
        A.Resize(width=input_size, height=input_size),
    ]


class ImageDataset(Dataset):
    def __init__(self,
                 target,
                 crop_size=512,
                 input_size=512,
                 fold=1,
                 num_folds=5,
                 aug_mode='same',
                 normalize=True,
                 seed=get_global_seed(),
                 autoload=True,
                 ):
        self.target = target
        self.input_size = input_size
        self.fold = fold
        self.num_folds = num_folds
        self.seed = seed
        self.autoload = autoload

        self.aug_mode = aug_mode
        self.normalize = normalize

        augs = {}
        augs['train'] = aug_train(crop_size, input_size)
        augs['test'] = aug_test(crop_size, input_size)
        augs['all'] = augs['test']

        # select aug
        if aug_mode == 'same':
            aug = augs[target]
        elif aug_mode == 'none':
            aug = []
        else:
            aug = augs[aug_mode]

        if normalize:
            aug += [A.Normalize(mean=0.5, std=0.5)]
        aug += [ToTensorV2()]

        self.albu = A.Compose(aug)

        df = self.load_df()
        if fold < 0:
            # split by filenames
            self.df = self.split_by_files(df)
        else:
            # split by patient folds
            folds = self.split_folds(num_folds)
            fold = folds[fold]
            self.df = self.split_by_fold(df, fold)
        print(self.df)
        self.items = self.load_data() if autoload else []

    def split_folds(self, num_folds):
        patients = {}
        for label in LABELS:
            patients[label] = set()
            for path in sorted(glob(os.path.join(BASE_DIR, label, '*.jpg'))):
                name = os.path.splitext(os.path.basename(path))[0]
                patient = name.rsplit('_', 1)[0]
                patients[label].add(patient)

        kf = KFold(n_splits=num_folds, shuffle=True, random_state=self.seed)

        folds = [
            {'train': [], 'test': []}
            for _ in range(num_folds)
        ]
        for label, data_by_label in patients.items():
            pp = list(data_by_label)
            for fold, (train_idx, test_idx) in enumerate(kf.split(pp)):
                folds[fold]['train'] += [pp[i] for i in train_idx]
                folds[fold]['test'] += [pp[i] for i in test_idx]

        return folds

    def split_by_fold(self, df, fold):
        for t in ('train', 'test'):
            df.loc[df['patient'].isin(fold[t]), 'test'] = t == 'test'
        return df

    def load_df(self):
        data = []
        for label in LABELS:
            for path in sorted(glob(os.path.join(BASE_DIR, label, '*.jpg'))):
                name = os.path.splitext(os.path.basename(path))[0]
                patient = name.rsplit('_', 1)[0]
                data.append({
                    'path': path,
                    'patient': patient,
                    'label': label,
                    # 'test': False,
                })
        df_all = pd.DataFrame(data)
        assert len(df_all) > 0, 'NO IMAGES FOUND'
        return df_all

    def split_by_files(self, df):
        df['test'] = False
        df_train, df_test = train_test_split(df, test_size=1/self.num_folds, stratify=df['label'], random_state=self.seed)
        df_test['test'] = True

        if self.target == 'train':
            df = df_train
        elif self.target == 'test':
            df = df_test
        elif self.target == 'all':
            df = pd.concat([df_train, df_test])
        else:
            raise ValueError(f'invalid target: {self.target}')
        return df

    def load_data(self):
        labels = ['FM', 'SFT']
        items = []
        for i, row in tqdm(self.df.iterrows(), leave=False, total=len(self.df)):
            image = Image.open(row['path'])
            ii = grid_split(image, self.input_size, overwrap=False, flattern=True)
            for idx, i in enumerate(ii):
                item = Item(
                    path=row['path'],
                    patient=row['patient'],
                    label=row['label'],
                    image=i,
                    index=idx,
                    test=row['test'],
                )
                items.append(item)
        print(f'[{self.target}] {len(self.df)} images loaded.')
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        x = self.albu(image=np.array(item.image))['image']
        y = torch.tensor(LABEL_TO_NUM[item.label])[None].float()
        return x, y



class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    class SamplesArgs(CommonArgs):
        dest: str = 'out/samples/'
        size: int = 512

    def run_samples(self, a:SamplesArgs):
        ds = ImageDataset(target='all', crop_size=a.size, input_size=a.size)

        def key_func(k):
            return k.name
        groups = groupby(ds.items, key_func)

        font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-M.ttf', 24)

        for name, ii in groups:
            ii = list(ii)
            idxs = np.random.choice(len(ii), size=5, replace=False)
            order = 0
            for idx in idxs:
                item = ii[idx]
                t = 'test' if item.test else 'train'
                d = J(a.dest, t)
                os.makedirs(d, exist_ok=True)
                i = item.image.crop((0 ,0, 512, 512))

                draw = ImageDraw.Draw(i)
                text = f'{item.label} {name}'
                box = draw.textbbox((0, 0), text, font=font)
                draw.rectangle(box, fill='black')
                draw.text((0, 0), text, font=font, fill='white')

                i.save(J(d, f'{item.label}_{name}_{order}.png'))
                order += 1



    class DfArgs(CommonArgs):
        size: int = 512

    def run_df(self, a:SamplesArgs):
        ds = ImageDataset(target='all', crop_size=a.size, input_size=a.size, autoload=False, fold=2)
        # ds.df.to_excel('out/df.xlsx')
        self.ds = ds


if __name__ == '__main__':
    cli = CLI()
    cli.run()
