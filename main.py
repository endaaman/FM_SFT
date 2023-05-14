import os
import re
from glob import glob
import base64

import numpy as np
import torch
from torch import nn
from torch import optim
# import torch_optimizer as optim2
from torchvision.utils import make_grid
from sklearn import metrics as skmetrics
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pydantic import Field

from endaaman.ml import BaseDLArgs, BaseMLCLI, BaseTrainer, BaseTrainerConfig
from endaaman.metrics import MultiAccuracy, AccuracyByChannel, BaseMetrics
from endaaman.functional import multi_accuracy

from models import TimmModel
from datasets import ImageDataset, LABELS, LABEL_TO_NUM


class TrainerConfig(BaseTrainerConfig):
    revision: int = 1
    model_name:str
    crop_size: int
    input_size: int


class Trainer(BaseTrainer):
    def prepare(self):
        self.criterion = nn.BCELoss()
        model = TimmModel(name=self.config.model_name, num_classes=1)
        return model

    def eval(self, inputs, gts):
        preds = self.model(inputs.to(self.device), activate=True)
        loss = self.criterion(preds, gts.to(self.device))
        return loss, preds.detach().cpu()

    def visualize_roc(self, ax, train_preds, train_gts, val_preds, val_gts):
        for t, preds, gts in (('train', train_preds, train_gts), ('val', val_preds, val_gts)):
            fpr, tpr, thresholds = skmetrics.roc_curve(gts, preds)
            auc = skmetrics.auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{t} AUC:{auc:.3f}')
        ax.set_title('ROC')
        ax.set_ylabel('Sensitivity')
        ax.set_xlabel('1 - Specificity')
        ax.legend(loc='lower right')

    def metrics_auc_recall_spec(self, preds, gts):
        if len(preds) < 30:
            return None
        preds = preds.detach().cpu().numpy()
        gts = gts.detach().cpu().numpy()
        fpr, tpr, __thresholds = skmetrics.roc_curve(gts, preds)
        auc = skmetrics.auc(fpr, tpr)
        youden_index = np.argmax(tpr - fpr)
        return auc, tpr[youden_index], -fpr[youden_index]+1


class CLI(BaseMLCLI):
    class CommonArgs(BaseDLArgs):
        pass

    class TrainArgs(CommonArgs):
        lr: float = 0.0001
        batch_size: int = 8
        num_workers: int = 4
        epoch: int = 10
        model_name: str = Field('tf_efficientnetv2_b0', cli=('--model', '-m'))
        suffix: str = ''
        crop_size: int = Field(512, cli=('--crop-size', '-c'))
        input_size: int = Field(512, cli=('--input-size', '-i'))
        size: int = Field(-1, cli=('--size', '-s'))
        overwrite: bool = Field(False, cli=('--overwrite', '-o'))

    def run_train(self, a:TrainArgs):
        config = TrainerConfig(
            batch_size=a.batch_size,
            num_workers=a.num_workers,
            lr=a.lr,
            model_name=a.model_name,
            crop_size=a.size if a.size > 0 else a.crop_size,
            input_size=a.size if a.size > 0 else a.input_size,
        )

        dss = [
            ImageDataset(
                target=t,
                aug_mode='same',
                crop_size=config.crop_size,
                input_size=config.input_size,
                seed=a.seed,
            ) for t in ('train', 'test')
        ]

        out_dir = f'out/models/{config.model_name}'
        if a.suffix:
            out_dir += f'_{a.suffix}'

        trainer = Trainer(
            config=config,
            out_dir=out_dir,
            train_dataset=dss[0],
            val_dataset=dss[1],
            use_gpu=not a.cpu,
            overwrite=a.overwrite,
        )

        trainer.start(a.epoch)



if __name__ == '__main__':
    cli = CLI()
    cli.run()

