import os
import re
from glob import glob
import base64
import json

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
import pytorch_grad_cam as CAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from endaaman import load_images_from_dir_or_file, grid_split_by_size
from endaaman.ml import BaseDLArgs, BaseMLCLI, BaseTrainer, BaseTrainerConfig, pil_to_tensor, tensor_to_pil, overlay_heatmap
from endaaman.metrics import MultiAccuracy, AccuracyByChannel, BaseMetrics
from endaaman.functional import multi_accuracy

from models import TimmModel
from datasets import ImageDataset, LABELS, LABEL_TO_NUM


J = os.path.join

class TrainerConfig(BaseTrainerConfig):
    fold: int
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
            if t == 'train':
                youden_index = np.argmax(tpr - fpr)
                threshold = thresholds[youden_index]
        ax.set_title(f'ROC (t={threshold:.2f})')
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
        lr: float = 0.0002
        batch_size: int = Field(16, cli=('--batch-size', ))
        epoch: int = 20
        fold: int = -1
        model_name: str = Field('tf_efficientnetv2_b0', cli=('--model', '-m'))
        suffix: str = ''
        crop_size: int = Field(512, cli=('--crop-size', '-c'))
        input_size: int = Field(512, cli=('--input-size', '-i'))
        size: int = Field(-1, cli=('--size', '-s'))
        overwrite: bool = Field(False, cli=('--overwrite', '-O'))

    def run_train(self, a:TrainArgs):
        config = TrainerConfig(
            batch_size=a.batch_size,
            num_workers=a.num_workers,
            lr=a.lr,
            fold=a.fold,
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
                fold=a.fold,
                seed=a.seed,
            ) for t in ('train', 'test')
        ]

        out_dir = f'out/models/{config.model_name}_fold{a.fold}'
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


    class PredictArgs(CommonArgs):
        model_dir: str = Field(..., cli=('--dir', '-d'))
        src: str = Field(..., cli=('--src', '-s'))

    def run_predict(self, a:PredictArgs):
        checkpoint = torch.load(J(a.model_dir, 'checkpoint_best.pt'))
        with open(J(a.model_dir, 'config.json'), mode='r', encoding='utf-8') as f:
            config = json.load(f)
            config = TrainerConfig(**config)
        device = 'cuda'
        model = TimmModel(name=config.model_name, num_classes=1)
        model.load_state_dict(checkpoint.model_state)
        model.to(device)

        images, paths = load_images_from_dir_or_file(a.src, with_path=True)
        image, path = images[0], paths[0]
        name = os.path.splitext(os.path.basename(path))[0]

        gradcam = CAM.GradCAM(
            model=model,
            target_layers=[model.get_cam_layer()],
            use_cuda=device=='cuda')

        T = 0.47
        t = pil_to_tensor(image)
        t = t[:, :512, :512]
        image = tensor_to_pil(t)
        t = t[None, ...]
        p = model(t.to(device), activate=True).cpu()
        targets = [BinaryClassifierOutputTarget(p > T)]
        mask = torch.from_numpy(gradcam(input_tensor=t, targets=targets))
        heatmap, masked = overlay_heatmap(mask, pil_to_tensor(image), alpha=0.3, threshold=0.5)
        d = f'{a.model_dir}/predict'
        os.makedirs(d, exist_ok=True)
        image.save(f'{d}/{name}.png')
        tensor_to_pil(masked).save(f'{d}/{name}_masked.png')
        print(p, p > T)


if __name__ == '__main__':
    cli = CLI()
    cli.run()

