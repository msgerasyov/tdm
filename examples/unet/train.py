import argparse
import random
from os.path import abspath, dirname, join

import numpy as np
import torch
import torch.nn as nn
from tdm.datasets import OxfordPetDataset
from tdm.metrics.segmentation import iou_score
from tdm.models import UNet
from tdm.transforms import segmentation as S
from tdm.utils import MetricMeter, metric_meter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train UNet on Oxford-IIIT Pet Dataset')
    parser.add_argument('--batch-size', default=16, metavar='BS', type=int)
    parser.add_argument('--data-dir',
                        default=join(dirname(abspath(__file__)), 'data/'),
                        metavar='DDIR',
                        help='Directory to store the dataset')
    parser.add_argument('--save-dir',
                        default=join(dirname(abspath(__file__)), 'saves/'),
                        metavar='SDIR',
                        help='Directory to save trained models')
    parser.add_argument('--n-epochs',
                        default=20,
                        metavar='NEP',
                        type=int,
                        help='Number of training epochs')
    parser.add_argument('--lr',
                        default=1e-3,
                        metavar='LR',
                        type=float,
                        help='Learning rate')
    parser.add_argument('--no-cuda',
                        action='store_false',
                        dest='cuda',
                        help='Disable GPU during training')
    parser.add_argument('--seed',
                        default=1,
                        metavar='SEED',
                        type=int,
                        help='Seed for random number generators')

    return parser.parse_args()


def train_one_epoch(model,
                    opt,
                    loader,
                    loss_fn,
                    device,
                    metric=None,
                    metric_meter=None):
    model.train()
    losses = []
    progress = tqdm(enumerate(loader), total=len(loader))
    for idx, batch in progress:
        imgs = batch[0].to(device)
        masks = batch[1].to(device)
        out = model(imgs).squeeze(1)
        loss = loss_fn(out, masks)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.cpu().data.numpy())
        if metric is not None:
            metric_value = metric(masks, out,
                                  from_logits=True).cpu().data.numpy()
            metric_meter.update(metric_value)
            progress.set_description(
                f"Train loss: {loss:.4f}, {metric_meter.metric_name}: {metric_value:.4f}"
            )
        else:
            progress.set_description(f"Train loss: {loss:.4f}")

    return np.mean(losses)


def validate(model, loader, loss_fn, device, metric=None, metric_meter=None):
    model.eval()
    losses = []
    progress = tqdm(enumerate(loader), total=len(loader))
    with torch.no_grad():
        for idx, batch in progress:
            imgs = batch[0].to(device)
            masks = batch[1].to(device)
            out = model(imgs).squeeze(1)
            loss = loss_fn(out, masks)
            losses.append(loss.cpu().data.numpy())
            if metric is not None:
                metric_value = metric(masks, out,
                                      from_logits=True).cpu().data.numpy()
                progress.set_description(
                    f"Val loss: {loss:.4f}, {metric_meter.metric_name}: {metric_value:.4f}"
                )
            else:
                progress.set_description(f"Val loss: {loss:.4f}")

    return np.mean(losses)


class PreprocessMask(object):
    def __call__(self, image, mask: torch.Tensor):
        mask = mask.float()
        mask[mask == 2.0] = 0.0
        mask[mask == 3.0] = 1.0
        return image, mask


def main():
    args = parse_args()
    fix_seed(args.seed)
    train_transform = S.Compose([
        S.Resize((256, 256)),
        S.RandomAffine(15, translate=(0.2, 0.2), scale=(0.75, 1.25)),
        S.ColorJitter(0.3, 0.3),
        S.ToTensor(),
        S.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        PreprocessMask(),
    ])
    val_transform = S.Compose([
        S.Resize((256, 256)),
        S.ToTensor(),
        S.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        PreprocessMask(),
    ])
    dataset = OxfordPetDataset(root=args.data_dir,
                               download=True,
                               transform=train_transform)
    val_size = int(len(dataset) * 0.25)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, lengths=(len(dataset) - val_size, val_size))
    val_dataset.transform = val_transform
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=0)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size * 2,
                                shuffle=False,
                                drop_last=False,
                                num_workers=0)
    model = UNet(in_channels=3, out_channels=1)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    model.to(device)
    metric_meter = MetricMeter('IoU score')
    for epoch in range(args.n_epochs):
        print(f"Epoch: {epoch+1}/{args.n_epochs}")
        train_loss = train_one_epoch(model,
                                     opt,
                                     train_dataloader,
                                     loss_fn,
                                     device,
                                     metric=iou_score,
                                     metric_meter=metric_meter)
        print('Average train loss:', train_loss)
        print(metric_meter)
        metric_meter.reset()
        val_loss = validate(model,
                            val_dataloader,
                            loss_fn,
                            device,
                            metric=iou_score,
                            metric_meter=metric_meter)
        print('Average validation loss:', val_loss)
        print(metric_meter)
        metric_meter.reset()


if __name__ == '__main__':
    main()
