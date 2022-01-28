import argparse

import numpy as np
import torch
import torch.nn as nn
from tdm.datasets import OxfordPetDataset
from tdm.models import UNet
from tdm.metrics.segmentation import dice_score 
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train UNet on Oxford-IIIT Pet Dataset')
    parser.add_argument('--batch-size', default=8, metavar='BS', type=int)
    parser.add_argument('--data-dir',
                        default='./data/',
                        metavar='DIR',
                        help='Directory to store the dataset')
    parser.add_argument('--n-epochs',
                        default=15,
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

    return parser.parse_args()


def train_one_epoch(model,
                    opt,
                    loader,
                    loss_fn,
                    device,
                    metric=None,
                    metric_name=None):
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
            progress.set_description(
                f"Train loss: {loss:.4f}, {metric_name}: {metric_value:.4f}")
        else:
            progress.set_description(f"Train loss: {loss:.4f}")

    return np.mean(losses)


def validate(model, loader, loss_fn, device, metric=None, metric_name=None):
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
                    f"Val loss: {loss:.4f}, {metric_name}: {metric_value:.4f}")
            else:
                progress.set_description(f"Val loss: {loss:.4f}")

    return np.mean(losses)


class MaskToTensor(object):
    def __call__(self, mask):
        mask = torch.as_tensor(np.array(mask), dtype=torch.float)
        return mask


class PreprocessMask(object):
    def __call__(self, mask: torch.Tensor):
        mask[mask == 2.0] = 0.0
        mask[mask == 3.0] = 1.0
        return mask


def main():
    args = parse_args()
    image_transform = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.ToTensor()])
    target_transform = transforms.Compose([
        transforms.Resize((256, 256),
                          interpolation=transforms.InterpolationMode.NEAREST),
        MaskToTensor(),
        PreprocessMask()
    ])
    dataset = OxfordPetDataset(root=args.data_dir,
                               download=True,
                               transform=image_transform,
                               target_transform=target_transform)
    val_size = int(len(dataset) * 0.25)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, lengths=(len(dataset) - val_size, val_size))
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
    for epoch in range(args.n_epochs):
        print(f"Epoch: {epoch}/{args.n_epochs-1}")
        train_loss = train_one_epoch(model,
                                     opt,
                                     train_dataloader,
                                     loss_fn,
                                     device,
                                     metric=dice_score,
                                     metric_name='Dice score')
        print('Average train loss:', train_loss)
        val_loss = validate(model,
                            val_dataloader,
                            loss_fn,
                            device,
                            metric=dice_score,
                            metric_name='Dice score')
        print('Average validation loss:', val_loss)


if __name__ == '__main__':
    main()
