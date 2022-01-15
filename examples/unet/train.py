import argparse

import numpy as np
import torch
import torch.nn as nn
from tdm.models import UNet
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train UNet on Pascal VOC dataset')
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


def train_one_epoch(model, opt, loader, loss_fn, device, metrics=[]):
    losses = []
    progress = tqdm(enumerate(loader), total=len(loader))
    for idx, batch in progress:
        imgs = batch[0].to(device)
        masks = batch[1].long().squeeze().to(device)
        out = model(imgs)
        loss = loss_fn(out, masks)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.cpu().data.numpy())
        progress.set_description(f"Loss: {loss}")

    return np.mean(losses)


def validate(model, loader, loss_fn, device, metrics=[]):
    losses = []
    progress = tqdm(enumerate(loader), total=len(loader))
    with torch.no_grad():
        for idx, batch in progress:
            imgs = batch[0].to(device)
            masks = batch[1].long().squeeze().to(device)
            out = model(imgs)
            loss = loss_fn(out, masks)
            losses.append(loss.cpu().data.numpy())
            progress.set_description(f"Loss: {loss}")

    return np.mean(losses)


def main():
    args = parse_args()
    train_image_transform = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.Pad(92, padding_mode='reflect'),
        transforms.ToTensor()
    ])
    train_target_transform = transforms.Compose(
        [transforms.Resize((244, 244)),
         transforms.ToTensor()])
    val_image_transform = train_image_transform
    val_target_trasfrom = train_target_transform
    train_dataset = VOCSegmentation(root=args.data_dir,
                                    image_set='train',
                                    download=True,
                                    transform=train_image_transform,
                                    target_transform=train_target_transform)
    val_dataset = VOCSegmentation(root=args.data_dir,
                                  image_set='val',
                                  download=False,
                                  transform=val_image_transform,
                                  target_transform=val_target_trasfrom)

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

    model = UNet(in_channels=3, n_classes=20)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    model.to(device)
    for epoch in range(args.n_epochs):
        print(f"Epoch: {epoch}/{args.n_epochs-1}")
        train_loss = train_one_epoch(model, opt, train_dataloader, loss_fn,
                                     device)
        val_loss = validate(model, val_dataloader, loss_fn, device)


if __name__ == '__main__':
    main()
