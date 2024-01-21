import os
import torch
import argparse
import pytorch_lightning as pl
from options import get_train_parser
from models.layers import DepthNet, PoseNet
from models.losses import PhotoAndGeometryLoss, SmoothLoss
from datasets.kitti import KittiDataset
from torch.utils.data.dataloader import DataLoader
from wrappers.data_modules import SequenceDataModule
from wrappers.sc_depthv1 import SCDepth
from datasets.custom_transforms import Compose, Resize, ToTensor, Normalize

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

pl.seed_everything(1994)
# if support
torch.set_float32_matmul_precision('high')


class ProgressBar(TQDMProgressBar):
    def get_metrics(self, *args, **kwargs):
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items


def debug(args):
    val_transform = Compose(
        transforms=[
            Resize(width=640, height=192),
            ToTensor(),
            Normalize()
        ]
    )
    v_data = KittiDataset(data_dir=args.kitti_dir,
                          split_txt=args.train_split,
                          ref_ids=args.ref_ids, transform=val_transform)
    v_loader = DataLoader(dataset=v_data,
                          batch_size=4,
                          num_workers=args.num_workers,
                          shuffle=False, drop_last=False)
    pose = PoseNet(args.pose_encode, args.pose_encode_pretrained)
    depth = DepthNet(args.depth_encode, args.depth_encode_pretrained)
    loss = PhotoAndGeometryLoss()
    smooth_loss = SmoothLoss()
    for data in v_loader:
        tgt_img, ref_imgs, k, k_inv = data['tgt_img'], data['ref_imgs'], data['k'], data['k_inv']
        tgt_depth = depth(tgt_img)
        print(tgt_depth.squeeze(1).shape)
        ref_depths = [depth(im) for im in ref_imgs]
        poses = [pose(tgt_img, im) for im in ref_imgs]
        poses_inv = [pose(im, tgt_img) for im in ref_imgs]
        ph_loss, geo_loss = loss(tgt_img, ref_imgs, tgt_depth, ref_depths, k, k_inv, poses, poses_inv)
        sm_loss = smooth_loss(tgt_depth, tgt_img)
        print(ph_loss, geo_loss, sm_loss)
        break


def main(args):
    system = SCDepth(args)
    dm = SequenceDataModule(args)
    logger = TensorBoardLogger(
        save_dir="workspace",
        name=args.model_name,
        default_hp_metric=False
    )
    bar_callback = ProgressBar(refresh_rate=5)
    ckpt_dir = "workspace/{:s}/version_{:d}".format(args.model_name, logger.version)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                          filename='{epoch}-{loss_val:.4f}',
                                          monitor='loss_val',
                                          mode='min',
                                          save_last=True,
                                          save_weights_only=True,
                                          save_top_k=5)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        default_root_dir=ckpt_dir,
        accelerator='gpu',
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        callbacks=[bar_callback, checkpoint_callback, lr_monitor],
        logger=logger,
        benchmark=True,
        # accumulate_grad_batches=args.accumulate_grad_batches
    )
    trainer.fit(system, dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("SC_Depth Train", parents=[get_train_parser()])
    train_args = parser.parse_args()
    main(train_args)
