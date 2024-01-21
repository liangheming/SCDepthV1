import os
import torch
import argparse
import pytorch_lightning as pl
from options import get_train_parser
from wrappers.data_modules import SequenceDataModule
from wrappers.sc_depthv1 import SCDepth

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


def main(args):
    model_name = "{:s}_{:d}x{:d}".format(args.model_name, args.width, args.height)
    system = SCDepth(args)
    dm = SequenceDataModule(args)
    logger = TensorBoardLogger(
        save_dir="workspace",
        name=model_name,
        default_hp_metric=False
    )
    bar_callback = ProgressBar(refresh_rate=5)
    ckpt_dir = "workspace/{:s}/version_{:d}/cpkt".format(model_name,
                                                         logger.version)
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
        devices=args.devices,
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        callbacks=[bar_callback, checkpoint_callback, lr_monitor],
        logger=logger,
        benchmark=True,
        sync_batchnorm=True if len(args.devices) > 1 else False
        # accumulate_grad_batches=args.accumulate_grad_batches
    )
    trainer.fit(system, dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("SC_Depth Train", parents=[get_train_parser()])
    train_args = parser.parse_args()
    main(train_args)
