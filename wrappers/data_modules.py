from datasets.kitti import KittiDataset
from pytorch_lightning import LightningDataModule
from datasets import custom_transforms as ctf
from torch.utils.data import DataLoader, RandomSampler


class SequenceDataModule(LightningDataModule):
    def __init__(self, hparams):
        super(SequenceDataModule, self).__init__()
        self.save_hyperparameters()
        self.train_transform = ctf.Compose([
            ctf.RandomHFlip(),
            ctf.RandomScaleCrop(),
            ctf.Resize(width=hparams.width, height=hparams.height),
            ctf.ToTensor(),
            ctf.Normalize()
        ])

        self.valid_transform = ctf.Compose([
            ctf.Resize(width=hparams.width, height=hparams.height),
            ctf.ToTensor(),
            ctf.Normalize()
        ])

    def setup(self, stage: str):
        self.train_dataset = KittiDataset(data_dir=self.hparams.hparams.kitti_dir,
                                          split_txt=self.hparams.hparams.train_split,
                                          ref_ids=self.hparams.hparams.ref_ids,
                                          transform=self.train_transform)
        if self.hparams.hparams.val_mode == "depth":
            self.val_dataset = KittiDataset(
                data_dir=self.hparams.hparams.kitti_dir,
                split_txt=self.hparams.hparams.val_split,
                ref_ids=[],
                transform=self.valid_transform
            )
            self.val_dataset.load_gt(
                self.hparams.hparams.gt_path,
                self.hparams.hparams.width,
                self.hparams.hparams.height
            )
        elif self.hparams.hparams.val_mode == "photo":
            self.val_dataset = KittiDataset(
                data_dir=self.hparams.hparams.kitti_dir,
                split_txt=self.hparams.hparams.val_split,
                ref_ids=self.hparams.hparams.ref_ids,
                transform=self.valid_transform
            )
        else:
            raise NotImplementedError("val_model only support 'depth' or 'photo' ")

    def train_dataloader(self):
        sampler = RandomSampler(self.train_dataset,
                                replacement=True,
                                num_samples=self.hparams.hparams.batch_size * self.hparams.hparams.epoch_size)
        return DataLoader(self.train_dataset,
                          sampler=sampler,
                          num_workers=self.hparams.hparams.num_workers,
                          batch_size=self.hparams.hparams.batch_size,
                          pin_memory=True,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.hparams.hparams.num_workers,
            batch_size=self.hparams.hparams.batch_size,
            pin_memory=True,
            persistent_workers=True
        )
