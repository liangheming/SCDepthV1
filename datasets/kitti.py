import os
import cv2 as cv
import numpy as np

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def cv_img_list_show(images):
    images = [(img * 255).permute(1, 2, 0).numpy().astype(np.uint8)[..., ::-1] for img in images]
    cat_imgs = np.concatenate(images, axis=1)
    return cat_imgs


class KittiDataset(Dataset):
    def __init__(self, data_dir, split_txt, ref_ids, transform):
        self.data_dir = data_dir
        self.split_txt = split_txt
        self.ref_ids = ref_ids
        self.transform = transform
        self.gt_data = None
        self.frame_ids = [0]
        self.frame_ids.extend(ref_ids)
        self.datalist = self.get_samples()
        self.normalized_k = np.array([[0.58, 0, 0.5],
                                      [0, 1.92, 0.5],
                                      [0, 0, 1]], dtype=np.float32)

    def get_samples(self):
        ret = list()
        side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        rf = open(self.split_txt, 'r')
        for line in rf.readlines():
            dir_name, frame_id, camera_id = line.split()
            frame_id = int(frame_id)
            img_sequence = list()
            for frame in self.frame_ids:
                img_name = "{:0>10d}.jpg".format(frame + frame_id)
                mid_dir = "image_0{:d}/data".format(side_map[camera_id])
                img_path = os.path.join(self.data_dir, dir_name, mid_dir, img_name)
                assert os.path.exists(img_path), "{:s} is not exit".format(img_path)
                img_sequence.append(img_path)
            ret.append(img_sequence)
        return ret

    def load_gt(self, gt_path, width, height):
        assert os.path.exists(gt_path), "{:s} is not exist".format(gt_path)
        with open(self.split_txt, 'r') as rf:
            text_len = len(rf.readlines())
        gt_data = np.load(gt_path, allow_pickle=True)["data"]
        assert len(gt_data) == text_len, \
            "lines of \"{:s}\" should be equal to the len of \"{:s}\"".format(self.split_txt, gt_path)
        self.gt_data = [cv.resize(d, (width, height), interpolation=cv.INTER_NEAREST) for d in gt_data]

    def __getitem__(self, item):
        sample = self.datalist[item]
        images = [pil_loader(path) for path in sample]
        intrinsics = self.normalized_k.copy()
        intrinsics[0] *= images[0].width
        intrinsics[1] *= images[0].height
        images, intrinsics = self.transform(images, intrinsics)
        inv_intrinsics = intrinsics.copy()
        inv_intrinsics[[0, 1], [2, 2]] = -inv_intrinsics[[0, 1], [2, 2]] / inv_intrinsics[[0, 1], [0, 1]]
        inv_intrinsics[[0, 1], [0, 1]] = 1 / inv_intrinsics[[0, 1], [0, 1]]

        # return images, intrinsics, inv_intrinsics
        return {
            "tgt_img": images[0],
            "ref_imgs": images[1:],
            "k": torch.from_numpy(intrinsics),
            "k_inv": torch.from_numpy(inv_intrinsics),
            "depth": torch.zeros(size=(0,)) if self.gt_data is None else torch.from_numpy(self.gt_data[item])
        }

    def __len__(self):
        return len(self.datalist)


if __name__ == '__main__':
    from datasets.custom_transforms import Compose, Resize, ToTensor, Normalize
    from torch.utils.data.dataloader import DataLoader

    train_transform = Compose(transforms=[
        # RandomHFlip(),
        Resize(width=640, height=192),
        ToTensor(),
        Normalize()
    ])

    data = KittiDataset(data_dir="/home/lion/large_data/data/kitti/raw",
                        split_txt="../splits/eigen/test_files.txt",
                        ref_ids= [], transform=train_transform)
    data.load_gt("/home/lion/temp/gt_depths.npz", width=640, height=192)
    loader = DataLoader(dataset=data, batch_size=8)
    for da in loader:
        print(da['tgt_img'].shape)
        print(len(da['ref_imgs']))
        print(da['depth'].shape)
        print(da['k'].shape, da['k_inv'].shape)
        break
