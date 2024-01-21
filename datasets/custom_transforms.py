import torch
import random
import numpy as np
from PIL import Image
from torchvision.transforms import functional as tf
from typing import List


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, intrinsics):
        for t in self.transforms:
            images, intrinsics = t(images, intrinsics)
        return images, intrinsics


class RandomHFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images: List[Image.Image], intrinsics: np.ndarray):
        if random.random() > self.p:
            return images, intrinsics

        images = [tf.hflip(img) for img in images]
        if intrinsics is not None:
            intrinsics = intrinsics.copy()
            intrinsics[0, 2] = images[0].width - intrinsics[0, 2]
        return images, intrinsics


class RandomScaleCrop(object):
    def __init__(self, scale=1.15, resample=Image.Resampling.BILINEAR):
        assert scale > 1.0
        self.scale = scale
        self.resample = resample

    def __call__(self, images: List[Image.Image], intrinsics: np.ndarray):
        if intrinsics is not None:
            intrinsics = intrinsics.copy()
        im_h, im_w = images[0].height, images[0].width
        x_scaling, y_scaling = np.random.uniform(1, self.scale, 2)
        scaled_h, scaled_w = int(im_h * y_scaling), int(im_w * x_scaling)
        if intrinsics is not None:
            intrinsics[0] *= x_scaling
            intrinsics[1] *= y_scaling

        images = [img.resize((scaled_w, scaled_h), resample=self.resample) for img in images]
        offset_y = np.random.randint(scaled_h - im_h + 1)
        offset_x = np.random.randint(scaled_w - im_w + 1)
        images = [img.crop([offset_x, offset_y, offset_x + im_w, offset_y + im_h]) for img in images]

        if intrinsics is not None:
            intrinsics[0, 2] -= offset_x
            intrinsics[1, 2] -= offset_y
        return images, intrinsics


class Resize(object):
    def __init__(self, width, height, resample=Image.Resampling.BILINEAR):
        self.width = width
        self.height = height
        self.resample = resample

    def __call__(self, images: List[Image.Image], intrinsics: np.ndarray):
        im_h, im_w = images[0].height, images[0].width
        if im_h == self.height and im_w == self.width:
            return images, intrinsics
        images = [img.resize((self.width, self.height), resample=self.resample) for img in images]
        if intrinsics is not None:
            intrinsics = intrinsics.copy()
            intrinsics[0] *= (self.width * 1.0 / im_w)
            intrinsics[1] *= (self.height * 1.0 / im_h)
        return images, intrinsics


class ToTensor(object):
    def __call__(self, images: List[Image.Image], intrinsics: np.ndarray):
        images = [tf.to_tensor(img) for img in images]
        return images, intrinsics


class Normalize(object):
    def __init__(self, mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, images: List[torch.Tensor], intrinsics: np.ndarray):
        images = [tf.normalize(img, self.mean, self.std) for img in images]
        return images, intrinsics
