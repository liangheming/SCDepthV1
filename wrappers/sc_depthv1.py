import torch
import numpy as np
from models.layers import DepthNet, PoseNet
from models.losses import PhotoAndGeometryLoss, SmoothLoss
from pytorch_lightning import LightningModule
from utils.metrics import compute_errors
from utils.visualize import visualize_image, visualize_depth


def compute_depth_metrics(preds: torch.Tensor, gts: torch.Tensor):
    batch_size, h, w = gts.size()
    crop_mask = gts[0] != gts[0]
    y1, y2 = int(0.40810811 * h), int(0.99189189 * h)
    x1, x2 = int(0.03594771 * w), int(0.96405229 * w)
    crop_mask[y1:y2, x1:x2] = True
    ret = list()
    for p, g in zip(preds, gts):
        valid_mask = (g > 0.001) & (g < 80.0) & crop_mask
        valid_gt = g[valid_mask]
        valid_pred = p[valid_mask]
        mean_scale = valid_gt.median() / valid_pred.median()
        valid_pred = (valid_pred * mean_scale).clamp(0.001, 80.0)
        ret.append(compute_errors(valid_gt, valid_pred))
    return torch.stack(ret).mean(dim=0)


# noinspection PyUnresolvedReferences
class SCDepth(LightningModule):
    def __init__(self, hparams):
        super(SCDepth, self).__init__()
        self.save_hyperparameters()
        self.depth_net = DepthNet(hparams.depth_encode, hparams.depth_encode_pretrained)
        self.pose_net = PoseNet(hparams.pose_encode, hparams.pose_encode_pretrained)
        self.main_loss = PhotoAndGeometryLoss()
        self.smooth_loss = SmoothLoss()
        self.validation_metrics = list()
        self.random_show_batch_idx = 0

    def configure_optimizers(self):
        optim_params = [
            {'params': self.depth_net.parameters(), 'lr': self.hparams.hparams.lr},
            {'params': self.pose_net.parameters(), 'lr': self.hparams.hparams.lr}
        ]
        optimizer = torch.optim.Adam(optim_params)
        return optimizer

    def training_step(self, batch, batch_idx):
        tgt_img, ref_imgs, k, k_inv = batch['tgt_img'], batch['ref_imgs'], batch['k'], batch['k_inv']
        tgt_depth = self.depth_net(tgt_img)
        ref_depths = [self.depth_net(im) for im in ref_imgs]
        poses = [self.pose_net(tgt_img, im) for im in ref_imgs]
        poses_inv = [self.pose_net(im, tgt_img) for im in ref_imgs]
        loss_1, loss_2 = self.main_loss(tgt_img, ref_imgs, tgt_depth, ref_depths, k, k_inv, poses, poses_inv)
        loss_3 = self.smooth_loss(tgt_depth, tgt_img)

        w1 = self.hparams.hparams.photo_weight
        w2 = self.hparams.hparams.geometry_weight
        w3 = self.hparams.hparams.smooth_weight
        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3
        self.log('train/total_loss', loss, prog_bar=True)
        self.log('train/photo_loss', loss_1)
        self.log('train/geometry_loss', loss_2)
        self.log('train/smooth_loss', loss_3)
        return loss

    def on_validation_epoch_start(self):
        self.random_show_batch_idx = np.random.randint(self.trainer.num_val_batches[0] - 1)
        self.validation_metrics.clear()

    def validation_step(self, batch, batch_idx):
        if self.hparams.hparams.val_mode == 'depth':
            tgt_img, gt_depth = batch['tgt_img'], batch['depth']
            tgt_depth = self.depth_net(tgt_img)
            metrics = compute_depth_metrics(tgt_depth.squeeze(1), gt_depth)
            self.validation_metrics.append(metrics)

        elif self.hparams.hparams.val_mode == 'photo':
            tgt_img, ref_imgs, k, k_inv = batch['tgt_img'], batch['ref_imgs'], batch['k'], batch['k_inv']
            tgt_depth = self.depth_net(tgt_img)
            ref_depths = [self.depth_net(im) for im in ref_imgs]
            poses = [self.pose_net(tgt_img, im) for im in ref_imgs]
            poses_inv = [self.pose_net(im, tgt_img) for im in ref_imgs]
            loss_1, loss_2 = self.main_loss(tgt_img, ref_imgs, tgt_depth, ref_depths, k, k_inv, poses, poses_inv)

            self.validation_metrics.append(loss_1.item())
        else:
            raise NotImplementedError("val_model only support 'depth' or 'photo' ")

        if batch_idx == self.random_show_batch_idx:
            for i in range(min(4, len(tgt_img))):
                vis_img = visualize_image(tgt_img[i])
                vis_depth = visualize_depth(tgt_depth[i, 0])
                stack = torch.cat([vis_img, vis_depth], dim=1).unsqueeze(0)
                self.logger.experiment.add_images(
                    'a_img/img_depth_{}'.format(i), stack, self.current_epoch)

    def on_validation_epoch_end(self):
        metrics = torch.stack(self.validation_metrics).mean(dim=0)
        if self.hparams.hparams.val_mode == 'depth':
            abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = metrics
            self.log("loss_val", abs_rel, prog_bar=True)
            self.logger.experiment.add_scalar("val/abs_rel", abs_rel, self.current_epoch)
            self.logger.experiment.add_scalar("val/sq_rel", sq_rel, self.current_epoch)
            self.logger.experiment.add_scalar("val/rmse", rmse, self.current_epoch)
            self.logger.experiment.add_scalar("val/rmse_log", rmse_log, self.current_epoch)
            self.logger.experiment.add_scalar("val/a1", a1, self.current_epoch)
            self.logger.experiment.add_scalar("val/a2", a2, self.current_epoch)
            self.logger.experiment.add_scalar("val/a3", a3, self.current_epoch)
        elif self.hparams.hparams.val_mode == 'photo':
            self.logger.experiment.add_scalar("val/ph_loss", metrics, self.current_epoch)
            self.log("loss_val", metrics, prog_bar=True)

    def forward(self, x):
        return self.depth_net(x)
