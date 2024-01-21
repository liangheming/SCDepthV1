import torch
from torch import nn
from utils.geometry import to_rt
from torch.nn import functional as f


def mean_on_mask(diff, mask):
    if mask.sum() > 100:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        mean_value = torch.tensor(0).float().to(mask.device)
    return mean_value


def get_homo_mesh(width, height):
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    mesh = torch.stack([x, y, torch.ones_like(x)], dim=-1).float()
    return mesh


def frame_to_frame(tgt_depth, r, t, k, k_inv):
    """
    :param tgt_depth:
    :param r:
    :param t:
    :param k:
    :param k_inv:
    :return:
    """
    _, _, h, w = tgt_depth.shape
    mesh = get_homo_mesh(height=h, width=w).to(tgt_depth.device)
    normalized_camera_points = torch.einsum("hwc,brc->bhwr", mesh, k_inv)
    camera_points = normalized_camera_points * tgt_depth.permute(0, 2, 3, 1)

    transformed_camera_points = torch.einsum("bhwc,brc->bhwr",
                                             camera_points,
                                             r) + t[:, None, None, :]
    projected_camera_points = torch.einsum("bhwc,brc->bhwr", transformed_camera_points, k)
    pix_coords = projected_camera_points[..., :2] / (projected_camera_points[..., 2:3] + 1e-6)
    return pix_coords, projected_camera_points[..., -1].unsqueeze(1)


def warp(ref_img, tgt_depth, ref_depth, pose, k, k_inv, padding_mode="zeros"):
    _, _, h, w = ref_img.shape
    r, t = to_rt(pose[:, :3], pose[:, 3:], mode="euler")
    pix_coords, computed_depth = frame_to_frame(tgt_depth, r, t, k, k_inv)
    pix_coords[..., 0] /= w - 1
    pix_coords[..., 1] /= h - 1
    pix_coords = (pix_coords - 0.5) * 2
    projected_img = f.grid_sample(ref_img, pix_coords, padding_mode=padding_mode, align_corners=False)
    projected_depth = f.grid_sample(ref_depth, pix_coords, padding_mode=padding_mode, align_corners=False)

    return projected_img, projected_depth, computed_depth


class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d(1)

    def __call__(self, disp, img):
        mean_disp = self.mean(disp)
        norm_disp = disp / (mean_disp + 1e-7)
        grad_disp_x = torch.abs(norm_disp[:, :, :, :-1] - norm_disp[:, :, :, 1:])
        grad_disp_y = torch.abs(norm_disp[:, :, :-1, :] - norm_disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        k = 7
        self.mu_x_pool = nn.AvgPool2d(k, 1)
        self.mu_y_pool = nn.AvgPool2d(k, 1)
        self.sig_x_pool = nn.AvgPool2d(k, 1)
        self.sig_y_pool = nn.AvgPool2d(k, 1)
        self.sig_xy_pool = nn.AvgPool2d(k, 1)
        self.refl = nn.ReflectionPad2d(k // 2)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0.0, 1.0)


class PhotoAndGeometryLoss(nn.Module):
    def __init__(self):
        super(PhotoAndGeometryLoss, self).__init__()
        self.ssim = SSIM()

    def forward(self, tgt_img, ref_imgs, tgt_depth, ref_depths, k, k_inv, poses, poses_inv):
        diff_img_list = []
        diff_color_list = []
        diff_depth_list = []
        valid_mask_list = []
        for ref_img, ref_depth, pose, pose_inv in zip(ref_imgs, ref_depths, poses, poses_inv):
            diff_img_tmp1, diff_color_tmp1, diff_depth_tmp1, valid_mask_tmp1 = \
                self.pairwise_photo_and_geometry_loss(tgt_img, ref_img, tgt_depth, ref_depth, pose, k, k_inv)
            diff_img_tmp2, diff_color_tmp2, diff_depth_tmp2, valid_mask_tmp2 = \
                self.pairwise_photo_and_geometry_loss(ref_img, tgt_depth, ref_depth, tgt_depth, pose_inv, k, k_inv)

            diff_img_list.extend([diff_img_tmp1, diff_img_tmp2])
            diff_color_list.extend([diff_color_tmp1, diff_color_tmp2])
            diff_depth_list.extend([diff_depth_tmp1, diff_depth_tmp2])
            valid_mask_list.extend([valid_mask_tmp1, valid_mask_tmp2])
        diff_img = torch.cat(diff_img_list, dim=1)
        diff_color = torch.cat(diff_color_list, dim=1)
        diff_depth = torch.cat(diff_depth_list, dim=1)
        valid_mask = torch.cat(valid_mask_list, dim=1)

        indices = torch.argmin(diff_color, dim=1, keepdim=True)
        diff_img = torch.gather(diff_img, 1, indices)
        diff_depth = torch.gather(diff_depth, 1, indices)
        valid_mask = torch.gather(valid_mask, 1, indices)

        photo_loss = mean_on_mask(diff_img, valid_mask)
        geometry_loss = mean_on_mask(diff_depth, valid_mask)
        return photo_loss, geometry_loss

    def pairwise_photo_and_geometry_loss(self, tgt_img, ref_img, tgt_depth, ref_depth, pose, k, k_inv):
        ref_img_warped, projected_depth, computed_depth = warp(ref_img, tgt_depth, ref_depth, pose, k, k_inv)
        # diff_depth = (computed_depth - projected_depth).abs() / (computed_depth + projected_depth + 1e-6)
        diff_depth = (computed_depth - projected_depth).abs() / (computed_depth + projected_depth + 1e-6)
        valid_compute_depth_mask = (computed_depth > 0.01).float() * (computed_depth < 100.0).float()
        valid_project_depth_mask = (projected_depth > 0.01).float() * (projected_depth < 100.0).float()
        valid_mask_ref = (ref_img_warped.abs().mean(dim=1, keepdim=True) > 1e-3).float()
        valid_mask_tgt = (tgt_img.abs().mean(dim=1, keepdim=True) > 1e-3).float()

        valid_mask = valid_mask_tgt * valid_mask_ref * valid_project_depth_mask * valid_compute_depth_mask

        diff_color = (tgt_img - ref_img_warped).abs().mean(dim=1, keepdim=True)
        identity_warp_err = (tgt_img - ref_img).abs().mean(dim=1, keepdim=True)
        auto_mask = (diff_color < identity_warp_err).float()
        valid_mask = auto_mask * valid_mask

        diff_img = (tgt_img - ref_img_warped).abs().clamp(0, 1)
        ssim_map = self.ssim(tgt_img, ref_img_warped)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)
        diff_img = diff_img.mean(dim=1, keepdim=True)
        weight_mask = (1 - diff_depth).detach()
        diff_img = diff_img * weight_mask

        return diff_img, diff_color, diff_depth, valid_mask
