import torch
import numpy as np


def compute_errors(gt, pred):
    assert gt.__class__ == pred.__class__
    if isinstance(pred, torch.Tensor):
        return compute_errors_torch(gt, pred)
    elif isinstance(pred, np.ndarray):
        return compute_errors_np(gt, pred)
    else:
        raise NotImplementedError()


def compute_errors_torch(gt, pred):
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())
    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean((gt - pred) ** 2 / gt)
    return torch.stack([abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])


def compute_errors_np(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return np.stack([abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])
