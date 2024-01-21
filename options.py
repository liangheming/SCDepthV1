import argparse


def get_train_parser():
    parser = argparse.ArgumentParser('Set SCDepth train', add_help=False)
    parser.add_argument('--depth_encode', default="resnet18", type=str)
    parser.add_argument('--depth_encode_pretrained', default=True, type=bool)
    parser.add_argument('--pose_encode', default="resnet18", type=str)
    parser.add_argument('--pose_encode_pretrained', default=True, type=bool)
    parser.add_argument('--width', default=832, type=int)
    parser.add_argument('--height', default=256, type=int)
    parser.add_argument('--photo_weight', default=1.0, type=float)
    parser.add_argument('--geometry_weight', default=0.1, type=float)
    parser.add_argument('--smooth_weight', default=0.1, type=float)

    parser.add_argument('--model_name', default="M", type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--devices', nargs="+", default=[0])

    parser.add_argument('--kitti_dir', default="/data/liangheming/depth/kitti")
    parser.add_argument('--train_split', default="splits/eigen_zhou/train_files.txt", type=str)
    parser.add_argument('--val_split', default="splits/eigen/test_files.txt", type=str)
    parser.add_argument('--val_mode', default="depth", type=str)
    parser.add_argument('--epoch_size', default=1000, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--gt_path', default="/data/liangheming/depth/kitti/gt_depths.npz", type=str)
    parser.add_argument('--ref_ids', nargs="+", default=[-1, 1])

    return parser


def get_evaluation_parser():
    parser = argparse.ArgumentParser('Set FeatDepth', add_help=False)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--split_path', default="splits/eigen/test_files.txt", type=str)
    parser.add_argument('--cuda', default=False, type=bool)
    return parser
