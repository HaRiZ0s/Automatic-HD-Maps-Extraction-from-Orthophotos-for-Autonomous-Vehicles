import argparse
import io
import os
import time

import PIL.Image
import lmdb
import json
import math
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.font_manager as font_manager


import matplotlib.pyplot as plt

from PIL import Image, ImageOps

from datetime import datetime, timedelta
from argparse import ArgumentParser


from collections import OrderedDict

import torch
import torchvision.transforms as T
import torchshow as ts

from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
from torchvision.utils import save_image

import src
import src.data.collate_funcs
from src.utils import MetricDict, make_grid2d
from src.data.dataloader import nuScenesMaps
import src.model.network as networks
from src import utils
from nuscenes.nuscenes import NuScenes
from pathlib import Path
from torchvision.transforms.functional import to_tensor, to_pil_image
import cv2

mpl.rcParams["font.family"] = "serif"
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + "/fonts/ttf/cmr10.ttf")
mpl.rcParams["font.serif"] = cmfont.get_name()
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True





def compute_loss(preds, labels, loss_name, args):

    scale_idxs = torch.arange(len(preds)).int()

    # Dice loss across classes at multiple scales
    ms_loss = torch.stack(
        [
            src.model.loss.__dict__[loss_name](pred, label, idx_scale, args)
            for pred, label, idx_scale in zip(preds, labels, scale_idxs)
        ]
    )

    if "90" not in args.model_name:
        total_loss = torch.sum(ms_loss[3:]) + torch.mean(ms_loss[:3])
    else:
        total_loss = torch.sum(ms_loss)

    # Store losses in dict
    total_loss_dict = {
        "loss": float(total_loss),
    }

    return total_loss, total_loss_dict

def image_calib_pad_and_crop(args, image, calib):

    og_w, og_h = 1600, 900
    desired_w, desired_h = args.desired_image_size
    scale_w, scale_h = desired_w / og_w, desired_h / og_h
    # Scale image
    image = image.resize((int(image.size[0] * scale_w), int(image.size[1] * scale_h)))
    # Pad images to the same dimensions
    w = image.size[0]
    h = image.size[1]
    delta_w = desired_w - w
    delta_h = desired_h - h
    pad_left = int(delta_w / 2)
    pad_right = delta_w - pad_left
    pad_top = int(delta_h / 2)
    pad_bottom = delta_h - pad_top
    left = 0 - pad_left
    right = pad_right + w
    top = 0 - pad_top
    bottom = pad_bottom + h
    image = image.crop((left, top, right, bottom))

    # Modify calibration matrices
    # Scale first two rows of calibration matrix
    calib[:2, :] *= scale_w
    # cx' = cx - du
    calib[0, 2] = calib[0, 2] + pad_left
    # cy' = cy - dv
    calib[1, 2] = calib[1, 2] + pad_top

    return image, calib


def visualize_score(scores,  heatmaps, grid, image, iou, num_classes, index):
    # Condese scores and ground truths to single map
    class_idx = torch.arange(len(scores)) + 1
    logits = scores.clone().cpu() * class_idx.view(-1, 1, 1)
    logits, _ = logits.max(dim=0)
    #
    scores = (scores.detach().clone().cpu()>0.5).float() * class_idx.view(-1, 1, 1)
    cls_idx = scores.clone()
    cls_idx = cls_idx.argmax(dim=0)
    cls_idx = cls_idx.numpy() * 20
    cls_idx = cv2.applyColorMap(cv2.convertScaleAbs(cls_idx, alpha=1), cv2.COLORMAP_JET)
    scores, _ = scores.max(dim=0)
    heatmaps = (heatmaps.detach().clone().cpu()>0.5).float() * class_idx.view(-1, 1, 1)

    heatmaps, _ = heatmaps.max(dim=0)



    # Visualize score
    fig = plt.figure(num="score", figsize=(8, 6))
    fig.clear()

    gs = mpl.gridspec.GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1:, 1])
    ax4 = fig.add_subplot(gs[1:, 2])

    image = ax1.imshow(image)
    ax1.grid(which="both")
    # ax2 = ax2.imshow(cls_idx)
    # ax3 = ax3.imshow(scores)
    # ax4 = ax4.imshow(heatmaps)
    image2 = ax2.imshow(cls_idx)

    image3 = ax3.imshow(scores)

#    image3.savefig('C:\\Users\\giann\\miniconda3\\envs\\planb\\translating-images-into-maps-main\\experiments\\fwto.jpg', 'JPEG')


#    image4 = ax4.imshow(heatmaps)


    grid = grid.cpu().detach().numpy()
    yrange = np.arange(grid[:, 0].max(), step=5)
    xrange = np.arange(start=grid[0, :].min(), stop=grid[0, :].max(), step=5)
    ymin, ymax = 0, grid[:, 0].max()
    xmin, xmax = grid[0, :].min(), grid[0, :].max()

    # ax2 = plt.vlines(xrange, ymin, ymax, color="white", linewidth=0.5)
    # ax2 = plt.hlines(yrange, xmin, xmax, color="white", linewidth=0.5)
    # ax3 = plt.vlines(xrange, ymin, ymax, color="white", linewidth=0.5)
    # ax3 = plt.hlines(yrange, xmin, xmax, color="white", linewidth=0.5)
    # ax4 = plt.vlines(xrange, ymin, ymax, color="white", linewidth=0.5)
    # ax4 = plt.hlines(yrange, xmin, xmax, color="white", linewidth=0.5)

    x2 = plt.vlines(xrange, ymin, ymax, color="white", linewidth=0.5)
    x2 = plt.hlines(yrange, xmin, xmax, color="white", linewidth=0.5)
    x3 = plt.vlines(xrange, ymin, ymax, color="white", linewidth=0.5)
    x3 = plt.hlines(yrange, xmin, xmax, color="white", linewidth=0.5)
    x4 = plt.vlines(xrange, ymin, ymax, color="white", linewidth=0.5)
    x4 = plt.hlines(yrange, xmin, xmax, color="white", linewidth=0.5)

    # ax2.vlines(xrange, ymin, ymax, color="white", linewidth=0.5)
    # ax2.hlines(yrange, xmin, xmax, color="white", linewidth=0.5)
    # ax3.vlines(xrange, ymin, ymax, color="white", linewidth=0.5)
    # ax3.hlines(yrange, xmin, xmax, color="white", linewidth=0.5)
    # ax4.vlines(xrange, ymin, ymax, color="white", linewidth=0.5)
    # ax4.hlines(yrange, xmin, xmax, color="white", linewidth=0.5)
    transform = T.RandomVerticalFlip(p=1)
    image3 = transform(scores)
    image4 = ax4.imshow(image3)

#    ts.show(image3)
#    save_image(image3, 'inference_map{}.jpg'.format(index))
# code gia white...amanatiadis
#    image4[np.where(np.all(image4[..., :3] == 0, -1))] = 0
#    cv2.imwrite("C:/Users/giann/miniconda3/envs/planb/translating-images-into-maps-main/experiments/transparent.png", image3)


    ax1.set_title("Input image", size=15)
    ax2.set_title("Model output logits", size=12)
    ax3.set_title("Model prediction = logits" + r"$ > 0.5$", size=12)
    ax4.set_title("opws Paper", size=12)

    # plt.suptitle(
    #     "IoU : {:.2f}".format(iou), size=14,
    # )

    gs.tight_layout(fig)
    gs.update(top=0.9)
    ts.save(image3, '.xarthsinference\inference_map{}.jpg'.format(index))
    return fig




def parse_args():
    parser = ArgumentParser()

    # ----------------------------- Data options ---------------------------- #
    parser.add_argument(
        "--root",
        type=str,
        default="nuscenes_data",
        help="root directory of the dataset",
    )
    parser.add_argument(
        "--nusc-version", type=str, default="v1.0-trainval", help="nuscenes version",
    )
    parser.add_argument(
        "--occ-gt",
        type=str,
        default="200down100up",
        help="occluded (occ) or unoccluded(unocc) ground truth maps",
    )
    parser.add_argument(
        "--gt-version",
        type=str,
        default="semantic_maps_new_200x200",
        help="ground truth name",
    )
    parser.add_argument(
        "--train-split", type=str, default="train_mini", help="ground truth name",
    )
    parser.add_argument(
        "--val-split", type=str, default="val_mini", help="ground truth name",
    )
    parser.add_argument(
        "--data-size",
        type=float,
        default=0.2,
        help="percentage of dataset to train on",
    )
    parser.add_argument(
        "--load-classes-nusc",
        type=str,
        nargs=14,
        default=[
            "drivable_area",
            "ped_crossing",
            "walkway",
            "carpark_area",
            "road_segment",
            "lane",
            "bus",
            "bicycle",
            "car",
        #    "construction_vehicle",
            "motorcycle",
            "trailer",
            "truck",
            "pedestrian",
            "trafficcone",
            "barrier",
        ],
        help="Classes to load for NuScenes",
    )
    parser.add_argument(
        "--pred-classes-nusc",
        type=str,
        nargs=12,
        default=[
            "drivable_area",
            "ped_crossing",
            "walkway",
            "carpark_area",
            "bus",
            "bicycle",
            "car",
        #    "construction_vehicle",
            "motorcycle",
            "trailer",
            "truck",
            "pedestrian",
            "trafficcone",
            "barrier",
        ],
        help="Classes to predict for NuScenes",
    )
    parser.add_argument(
        "--lidar-ray-mask",
        type=str,
        default="dense",
        help="sparse or dense lidar ray visibility mask",
    )
    parser.add_argument(
        "--grid-size",
        type=float,
        nargs=2,
        default=(50.0, 50.0),
        help="width and depth of validation grid, in meters",
    )
    parser.add_argument(
        "--z-intervals",
        type=float,
        nargs="+",
        default=[1.0, 9.0, 21.0, 39.0, 51.0],
        help="depths at which to predict BEV maps",
    )
    parser.add_argument(
        "--grid-jitter",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="magn. of random noise applied to grid coords",
    )
    parser.add_argument(
        "--aug-image-size",
        type=int,
        nargs="+",
        default=[1280, 720],
        help="size of random image crops during training",
    )
    parser.add_argument(
        "--desired-image-size",
        type=int,
        nargs="+",
        default=[1600, 900],
        help="size images are padded to before passing to network",
    )
    parser.add_argument(
        "--yoffset",
        type=float,
        default=1.74,
        help="vertical offset of the grid from the camera axis",
    )

    # -------------------------- Model options -------------------------- #
    parser.add_argument(
        "--model-name",
        type=str,
        default="PyrOccTranDetr_S_0904_old_rep100x100_out100x100",
        help="Model to train",
    )
    parser.add_argument(
        "-r",
        "--grid-res",
        type=float,
        default=0.5,
        help="size of grid cells, in meters",
    )
    parser.add_argument(
        "--frontend",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50"],
        help="name of frontend ResNet architecture",
    )
    parser.add_argument(
        "--pretrained",
        type=bool,
        default=True,
        help="choose pretrained frontend ResNet",
    )
    parser.add_argument(
        "--pretrained-bem",
        type=bool,
        default=True,
        help="choose pretrained BEV estimation model",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="experiments/tiim_220613",
        help="name of pretrained model to load",
    )
    parser.add_argument(
        "--load-ckpt",
        type=str,
        default="checkpoint-0600.pth.gz",
        help="name of checkpoint to load",
    )
    parser.add_argument(
        "--ignore", type=str, default=["nothing"], help="pretrained modules to ignore",
    )
    parser.add_argument(
        "--ignore-reload",
        type=str,
        default=["nothing"],
        help="pretrained modules to ignore",
    )
    parser.add_argument(
        "--focal-length", type=float, default=1266.417, help="focal length",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs=4,
        default=[8.0, 16.0, 32.0, 64.0],
        help="resnet frontend scale factor",
    )
    parser.add_argument(
        "--cropped-height",
        type=float,
        nargs=4,
        default=[20.0, 20.0, 20.0, 20.0],
        help="resnet feature maps cropped height",
    )
    parser.add_argument(
        "--y-crop",
        type=float,
        nargs=4,
        default=[15, 15.0, 15.0, 15.0],
        help="Max y-dimension in world space for all depth intervals",
    )
    parser.add_argument(
        "--dla-norm",
        type=str,
        default="GroupNorm",
        help="Normalisation for inputs to topdown network",
    )
    parser.add_argument(
        "--bevt-linear-additions",
        type=str2bool,
        default=False,
        help="BatchNorm, ReLU and Dropout addition to linear layer in BEVT",
    )
    parser.add_argument(
        "--bevt-conv-additions",
        type=str2bool,
        default=False,
        help="BatchNorm, ReLU and Dropout addition to conv layer in BEVT",
    )
    parser.add_argument(
        "--dla-l1-nchannels",
        type=int,
        default=64,
        help="vertical offset of the grid from the camera axis",
    )
    parser.add_argument(
        "--n-enc-layers",
        type=int,
        default=2,
        help="number of transfomer encoder layers",
    )
    parser.add_argument(
        "--n-dec-layers",
        type=int,
        default=2,
        help="number of transformer decoder layers",
    )

    # ---------------------------- Loss options ---------------------------- #
    parser.add_argument(
        "--loss", type=str, default="dice_loss_mean", help="Loss function",
    )
    parser.add_argument(
        "--exp-cf",
        type=float,
        default=0.0,
        help="Exponential for class frequency in weighted dice loss",
    )
    parser.add_argument(
        "--exp-os",
        type=float,
        default=0.2,
        help="Exponential for object size in weighted dice loss",
    )

    # ------------------------ Optimization options ----------------------- #
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    parser.add_argument("-l", "--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.99,
        help="factor to decay learning rate by every epoch",
    )

    # ------------------------- Training options ------------------------- #
    parser.add_argument(
        "-e", "--epochs", type=int, default=100, help="number of epochs to train for"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=1, help="mini-batch size for training"
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=5,
        help="Gradient accumulation over number of batches",
    )

    # ------------------------ Experiment options ----------------------- #
    parser.add_argument(
        "--name", type=str,
        default="inference",
        help="name of experiment",
    )
    parser.add_argument(
        "-s",
        "--savedir",
        type=str,
        default="experiments",
        help="directory to save experiments to",
    )
    # todo
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        nargs="*",
        default=[0],
        help="ids of gpus to train on. Leave empty to use cpu",
    )
    parser.add_argument(
        "--num-gpu", type=int, default=1, help="number of gpus",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=0,
        help="number of worker threads to use for data loading",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=1,
        help="number of epochs between validation runs",
    )
    parser.add_argument(
        "--print-iter",
        type=int,
        default=5,
        help="print loss summary every N iterations",
    )
    parser.add_argument(
        "--vis-iter",
        type=int,
        default=20,
        help="display visualizations every N iterations",
    )
    return parser.parse_args()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def _make_experiment(args):
    print("\n" + "#" * 80)
    # print(datetime.now().strftime("%A %-d %B %Y %H:%M"))
    print(datetime.now().strftime("%A %d %B %Y %H:%M"))
    print(
        "Creating experiment '{}' in directory:\n  {}".format(args.name, args.savedir)
    )
    print("#" * 80)
    print("\nConfig:")
    for key in sorted(args.__dict__):
        print("  {:12s} {}".format(key + ":", args.__dict__[key]))
    print("#" * 80)

    # Create a new directory for the experiment
    savedir = os.path.join(args.savedir, args.name)
    os.makedirs(savedir, exist_ok=True)

    # # Create tensorboard summary writer
    summary = SummaryWriter(savedir)

    # # Save configuration to file
    with open(os.path.join(savedir, "config.txt"), "w") as fp:
        json.dump(args.__dict__, fp)

    # # Write config as a text summary
    # summary.add_text(
    #     "config",
    #     "\n".join("{:12s} {}".format(k, v) for k, v in sorted(args.__dict__.items())),
    # )
    # summary.file_writer.flush()

    return None

def merge_map_classes(mapsdict):
    classes_to_merge = ["drivable_area", "road_segment", "lane"]
    merged_class = 'drivable_area'
    maps2merge = torch.stack([mapsdict[k] for k in classes_to_merge])  # [n, 1, 200, 200]
    maps2merge = maps2merge.sum(dim=0)
    maps2merge = (maps2merge > 0).float()
    mapsdict[merged_class] = maps2merge
    del mapsdict['road_segment'], mapsdict['lane']
    return mapsdict


def main():

    # Parse command line arguments
    args = parse_args()
    args.root = os.path.join(os.getcwd(), args.root)
    print(args.root)
    args.savedir = os.path.join(os.getcwd(), args.savedir)
    print(args.savedir)


    # Build depth intervals along Z axis and reverse
    z_range = args.z_intervals
    # BEV图预测的深度是（50，50）
    args.grid_size = (z_range[-1] - z_range[0], z_range[-1] - z_range[0])

    # Calculate cropped heights of feature maps
    h_cropped = src.utils.calc_cropped_heights(
        args.focal_length, np.array(args.y_crop), z_range, args.scales
    )


    model = networks.__dict__[args.model_name](
        num_classes=len(args.pred_classes_nusc),
        frontend=args.frontend,
        grid_res=args.grid_res,
        pretrained=args.pretrained,
        img_dims=args.desired_image_size,
        z_range=z_range,
        h_cropped=args.cropped_height,
        dla_norm=args.dla_norm,
        additions_BEVT_linear=args.bevt_linear_additions,
        additions_BEVT_conv=args.bevt_conv_additions,
        dla_l1_n_channels=args.dla_l1_nchannels,
        n_enc_layers=args.n_enc_layers,
        n_dec_layers=args.n_dec_layers,
    )

    # print(model.frontend)

    print("loaded pretrained model")
    device = torch.device("cuda")

    # model_pth = './best2.pth.gz'
    # model_dict = torch.load(model_pth, map_location = 'cpu')["model"]
    # # mod_dict = OrderedDict()
    # mod_dict = OrderedDict()
    #
    pretrained_pth = './checkpoint-0008.pth.gz'
    #pretrained_dict = torch.load(pretrained_pth)["model"]
    pretrained_dict = torch.load(pretrained_pth, map_location='cpu')["model"]
    #pretrained_dict = torch.load(pretrained_pth, map_location='gpu')["model"]
    mod_dict = OrderedDict()
    # # # Remove "module" from name

    for k, v in pretrained_dict.items():
         if any(module in k for module in args.ignore):
             continue
         else:
             name = k[7:]
             mod_dict[name] = v

    model.load_state_dict(mod_dict, strict=False)



    # # Remove "module" from name
    for k, v in pretrained_dict.items():
        if any(module in k for module in args.ignore):
            continue
        else:
            name = k[7:]
            mod_dict[name] = v
    miss_keys, unexpect_keys = model.load_state_dict(mod_dict, strict=False)
    # print(miss_keys)
    # print(unexpect_keys)
    model.eval()



    #root = 'D:/BEV/num/v1.0-mini'
    root = 'nuscenes_data'
    map_root = 'nuscenes_data\\v1.0-mini\\label'
    nusc = NuScenes(version="v1.0-mini",
                    dataroot=root,
                    verbose=False)

    def read_split(filename):
        """
        Read a list of NuScenes sample tokens
        """
        with open(filename, "r") as f:
            lines = f.read().split("\n")
            return [val for val in lines if val != ""]

    # 选择使用mini数据集的训练集还是测试集
    split = 'train_mini'
    tokens = read_split(
        os.path.join(root, "splits", "{}.txt".format(split)))

    for index in range(len(tokens)):
        print("\n==> inference begin")
        #time1 = time.time()
        img_path = 'nuscenes_data\\lmdb\\samples\\CAM_FRONT\\'
        map_path = 'nuscenes_data\\v1.0-mini\\maps\\'
        path_name = 'try'
        #path_name = 'n015-2018-07-24-11-22-345-0800__CAM_FRONT__'
        # print(tokens)
        print()
        sample_token = tokens[index]
        # 关键帧的内容
        sample_record = nusc.get("sample", sample_token)
        # 获取关键帧前方相机的识别序号
        cam_token = sample_record["data"]["CAM_FRONT"]
        # pic = Image.open(nusc.get_sample_data_path(cam_token))
        # pic.show()

        label_path = os.path.join(map_root, cam_token + '.png')
        # label = Image.open(label_path)
        # label.show()


        # 获取前方相机关键帧的数据值
        cam_record = nusc.get("sample_data", cam_token)

        cam_path = nusc.get_sample_data_path(cam_token)
        img_id = Path(cam_path).stem


        # 矫正参数
        # print(cam_record)
        # print()
        calib = nusc.get(
            "calibrated_sensor", cam_record["calibrated_sensor_token"]
        )["camera_intrinsic"]
        calib = np.array(calib)
        # 读取图片 read image
        image_save_path = './experiments/img/'
        if not os.path.exists(image_save_path):
            os.mkdir(image_save_path)
        #image = Image.open(img_path + img_id + '.jpg')
        image = Image.open( img_path + 'new_image{}.jpg'.format(index))
        image.save(image_save_path + 'inference_token{}.png'.format(index))
        # image = Image.open('./3.jpg')
        image, calib = image_calib_pad_and_crop(args, image, calib)
        image = to_tensor(image)
        calib = to_tensor(calib).reshape(3, 3)

        # image.save('1.jpg')


        # 这块照搬的原作者，因为不知道如何产生地图数据，这部分也可以不需要，只是为了将gt显示出来
        # 读取地图
        classes =[
            "drivable_area",
            "ped_crossing",
            "walkway",
            "carpark_area",
            "road_segment",
            "lane",
            "bus",
            "bicycle",
            "car",
        #    "construction_vehicle",
            "motorcycle",
            "trailer",
            "truck",
            "pedestrian",
            "trafficcone",
            "barrier",
            "lidar_ray_mask_dense",
        ]
        # Load ground truth maps

        gtmaps_db = lmdb.open(
            path='./nuscenes_data/lmdb/semantic_maps_new_200x200',
            readonly=True,
            readahead=False,
            max_spare_txns=128,
            lock=False,
        )
        gt_out_size = (200, 200)
        fov_mask = Image.open('./nuscenes_data/lmdb/semantic_maps_new_200x200/fov_mask.png')
        gtmaps_key = [pickle.dumps("{}___{}".format(img_id, cls)) for cls in classes]
        with gtmaps_db.begin() as txn:
            value = [txn.get(key=key) for key in gtmaps_key]
            # dst = './pic/'
            # for i, im in enumerate(value):
            #     path = io.BytesIO(im)
            #     pic = Image.open(path)
            #     pic.save(dst +  str(i) + '.jpg')
            gtmaps = [Image.open(io.BytesIO(im)) for im in value]

        # each map is of shape [1, 200, 200]
        mapsdict = {cls: to_tensor(map) for cls, map in zip(classes, gtmaps)}
        mapsdict["fov_mask"] = to_tensor(fov_mask)
        mapsdict = merge_map_classes(mapsdict)

        # Create visbility mask from lidar and fov masks
        lidar_ray_mask = mapsdict['lidar_ray_mask_dense']
        fov_mask = mapsdict['fov_mask']
        vis_mask = lidar_ray_mask * fov_mask
        mapsdict['vis_mask'] = vis_mask

        del mapsdict['lidar_ray_mask_dense'], mapsdict['fov_mask']

        # downsample maps to required output resolution
        mapsdict = {
            cls: F.interpolate(cls_map.unsqueeze(0), size=gt_out_size).squeeze(0)
            for cls, cls_map in mapsdict.items()
        }

        # apply vis mask to maps
        mapsdict = {
            cls: cls_map * mapsdict['vis_mask'] for cls, cls_map in mapsdict.items()
        }

        cls_maps = torch.cat(
            [cls_map for cls, cls_map in mapsdict.items() if 'mask' not in cls], dim=0
        )
        vis_mask = mapsdict['vis_mask']

        # gtmaps = Image.open(map_path + )

        grid_res = 0.5
        grid_size = args.grid_size
        grid2d = utils.make_grid2d(grid_size, (-grid_size[0] / 2.0, 0.0), grid_res)


        image = torch.unsqueeze(image, dim = 0)
        calib = torch.unsqueeze(calib, dim = 0)
        grid2d = torch.unsqueeze(grid2d, dim = 0)
        cls_maps = torch.unsqueeze(cls_maps, dim = 0)
        vis_mask = torch.unsqueeze(vis_mask, dim = 0)
        num_classes = 14

        with torch.no_grad():

            # Run network forwards
            #pred_ms = model(image, calib, grid2d)
            pred_ms = model(image, calib, grid2d)


            # Upsample largest prediction to 200x200
            pred_200x200 = F.interpolate(
                pred_ms[0], size=(200, 200), mode="bilinear"
            )

            # 将array的格式转化为图片格式 Convert array format to image format
            pred_200x200 = (pred_200x200 > 0).float()

            pred_ms = [pred_200x200, *pred_ms]





            # Get required gt output sizes
            map_sizes = [pred.shape[-2:] for pred in pred_ms]


            # Convert ground truth to binary mask
            gt_s1 = (cls_maps > 0).float()
            vis_mask_s1 = (vis_mask > 0.5).float()

            # Downsample to match model outputs
            gt_ms = src.utils.downsample_gt(gt_s1, map_sizes)
            vis_ms = src.utils.downsample_gt(vis_mask_s1, map_sizes)

            # Compute IoU
            iou_per_sample, iou_dict = src.utils.compute_multiscale_iou(
                pred_ms, gt_ms, vis_ms, num_classes
            )


            # Visualize predictions
            vis_img = ToPILImage()(image[0].detach().cpu())
            # vis_img.show()
            pred_vis = pred_ms[0].detach().cpu()
            label_vis = gt_ms[1]

            # Visualize scores
            vis_fig = visualize_score(
                pred_vis[0],
                label_vis[0],
                grid2d[0],
                vis_img,
                iou_per_sample[0],
                num_classes,
                index

            )
            plt.savefig(
                os.path.join(
                    args.savedir,
                    args.name,
                    "inference_token{}.png".format(index),
                )
            )
        time2 = time.time()
        #print(time2 - time1)
        print("\n==> inference complete")





if __name__ == "__main__":
    main()
