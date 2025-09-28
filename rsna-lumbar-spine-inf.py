#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-09-28T02:07:35.859Z
"""

# ### Imports


import os
import gc
import re
import sys
import cv2
import glob
import json
import torch
import shutil
import warnings
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from scipy.special import softmax
from collections import Counter
from joblib import Parallel, delayed

warnings.simplefilter("ignore", FutureWarning)

if os.path.exists("/kaggle/input/rsna-lumbar-spine-code/src"):
    !cp -r /kaggle/input/rsna-lumbar-spine-code/src ./
    sys.path.append("src")

from util.torch import load_model_weights
from util.metrics import rsna_loss

from data.processing import process_and_save
from data.transforms import get_transfos
from data.dataset import CropDataset, CoordsDataset
from data.preparation import prepare_data_crop

from inference.dataset import FeatureInfDataset, SafeDataset
from inference.lvl1 import predict, Config
from inference.utils import sub_to_dict

if os.path.exists("/kaggle/input/timm-smp"):
    sys.path.append(
        "/kaggle/input/timm-smp/pytorch-image-models-main/pytorch-image-models-main"
    )
    sys.path.append(
        "/kaggle/input/timm-smp/segmentation_models.pytorch-master/segmentation_models.pytorch-master"
    )
from model_zoo.models import define_model
from model_zoo.models_lvl2 import define_model as define_model_2
# from model_zoo.models_seg import define_model as define_model_seg
# from model_zoo.models_seg import convert_3

from params import LEVELS_, SEVERITIES, LEVELS

# ### Params


EVAL = False
DEBUG = False

# ROOT_DATA_DIR = "../input/"
# DEBUG_DATA_DIR = "../output/dataset_debug/"  # Todo
# SAVE_FOLDER = "../output/tmp/"
# shutil.rmtree(SAVE_FOLDER)

ROOT_DATA_DIR = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/"
DEBUG_DATA_DIR = "/kaggle/input/rsna-2024-debug/"
SAVE_FOLDER = "/tmp/"

os.makedirs(SAVE_FOLDER, exist_ok=True)
os.makedirs(SAVE_FOLDER + "npy/", exist_ok=True)
os.makedirs(SAVE_FOLDER + "mid/", exist_ok=True)
os.makedirs(SAVE_FOLDER + "csv/", exist_ok=True)

DATA_PATH = ROOT_DATA_DIR + "test_images/"
folds_dict = {}

if DEBUG:
    df_meta = pd.read_csv(ROOT_DATA_DIR + "train_series_descriptions.csv")
else:
    df_meta = pd.read_csv(ROOT_DATA_DIR + "test_series_descriptions.csv")

df_meta["weighting"] = df_meta["series_description"].apply(lambda x: x.split()[1][:2])
df_meta["orient"] = df_meta["series_description"].apply(lambda x: x.split()[0])
df_meta["study_series"] = df_meta["study_id"].astype(str) + "_" + df_meta["series_id"].astype(str)

if DEBUG:
    if EVAL:
        DATA_PATH = ROOT_DATA_DIR + "train_images/"
        FOLDS_FILE = DEBUG_DATA_DIR + "train_folded_v1.csv"
        folds = pd.read_csv(FOLDS_FILE)
        df_meta = df_meta.merge(folds, how="left")
        df_meta = df_meta[df_meta['fold'] == 1].reset_index(drop=True)
    else:
        DATA_PATH = DEBUG_DATA_DIR + "debug_images/"
        df_meta = df_meta.head(3)

        # df_meta_ = df_meta.copy()
        # df_meta_['study_id'] += 1
        # df_meta_ = df_meta_[df_meta_['orient'] == "Axial"]
        # df_meta = pd.concat([df_meta, df_meta_], ignore_index=True)
        # df_meta["study_series"] = df_meta["study_id"].astype(str) + "_" + df_meta["series_id"].astype(str)

BATCH_SIZE = 32
BATCH_SIZE_2 = 512
USE_FP16 = True

NUM_WORKERS = os.cpu_count()

FOLD = 1 if DEBUG else "fullfit_0"
PLOT = DEBUG and not EVAL

EXP_FOLDERS = {}

COORDS_FOLDERS = {
    "sag": ("/kaggle/input/rsna-2024-weights-1/2024-08-29_0/", FOLD),
}

CROP_EXP_FOLDERS = {
    "crop": ("/kaggle/input/rsna-2024-weights-2/2024-10-04_1/", [FOLD], "crops_0.1"),
    "crop_2": ("/kaggle/input/rsna-2024-weights-2/2024-10-04_9/", [FOLD], "crops_0.1"),
    "scs_crop_coords": ("/kaggle/input/rsna-2024-weights-2/2024-10-04_34/", [FOLD], "crops_0.1"),  # 5f -0.005 scs
    "scs_crop_coords_2": ("/kaggle/input/rsna-2024-weights-2/2024-10-04_37/", [FOLD], "crops_0.1"),  # 3f -0.005 scs
}

EXP_FOLDERS_2 = [
    "/kaggle/input/rsna-2024-weights-2/2024-10-04_42/",  # 0.3861
]

FOLDS_2 = [FOLD] # if DEBUG else [0, 1, 2, 3]

# EXP_FOLDER_3D = "../logs/2024-07-31/25/"

for f in EXP_FOLDERS_2:
    folders = Config(json.load(open(f + "config.json", "r"))).exp_folders
    print("-> Level 2 model:", f)
    for k in folders:
        print(k, folders[k], EXP_FOLDERS.get(k, CROP_EXP_FOLDERS.get(k, ["?"]))[0])
    print()

    
for k in EXP_FOLDERS:
    assert os.path.exists(EXP_FOLDERS[k][0]), f"Model not found: {k}"
for k in CROP_EXP_FOLDERS:
    assert os.path.exists(CROP_EXP_FOLDERS[k][0]), f"Crop model not found: {k}"
for k in COORDS_FOLDERS:
    assert os.path.exists(COORDS_FOLDERS[k][0]), f"Coords model not found: {k}"

df_meta.head(5)

# ## Preparation


_ = Parallel(n_jobs=NUM_WORKERS)(
    delayed(process_and_save)(
        df_meta['study_id'][i],
        df_meta['series_id'][i],
        df_meta['orient'][i],
        DATA_PATH,
        save_folder=SAVE_FOLDER,
        save_meta=False,
        save_middle_frame=True,
    ) for i in tqdm(range(len(df_meta)))
)

# ## Sagittal Coords


df_sag = df_meta[df_meta["orient"] == "Sagittal"].reset_index(drop=True)
df_sag = df_sag[df_sag.columns[:6]]

df_sag['img_path'] = SAVE_FOLDER + "mid/" + df_sag["study_series"] + ".png"
df_sag['target'] = [np.ones((5, 2)) for _ in range(len(df_sag))]

df_sag.head(3)

config_sag = Config(json.load(open(COORDS_FOLDERS['sag'][0] + "config.json", "r")))

model_sag = define_model(
    config_sag.name,
    drop_rate=config_sag.drop_rate,
    drop_path_rate=config_sag.drop_path_rate,
    pooling=config_sag.pooling,
    num_classes=config_sag.num_classes,
    num_classes_aux=config_sag.num_classes_aux,
    n_channels=config_sag.n_channels,
    reduce_stride=config_sag.reduce_stride,
    pretrained=False,
)
model_sag = model_sag.cuda().eval()

weights = COORDS_FOLDERS['sag'][0] + f"{config_sag.name}_{COORDS_FOLDERS['sag'][1]}.pt"
model_sag = load_model_weights(model_sag, weights, verbose=1)

%%time
transfos = get_transfos(augment=False, resize=config_sag.resize, use_keypoints=True)
dataset = CoordsDataset(df_sag, transforms=transfos)
dataset = SafeDataset(dataset)

preds_sag, _ = predict(model_sag, dataset, config_sag.loss_config, batch_size=32, use_fp16=True)

DELTAS = [0.1]  #, 0.15]

for delta in DELTAS:
    os.makedirs(SAVE_FOLDER + f"crops_{delta}", exist_ok=True)

for idx in tqdm(range(len(df_sag))):
    study_series = df_sag["study_series"][idx]
    imgs_path = SAVE_FOLDER + "npy/" + study_series + ".npy"

    imgs = np.load(imgs_path)

    preds = preds_sag[idx].reshape(-1, 2).copy()

    for delta in DELTAS:  # , 0.15
        crops = np.concatenate([preds, preds], -1)
        crops[:, [0, 1]] -= delta
        crops[:, [2, 3]] += delta
        crops = crops.clip(0, 1)

        crops[:, [0, 2]] *= imgs.shape[2]
        crops[:, [1, 3]] *= imgs.shape[1]
        crops = crops.astype(int)

        img_crops = []
        for i, (x0, y0, x1, y1) in enumerate(crops):

            crop = imgs[:, y0: y1, x0: x1].copy()
            # crop = np.zeros((3, 1, 1))
            try:
                assert crop.shape[2] >= 1 and crop.shape[1] >= 1
            except AssertionError:
                # print('!!')
                # pass
                crop = imgs.copy()

            np.save(SAVE_FOLDER + f"crops_{delta}/{study_series}_{LEVELS_[i]}.npy", crop)
            img_crops.append(crop[len(crop) // 2])

        if PLOT:
            preds[:, 0] *= imgs.shape[2]
            preds[:, 1] *= imgs.shape[1]

            plt.figure(figsize=(8, 8))
            plt.imshow(imgs[len(imgs) // 2], cmap="gray")
            plt.scatter(preds[:, 0], preds[:, 1], marker="x", label="center")
            plt.title(study_series)
            plt.axis(False)
            plt.legend()
            plt.show()

            plt.figure(figsize=(20, 4))
            for i in range(5):
                plt.subplot(1, 5, i + 1)
                plt.imshow(img_crops[i], cmap="gray")
                plt.axis(False)
                plt.title(LEVELS[i])
            plt.show()

if DEBUG and not EVAL:
    ref_folder = DEBUG_DATA_DIR + "coords_crops_0.1_2/"
    df_ref = prepare_data_crop(ROOT_DATA_DIR, ref_folder).head(10)

    df_ref['img_path_2'] = df_ref['img_path'].apply(
        lambda x: re.sub(ref_folder, SAVE_FOLDER + f"crops_0.1/", x)
    )

    for i in range(len(df_ref)):
        cref = np.load(df_ref['img_path'][i])
        c = np.load(df_ref['img_path_2'][i])
        assert (cref == c).all()
        # plt.subplot(1, 2, 1)
        # plt.imshow(c[len(c) // 2], cmap="gray")
        # plt.subplot(1, 2, 2)
        # plt.imshow(cref[len(cref) // 2], cmap="gray")
        # plt.show()
        # break

# ## Crop models


df = df_meta.copy()

df["target"] = 0
df["coords"] = 0

df["level"] = [LEVELS for _ in range(len(df))]
df["level_"] = [LEVELS_ for _ in range(len(df))]
df = df.explode(["level", "level_"]).reset_index(drop=True)
df["img_path_"] = df["study_series"] + "_" + df["level_"] + ".npy"

crop_fts = {}
for mode in tqdm(CROP_EXP_FOLDERS, total=len(CROP_EXP_FOLDERS)):
    exp_folder, folds, crop_folder = CROP_EXP_FOLDERS[mode]
    print(f"- Model {mode} - {exp_folder}")

    config = Config(json.load(open(exp_folder + "config.json", "r")))

    if mode in ["crop", "crop_2"]:
        df_mode = df[df['orient'] == "Sagittal"].reset_index(drop=True)
        df_mode["side"] = "Center"
    elif "scs" in mode:
        df_mode = df[df['orient'] == "Sagittal"]
        df_mode = df_mode[df_mode["weighting"] == "T2"].reset_index(drop=True)
        df_mode["side"] = "Center"
    elif "nfn" in mode:
        df_mode = df[df['orient'] == "Sagittal"]
        df_mode["side"] = ["Right", "Left"]
        df_mode = df_mode.explode("side").reset_index(drop=True)
        df_mode = df_mode.sort_values(
            ["study_id", "series_id", "side", "level"],
            ascending=[True, True, False, True],
            ignore_index=True
        )
    elif "ss" in mode:
        df_mode = df[df['orient'] == "Axial"]
        df_mode["side"] = ["Right", "Left"]
        df_mode = df_mode.explode("side").reset_index(drop=True)
        df_mode = df_mode.sort_values(
            ["study_id", "series_id", "side", "level"],
            ascending=[True, True, False, True],
            ignore_index=True
        )

    df_mode['img_path'] = SAVE_FOLDER + crop_folder + "/" + df_mode["img_path_"]

    transfos = get_transfos(augment=False, resize=config.resize, crop=config.crop)
    dataset = CropDataset(
        df_mode,
        targets="target",
        transforms=transfos,
        frames_chanel=config.frames_chanel,
        n_frames=config.n_frames,
        stride=config.stride,
        train=False,
        # load_in_ram=False,
    )
    dataset = SafeDataset(dataset)

    model = define_model(
        config.name,
        drop_rate=config.drop_rate,
        drop_path_rate=config.drop_path_rate,
        pooling=config.pooling,
        head_3d=config.head_3d,
        n_frames=config.n_frames,
        num_classes=config.num_classes,
        num_classes_aux=config.num_classes_aux,
        n_channels=config.n_channels,
        reduce_stride=config.reduce_stride,
        pretrained=False,
    )
    model = model.cuda().eval()

    if mode == "crop_2":
        model.delta = 1

    preds = []
    for fold in folds:
        weights = exp_folder + f"{config.name}_{fold}.pt"
        model = load_model_weights(model, weights, verbose=1)

        pred, _ = predict(
            model,
            dataset,
            config.loss_config,
            batch_size=BATCH_SIZE,
            use_fp16=USE_FP16,
            num_workers=NUM_WORKERS,
        )
        preds.append(pred)

    preds = np.mean(preds, 0)

    if PLOT:
        df_ref = pd.read_csv(exp_folder + f"df_val_{FOLD}.csv").head(len(preds))
        # order_ref = df_ref.sort_values(["side", "level"]).index.values
        preds_ref = np.load(exp_folder + f"pred_inf_{FOLD}.npy")[: len(preds)]  # [order_ref]

        # plt.figure(figsize=(8, 4))
        # plt.subplot(1, 2, 1)
        # plt.plot(preds)
        # plt.subplot(1, 2, 2)
        # plt.plot(preds_ref)
        # plt.show()

        delta = (np.abs(preds - preds_ref)).max()
        print(preds.shape, preds_ref.shape)
        print(f"{mode} delta:", delta)

    idx = df_mode[["study_id", "series_id", "level", "side"]].values.astype(str).tolist()
    idx = ["_".join(i) for i in idx]
    crop_fts[mode] = dict(zip(idx, preds))

# ## Level 2


# csv_fts = {
#     "ch": sub_to_dict("submission.csv"),
#     "dh": sub_to_dict("submission.csv"),
# }

# csv_fts = {}
# for k in ['ch', 'dh']:
#     # config_2.exp_folders['dh'], config_2.exp_folders['ch']
#     file = torch.load(config_2.exp_folders[k])
#     csv_fts[k] = dict(zip(
#         file["study_id"].tolist(),
#         file['logits'].float().cpu().numpy(),
#     ))


DELTA_SCS = [0, 0, 0]

df_2 = df_meta[
    ["study_id", "series_id", "series_description"]
].groupby('study_id').agg(list).reset_index()

all_preds = []
for exp_folder in EXP_FOLDERS_2:
    config_2 = Config(json.load(open(exp_folder + "config.json", "r")))

    # LOCAL
    csv_fts = {}
    # for k in config_2.exp_folders:
    #     if "ch" in k or "dh" in k:
    #         file = torch.load(config_2.exp_folders[k])
    #         csv_fts[k] = dict(zip(
    #             file["study_id"].tolist(),
    #             file['logits'].float().cpu().numpy(),
    #         ))

    dataset = FeatureInfDataset(
        df_2,
        config_2.exp_folders,
        crop_fts,
        csv_fts,
        save_folder=SAVE_FOLDER,
    )
    dataset = SafeDataset(dataset)

    model = define_model_2(
        config_2.name,
        ft_dim=config_2.ft_dim,
        layer_dim=config_2.layer_dim,
        dense_dim=config_2.dense_dim,
        p=config_2.p,
        n_fts=config_2.n_fts,
        # resize=config_2.resize,
        num_classes=config_2.num_classes,
        num_classes_aux=config_2.num_classes_aux,
    )
    model = model.eval().cuda()

    for fold in FOLDS_2:
        weights = exp_folder + f"{config_2.name}_{fold}.pt"
        model = load_model_weights(model, weights, verbose=config_2.local_rank == 0)

        preds, _ = predict(
            model,
            dataset,
            {"activation": ""},
            batch_size=BATCH_SIZE_2,
            use_fp16=USE_FP16,
            num_workers=NUM_WORKERS,
        )

#         preds[:, :5, 0] += DELTA_SCS[0]
#         preds[:, :5, 1] += DELTA_SCS[1]
#         preds[:, :5, 2] += DELTA_SCS[2]

        preds = softmax(preds, axis=-1)

#         print(preds[:, :5, 2].mean())
#         print(preds[:, :5, 1].mean())

        if DEBUG and not EVAL:
            preds_ref = np.load(EXP_FOLDERS_2[0] + f"pred_val_{fold}.npy")[:1]
            delta = np.abs(preds - preds_ref).max()
            print(f"Model {exp_folder} delta:", delta)

        all_preds.append(preds)

preds = np.mean(all_preds, 0).astype(np.float64)
studies = df_2[["study_id"]].copy().astype(int)

rows = []
for i in range(len(studies)):
    for c, injury in enumerate(config_2.targets):
        rows.append(
            {
                "row_id": f'{studies["study_id"].values[i]}_{injury}',
                "normal_mild": preds[i, c, 0],
                "moderate": preds[i, c, 1],
                "severe": preds[i, c, 2],
            }
        )

sub = pd.DataFrame(rows)
sub.to_csv("submission.csv", index=False)
sub.tail(25)

if EVAL:
    y = pd.read_csv(ROOT_DATA_DIR + "train.csv")

    for c in y.columns[1:]:
        y[c] = y[c].map(dict(zip(SEVERITIES, [0, 1, 2]))).fillna(-1)
    y = y.astype(int)

    df_val = studies.copy().merge(y, how="left")

    avg_loss, losses = rsna_loss(df_val[config_2.targets].values, preds, verbose=1)

    for k, v in losses.items():
        print(f"- {k}_loss\t: {v:.3f}")

    print(f"\n -> CV Score : {avg_loss :.3f}")

# Done !