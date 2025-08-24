"""
Author: Kaiqing Lin
Date: 2024/6/23
File: Data_Prepare.py.py
"""
import os
import os.path as osp
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.io as sio
import imageio.v2 as imageio
import cv2
from termcolor import cprint
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from termcolor import cprint
import torchvision
from utils.DFD_DataSet.DFD_datasets import FFDataset, CDFDataset
# from utils.DFD_DataSet.DFD_datasets import CDFDataset
import random


def load_txt(txt_path, root_path, method, qp):
    with open(os.path.join(txt_path, 'train_ff.txt'), 'r') as f:
        train_videos = f.readlines()
        train_videos = [
            os.path.join(root_path, i.strip().replace('Deepfakes', method).replace('c23', qp))
            for i in train_videos]
    with open(os.path.join(txt_path, 'val_ff.txt'), 'r') as f:
        val_videos = f.readlines()
        val_videos = [os.path.join(root_path, i.strip().replace('Deepfakes', method).replace('c23', qp))
                      for i in val_videos]
    return train_videos, val_videos


def load_txt_CDF_Test(root_path, method, qp):
    # NOTE: Load the image and viedo list for CDF test!
    with open(os.path.join('/data2/linkaiqing/code/Reprogramming/dataset/Celeb_DF/List_of_testing_videos.txt'), 'r') as f:
        test_videos = f.readlines()
        test_videos = [os.path.join(root_path, i.strip().split(' ')[1]) for i in test_videos]
    return test_videos


def load_txt_Test(txt_path, root_path, method, qp):
    with open(os.path.join(txt_path, 'test_ff.txt'), 'r') as f:
        test_videos = f.readlines()
        test_videos = [
            os.path.join(root_path, i.strip().replace('Deepfakes', method).replace('c23', qp))
            for i in test_videos]
    return test_videos


def DataPrepare(dataset_name, dataset_dir, train_batch_size=64, test_batch_size=64, random_state=1,
                clip_transform=None, args=None):
    transform = clip_transform

    generator = torch.Generator().manual_seed(random_state)

    # NOTE: Your Dataset

    if dataset_name == 'Celeb_DF':
        root_path = '/data2/linkaiqing/code/Reprogramming/dataset/Celeb_DF'
        cprint(f"Celeb_DF Dir: {root_path}", 'yellow')
        test_viedos = load_txt_CDF_Test(root_path, 'Celeb_DF', 'c23')
        testset = CDFDataset(video_names=test_viedos, phase='test',
                             is_pillow=True, transform=transform, test_frame_nums=-1)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=test_batch_size, shuffle=False, num_workers=args.num_workers, generator=generator
        )
        return testloader
    else:
        txt_path = '/data2/linkaiqing/code/Reprogramming/VP_Deepfake/dataset_info/save_txt'
        # root_path = '/data2/linkaiqing/code/Reprogramming/dataset/FaceForensics_Fingerprints'
        root_path = dataset_dir
        method = dataset_name
        qp = args.quailty
        train_videos, val_videos, test_viedos = [], [], []
        for method in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']:
            train_videos_, val_videos_ = load_txt(txt_path=txt_path, root_path=root_path, method=method, qp=qp)
            test_viedos_ = load_txt_Test(txt_path=txt_path, root_path=root_path, method=method, qp=qp)
            train_videos += train_videos_
            val_videos += val_videos_
            test_viedos += test_viedos_
        
        trainset = FFDataset(video_names=train_videos, phase='train',
                             is_pillow=True, transform=transform)
        testset = FFDataset(video_names=val_videos, phase='valid', test_frame_nums=5,
                            is_pillow=True, transform=transform)
        test_ff_set = FFDataset(video_names=test_viedos, phase='test', test_frame_nums=20,
                                is_pillow=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch_size, shuffle=True, num_workers=args.num_workers, generator=generator
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=test_batch_size, shuffle=False, num_workers=args.num_workers, generator=generator
        )
        test_ff_loader = torch.utils.data.DataLoader(
            test_ff_set, batch_size=test_batch_size, shuffle=False, num_workers=args.num_workers, generator=generator
        )

        cprint(f"=================== Data Info ===================", 'blue')
        cprint(f"Dataset Name {dataset_name}", 'blue')
        cprint(f"Data Slice {txt_path}", 'blue')
        cprint(f"Data Root {root_path}", 'blue')
        cprint(f"Method {method}", 'blue')
        cprint(f"Quality {qp}", 'blue')

        print('All Train videos Number: %d' % (
                len(trainset.videos_by_class['real']) + len(testset.videos_by_class['fake'])))
        class_names = ['a real face', 'a fake face']
        cprint(f"================================================", 'blue')
    return trainloader, testloader, class_names, trainset, test_ff_loader


if __name__ == '__main__':
    pass
