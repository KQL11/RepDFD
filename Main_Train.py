"""
Author: Kaiqing Lin
Date: 2024/6/23
File: Main_Ours.py.py
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
from termcolor import cprint
from tqdm import tqdm
import time
import random
import warnings
from torch.cuda.amp import autocast, GradScaler
from Models.CLIP_Wrapper import Load_Model
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
from termcolor import cprint


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def args_get():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:0', type=str)
    p.add_argument('--dataset', default='Deepfakes')
    p.add_argument('--datapath', default='/data2/linkaiqing/code/Reprogramming/dataset/FaceForensics_Fingerprints', type=str)

    p.add_argument('--quailty', type=str, default='c23', choices=['c23', 'c40'])
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--pretrained',
                   choices=["clip", "clip_large", "clip_ViT_B_32", "clip_ViT_B_16", 'clip_large_336',
                            'clip_ViT_B_16_openclip', 'clip_large_openclip', 'clip_large_336_openclip'],
                   default="clip_large_openclip")
    
    p.add_argument('--face_emb_random_proj', default=True, action='store_true')
    p.add_argument('--mapping_method',
                   choices=["fully_connected_layer_mapping", "frequency_based_mapping", "self_definded_mapping",
                            "semantic_mapping", "no_mapping", 'backbone_linear_mapping'], default="no_mapping")
    p.add_argument('--img_resize', type=int, default=None)
    
    # NOTE: The Image Size Ratio after Resize
    p.add_argument('--img_temp_ratio', type=float, default=0.7,
                   choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # NOTE: The Image Input Size of the pretrained model
    p.add_argument('--img_full_size', type=int, default=224)
    p.add_argument('--train_resize', type=int, default=0)  # 1, 04
    p.add_argument('--freqmap_interval', type=int, default=-1)  # -1 or 1,2,3..

    p.add_argument('--epoch', type=int, default=100)
    p.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'])
    p.add_argument('--lr', type=float, default=1)
    p.add_argument('--L2', type=float, default=2e-4)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--random_state', type=int, default=7)
    
    # Some tries to improve the performance
    args = p.parse_args()
    return args


def get_auc(logit_list: np.ndarray, label_list: np.ndarray):
    """
        logit_list: [logit_0, logit_1]
        label_list: [label_0, label_1]

        return auc score
    """
    # AUC Score
    auc = roc_auc_score(y_true=label_list, y_score=logit_list)

    return auc


def CLIP_Training(save_ckpt_dir, dataset, fname, model, trainloader, testloader, Epoch, lr, weight_decay,
                  device, freqmap_interval=None, args=None, TestLoader_IID=None, TestLoader=None):
    # Prepare text embedding
    template_number = 0  # use default template
    # loss
    criterion = nn.CrossEntropyLoss()

    # Get parameters whose requires_grad is True
    name_list = []
    input_perturbation_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            name_list.append(name)
            if "input_perturbation" in name:
                input_perturbation_params.append(param)
    
    cprint("Visual Prompt Fine Tune", 'red')
    train_params = [
            {'params': input_perturbation_params, 'lr': lr, 'weight_decay': weight_decay},
            ]

    # Optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.AdamW(train_params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum)
    t_max = Epoch * len(trainloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    print("Params to learn:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("\t", name)

    # Convergence loss
    loss2 = torch.zeros([]).to(device)

    f = open(fname, "a")
    best_result = [-1, 0., 0., 1.]  # epoch, traing acc, validation acc, resize scale
    best_loss = 99999
    best_auc = 0
    total_train_acc = 0
    total_valid_acc = 0
    scale_grad = []

    scaler = GradScaler()
    for epoch in range(Epoch):
        # Training
        if 1:
            model.train()
            model.model.eval()
            train_loss = []
            train_loss2 = []
            train_accs = []
            bar_format = '{l_bar}{bar:5}{r_bar}{bar:-10b}'
            pbar = tqdm(trainloader, total=len(trainloader),
                        desc=f"Epoch {epoch + 1}, Lr {optimizer.param_groups[0]['lr']:.1e}", bar_format=bar_format, ncols=100)
            for pb in pbar:
                imgs, labels, face_emb = pb

                if imgs.get_device() == -1:
                    imgs = imgs.to(device)
                    labels = labels.to(device)

                optimizer.zero_grad()
                with autocast():
                    logits = model(imgs, face_emb)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                # loss.backward()

                # clip scale's gradient
                if model.no_trainable_resize == 0:
                    nn.utils.clip_grad_value_(model.train_resize.scale, 0.001)

                scaler.step(optimizer)
                scaler.update()
                # optimizer.step()

                model.model.logit_scale.data = torch.clamp(model.model.logit_scale.data, 0,
                                                           4.6052)  # Clamps all elements in input into the range in CLIP model

                if model.no_trainable_resize == 0:
                    with torch.no_grad():
                        model.train_resize.scale = model.train_resize.scale.clamp_(0.1, 5.0)

                acc = (logits.argmax(dim=-1) == labels).float().mean()
                train_loss.append(loss.item())
                train_loss2.append(loss2.item())
                train_accs.append(acc)

                total_train_loss = sum(train_loss) / len(train_loss)
                total_train_acc = sum(train_accs) / len(train_accs)
                if model.no_trainable_resize == 0:
                    pbar.set_postfix_str(
                        f"ACC: {total_train_acc * 100:.2f}%, Loss: {total_train_loss:.4f}, Scale: {model.train_resize.scale.item():.4f}")
                else:
                    pbar.set_postfix_str(
                        f"ACC: {total_train_acc * 100:.2f}%, Loss: {total_train_loss:.4f}")
                scheduler.step()

        f.write(
                f"Epoch {epoch + 1} Training Lr {optimizer.param_groups[0]['lr']:.1e}, ACC: {total_train_acc * 100:.2f}%, Loss: {total_train_loss:.4f}\n")

        if epoch % 5 == 0 or epoch == Epoch - 1:
            # Validation

            model.eval()
            valid_loss = []
            valid_accs = []
            pbar = tqdm(testloader, total=len(
                testloader), desc=f"Epoch {epoch + 1} Testing", ncols=80)
            y_true = np.zeros((TestLoader_IID.dataset.__len__(),))
            y_score = np.zeros((TestLoader_IID.dataset.__len__(),))
            counter_ = 0
            for pb in pbar:
                imgs, labels, face_emb = pb

                if imgs.get_device() == -1:
                    imgs = imgs.to(device)
                    labels = labels.to(device)

                with torch.no_grad():
                    logits = model(imgs, face_emb)
                    loss = criterion(logits, labels)
                    prob_ = nn.Softmax(dim=1)(logits)
                    for i in range(prob_.shape[0]):
                        y_true[counter_] = labels[i].cpu().numpy()
                        y_score[counter_] = prob_[i, 1].cpu().numpy()
                        counter_ += 1

                acc = (logits.argmax(dim=-1) == labels).float().mean()
                valid_loss.append(loss.item())
                valid_accs.append(acc)

                total_valid_loss = sum(valid_loss) / len(valid_loss)
                total_valid_acc = sum(valid_accs) / len(valid_accs)
                pbar.set_postfix_str(f"ACC: {total_valid_acc * 100:.2f}%, Loss: {total_valid_loss:.4f}")

            auc = get_auc(logit_list=y_score, label_list=y_true)
            cprint(
                f"ACC: {total_valid_acc * 100:.2f}%, Loss: {total_valid_loss:.4f}, AUC: {auc * 100:.2f}%", 'green')

            # update log
            f.write(f"Epoch {epoch + 1} Testing, ACC: {total_valid_acc * 100:.2f}%, Loss: {total_valid_loss:.4f}, AUC: {auc * 100:.2f}%\n")

            # if (total_valid_acc > best_result[2] or epoch == Epoch - 1):
            if best_loss > total_valid_loss or epoch == Epoch - 1:
                # save model
                print("Save! Acc: ", total_valid_acc, ", Loss: ", total_valid_loss)
                f.write(f"Save! Acc: {total_valid_acc}, Loss: {total_valid_loss}\n")
                state_dict = {
                        "pretrained_model": model.model_name,
                        "resize_dict": model.train_resize.state_dict(),
                        "perturb_dict": model.input_perturbation.state_dict(),
                        "face_random_proj_dict": model.model.random_proj_layer.state_dict(),
                    }

                if best_loss > total_valid_loss:
                    best_result = [epoch, total_train_acc, total_valid_acc]
                    best_loss = total_valid_loss
                    best_auc = auc
                    torch.save(state_dict, os.path.join(save_ckpt_dir, str(dataset) + "_best.pth"))
                
    if args is not None:
        f.write("\nSetting:\n")
        f.write("Pretrained Model: " + str(args.pretrained) + "\n")
        f.write("Lr: " + str(args.lr) + "\n")
        f.write("Epoch" + str(args.epoch) + "\n")
        f.write("Out Mapping" + str(args.mapping_method) + "\n")
        f.write("Image Size: " + str(args.img_resize) + "\n")
        f.write("Train Dataset: " + str(args.dataset) + "\n")
        f.write("Quailty: " + str(args.quailty) + "\n")

    f.close()

    return best_result


def main(args):
    start_time = time.time()

    if args.img_resize is None and args.img_full_size is not None:
        args.img_resize = int(args.img_full_size * args.img_temp_ratio)

    print(args.datapath)

    if args.train_resize > 0:
        args.set_train_resize = True
    else:
        args.set_train_resize = False

    # set random seed
    set_seed(args.random_state)

    device = f'{args.device}'
    if 'cuda' not in device:
        args.device = f'cuda:{device}'
    print("device: ", args.device)

    reprogram_model = Load_Model(dataset=args.dataset, device=args.device, img_resize=args.img_resize,
                                 img_full_size=args.img_full_size, file_path=None,
                                 pretrained_model=args.pretrained, args=args)
    clip_transform = reprogram_model.get_transform()

    # ================ Data Prepare ================
    from Models.Data_Prepare import DataPrepare
    trainloader, testloader, class_names, trainset, test_ff_loader = \
        DataPrepare(dataset_name=args.dataset, dataset_dir=args.datapath,
                    train_batch_size=32, test_batch_size=128,
                    random_state=args.random_state, clip_transform=clip_transform, args=args)   # train_batch = 32, test_batch = 128

    CDF_test_loader = DataPrepare(dataset_name='Celeb_DF', dataset_dir=args.datapath,
                                  train_batch_size=32, test_batch_size=128,
                                  random_state=args.random_state, clip_transform=clip_transform, args=args)

    # ================ Info Setting ================
    crop_flag = 'nocrop'
    
    if args.face_emb_random_proj:
        face_trans_flag = '_face_random_proj'
    else:
        face_trans_flag = ''

    repro_size = args.img_full_size - args.img_resize

    # SAVE DIR Definition
    save_ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"checkpoints_{args.optimizer}_Face_Emb_All_Image",
                                     f"img_full_size_{args.img_full_size}", f"epoch_{args.epoch}", args.pretrained + face_trans_flag,
                                     args.dataset + f'_{crop_flag}_' + 'quailty_' + args.quailty,
                                     f'reproSize_{repro_size}_lr_{args.lr}_L2_{args.L2}_momentum_{args.momentum}')
    if not os.path.exists(save_ckpt_dir):
        os.makedirs(save_ckpt_dir)

    fname = os.path.join(save_ckpt_dir, f'{args.dataset}_log.txt')
    cprint(f"Save ckpt dir: {save_ckpt_dir}", 'yellow')

    CLIP_Training(save_ckpt_dir=save_ckpt_dir, dataset=args.dataset, fname=fname, model=reprogram_model, trainloader=trainloader,
                testloader=testloader, Epoch=args.epoch, lr=args.lr, weight_decay=args.L2,
                device=args.device, freqmap_interval=None, args=args, TestLoader_IID=test_ff_loader, TestLoader=CDF_test_loader)

    f = open(fname, "a")
    f.write(f"Total Exection Time (second) : %s" % (time.time() - start_time))
    f.close()


if __name__ == '__main__':
    args = args_get()
    main(args)
