"""
Author: Kaiqing Lin
Date: 2024/6/23
File: CLIP_Wrapper.py.py
"""
import os
import os.path as osp
import argparse
import sys
sys.path.append(osp.dirname(osp.abspath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.io as sio
import imageio.v2 as imageio
from termcolor import cprint
from tqdm import tqdm
from termcolor import cprint
import torchvision.transforms as transforms
from Programming import InputPadding


def L1_loss(input_, target_):
    # torch.mean(torch.abs(input_ - target_))
    return torch.mean(torch.abs(input_ - target_))


class Model_Wrapper(nn.Module):
    def __init__(self, model_name=None, input_perturbation=None, output_mapping=None,
                 train_resize=None, repro_img_size=128, device=None, img_full_size=224,
                 face_emb_random_proj=False):
        super(Model_Wrapper, self).__init__()

        self.model_name = model_name
        self.No_operation = nn.Identity()
        self.device = device
        self.repro_img_size = repro_img_size
        self.face_emb_random_proj = face_emb_random_proj

        # =========== Trainable Resize ===========
        if train_resize is None:
            self.train_resize = self.No_operation.to(device)
            self.no_trainable_resize = 1
        else:
            self.train_resize = train_resize.to(device)

        # =========== Load CLIP Model ===========
        if 1:
            if self.model_name == "clip" or self.model_name == "clip_ViT_B_32":
                import clip
                model, self.clip_preprocess = clip.load("ViT-B/32", device=device,
                                                        download_root='./CLIP_Pretrained/ViT-B_32')
                print("Load Pretrained!")
                # https://github.com/openai/CLIP/issues/57
                from Our_CLIP import our_clip
                clip_org_size = 224
                self.clip_org_size = clip_org_size
                model = our_clip(model, device=device,
                                 face_emb_random_proj=self.face_emb_random_proj,).to(device)
            elif self.model_name == 'clip_ViT_B_16':
                import clip
                model, self.clip_preprocess = clip.load("ViT-B/16", device=device,
                                                        download_root='./CLIP_Pretrained/ViT-B_16')
                print("Load Pretrained!")
                #
                from Our_CLIP import our_clip
                clip_org_size = 224
                self.clip_org_size = clip_org_size
                model = our_clip(model, device=device,
                                 face_emb_random_proj=self.face_emb_random_proj,
                                 face_out_dim=self._get_img_feature_dim(model=model)).to(device)
            elif self.model_name == "clip_large":
                import clip
                model, self.clip_preprocess = clip.load("ViT-L/14", device=device,
                                                        download_root='./CLIP_Pretrained/ViT-L_14')
                print("Load Pretrained!")
                # https://github.com/openai/CLIP/issues/57
                from Our_CLIP import our_clip
                clip_org_size = 224
                self.clip_org_size = clip_org_size
                model = our_clip(model, device=device,
                                 face_emb_random_proj=self.face_emb_random_proj,
                                 face_out_dim=self._get_img_feature_dim(model=model)).to(device)
            elif self.model_name == 'clip_large_336':
                import clip
                model, self.clip_preprocess = clip.load("ViT-L/14@336px", device=device,
                                                        download_root='./CLIP_Pretrained/ViT_L_14@336')
                tokenizer = None
                print("Load Pretrained!")
                from Our_CLIP import our_clip
                clip_org_size = 336
                self.clip_org_size = clip_org_size
                model = our_clip(model, device=device, clip_tokenizer=tokenizer,
                                 face_emb_random_proj=self.face_emb_random_proj,
                                 face_out_dim=self._get_img_feature_dim(model=model)).to(device)
            elif self.model_name == 'clip_ViT_B_16_openclip':
                import open_clip
                model, self.clip_preprocess = open_clip.create_model_from_pretrained(
                    model_name='ViT-B-16',
                    pretrained='openai',
                    device=device,
                    cache_dir=f'./CLIP_Pretrained/ViT-B-16_openai'
                )
                tokenizer = open_clip.get_tokenizer('ViT-B-16')
                print("Load Pretrained!")
                from Our_CLIP import our_clip
                clip_org_size = 224
                self.clip_org_size = clip_org_size
                model = our_clip(model, device=device, clip_tokenizer=tokenizer,
                                 face_emb_random_proj=self.face_emb_random_proj,
                                 face_out_dim=self._get_img_feature_dim(model=model), open_clip=True).to(device)
            elif self.model_name == 'clip_large_openclip':
                import open_clip
                model, self.clip_preprocess = open_clip.create_model_from_pretrained(
                    model_name='ViT-L-14',
                    pretrained='openai',
                    device=device,
                    cache_dir=f'./CLIP_Pretrained/ViT-L-14_openai'
                )
                tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
                                
                print("Load Pretrained!")
                from Our_CLIP import our_clip
                clip_org_size = 224
                self.clip_org_size = clip_org_size
                model = our_clip(model, device=device, clip_tokenizer=tokenizer,
                                 face_emb_random_proj=self.face_emb_random_proj,
                                 face_out_dim=self._get_img_feature_dim(model=model), open_clip=True).to(device)
            elif self.model_name == 'clip_large_336_openclip':
                import open_clip
                model, self.clip_preprocess = open_clip.create_model_from_pretrained(
                    model_name='ViT-L-14-336',
                    pretrained='openai',
                    device=device,
                    cache_dir=f'./CLIP_Pretrained/ViT-L-14-336_openai'
                )
                tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
                                
                print("Load Pretrained!")
                from Our_CLIP import our_clip
                clip_org_size = 336
                self.clip_org_size = clip_org_size
                model = our_clip(model, device=device, clip_tokenizer=tokenizer,
                                 face_emb_random_proj=self.face_emb_random_proj,
                                 face_out_dim=self._get_img_feature_dim(model=model), open_clip=True).to(device)
            
            assert clip_org_size == img_full_size, f"Clip model size {clip_org_size} is not equal to the target size {img_full_size}"

            for p in model.parameters():
                p.data = p.data.float()
                if p.grad:
                    p.grad.data = p.grad.data.float()
            model.__init_text_template__()  # Due to the parameter dtype is float16 for CLIP-ViT-B-14 and some other models, we need to convert it to float32. Therefore, it is necessary to reinitialize the text template.

            model.model.requires_grad_(False)
            self.clip_rz_transform = transforms.Resize([repro_img_size, repro_img_size])

        self.clip_org_size = clip_org_size
        # =========== Reprogramming (Input Perturbation) ===========
        if 1:
            if input_perturbation is None:
                self.input_perturbation = self.No_operation.to(device)
            else:
                self.input_perturbation = input_perturbation.to(device)
        # =========== Output Mapping ===========
        self.output_mapping = self.No_operation.to(device)

        # Set to evaluation mode
        self.model = model
        self.model.eval()
        self.model.to(device)
        self.input_perturbation.to(device)
        self.output_mapping.to(device)

    def get_transform(self):
        # Return the transform of pretrained model
        for _, t in enumerate(self.clip_preprocess.transforms):
            if isinstance(t, transforms.Resize):
                self.clip_preprocess.transforms[_] = transforms.Resize([self.repro_img_size, self.repro_img_size])
            if isinstance(t, transforms.CenterCrop):
                self.clip_preprocess.transforms[_] = transforms.CenterCrop([self.repro_img_size, self.repro_img_size])
        return self.clip_preprocess

    def _get_img_feature_dim(self, model=None):
        x = torch.randn(1, 3, self.clip_org_size, self.clip_org_size).to(self.device)
        if model is not None:
            x_emb = model.encode_image(x)
        else:
            x_emb = self.model.encode_image(x)
        return x_emb.shape[1]

    def CLIP_network(self, x, face_emb):
        x_emb = self.model.encode_image(x)
        x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)

        # === new ===
        x_emb = x_emb.unsqueeze(dim=1)
        txt_emb = self.model.encode_text(face_embeddings=face_emb)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb.permute(-3, -1, -2)

        logits = self.model.logit_scale.exp() * x_emb @ txt_emb
        logits = logits.squeeze(dim=1)
        # === new ===

        # logits = self.model.logit_scale.exp() * x_emb @ self.txt_emb.t()
        return logits

    def CLIP_feature_extract(self, x, face_emb):
        x = x.to(self.device)
        img_h = -1
        img_w = -1

        x = self.input_perturbation(x, img_h, img_w)

        img_emb = self.model.encode_image(x)
        txt_emb = self.model.encode_text(face_emb)

        return img_emb, txt_emb

    def CLIP_feature_extract_wo_Prompt(self, x, face_emb):
        x = x.to(self.device)
        img_h = -1
        img_w = -1

        x = self.input_perturbation(x, img_h, img_w, zero_padding=True)

        img_emb = self.model.encode_image(x)
        txt_emb = self.model.encode_text(face_emb)

        return img_emb, txt_emb

    def forward(self, input_img, face_emb):
        input_img = input_img.to(self.device)
        face_emb = face_emb.to(self.device)
        x = input_img
        img_h = -1
        img_w = -1
        if self.no_trainable_resize == 0:
            x, img_h, img_w = self.train_resize(x)
        else:
            x = self.train_resize(x)

        x = self.input_perturbation(x, img_h, img_w)
        face_emb = face_emb.to(self.device)
        x = self.CLIP_network(x, face_emb)
        x = self.output_mapping(x)

        return x


def Load_Model(dataset, device, img_resize, img_full_size, file_path=None,
               pretrained_model="resnet18", args=None):
    cprint("==== Load Visual Reprogramming Model ====", 'red')
    # =========== Input Padding (Visual Prompt)  ===========
    channel = 3
    normalization = None
    padding_size = None

    input_pad = InputPadding(img_size=(channel, img_resize, img_resize), output_size=(3, img_full_size, img_full_size),
                             normalization=normalization, input_aware=False,
                             padding_size=padding_size, model_name=pretrained_model,
                             device=device)
    
    cprint(f"Input Padding Layer Transform Image to Size {img_full_size}", 'red')

    # # =========== Output Padding  ===========
    out_map = None

    # =========== Reprogramming Model  ===========
    
    reprogram_model = Model_Wrapper(model_name=pretrained_model, input_perturbation=input_pad,
                                        output_mapping=out_map, train_resize=None, repro_img_size=img_resize,
                                        device=device,
                                        img_full_size=img_full_size,
                                        face_emb_random_proj=args.face_emb_random_proj)
        
    # If file_path is not None, load the pretrained model from the file_path
    if file_path is not None:            
        state_dict = torch.load(file_path, map_location=device)
        cprint(f"Load the model from {file_path}", 'red')

        # Load the input padding layer
        reprogram_model.input_perturbation.load_state_dict(state_dict["perturb_dict"])

        # Load the random projection layer
        reprogram_model.model.random_proj_layer.load_state_dict(state_dict["face_random_proj_dict"])
        
    return reprogram_model


if __name__ == '__main__':
    img_resize = 180
    img_full_size = 224
    device = 'cuda:1'
    net = Load_Model(dataset='', device=device, pretrained_model='clip_large', img_resize=img_resize,
                     img_full_size=img_full_size).to(device)

    input_img = torch.randn(32, 3, img_resize, img_resize).to(device)
    face_emb = torch.randn(32, 512).to(device)

    output = net(input_img, face_emb)

    print(1)
