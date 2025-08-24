"""
@File    :   Face_Emb_CLIP.py
@Author  :   Kaiqing.Lin
@Update  :   2024/05/24
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
from clip.model import CLIP
import clip
from open_clip.transformer import text_global_pool


class our_clip(nn.Module):
    def __init__(self, model, device, clip_tokenizer=None, face_emb_random_proj=False, face_out_dim=512, open_clip=False):
        super().__init__()
        self.device = device
        self.model = model.to(self.device)      # CLIP model
        self.logit_scale = self.model.logit_scale       # logit scale
        self.model.encode_text = self.encode_text
        
        self.open_clip = open_clip  # whether to use the lib "open_clip"

        if clip_tokenizer is None:
            self.clip_tokenizer = clip.tokenize
            cprint("The clip tokenizer has been initialized as 'clip.tokenize()'", "yellow")
        else:
            self.clip_tokenizer = clip_tokenizer
            cprint(f"The clip tokenizer is '{type(self.clip_tokenizer)}'", "yellow")
        
        self.face_emb_random_proj = face_emb_random_proj  # The random projection for face embeddings
        random_proj_layer = torch.randn(face_out_dim, 512).to(self.device)
        nn.init.normal_(random_proj_layer, mean=0, std=1/face_out_dim)
        self.random_proj_layer = nn.Linear(512, face_out_dim, bias=False).to(self.device)
        self.random_proj_layer.weight.data = random_proj_layer
        self.random_proj_layer.weight.requires_grad = False
            
        if hasattr(self.model, 'dtype') is False:
            try:
                # when the model is not from TimmModel
                self.model.dtype = self.model.visual.conv1.weight.dtype
            except:
                # when the model is from TimmModel
                self.model.dtype = self.model.visual.head.proj.weight.dtype

        print(f"self.model.dtype: {self.model.dtype}")
        self.__init_text_template__()       # Initialize the text template

    @torch.no_grad()
    def __init_text_template__(self):
        """
        Set the text template for real and fake categories
        """

        self.face_id_model = None  # face id model

        real_template = 'A real photo of a person'
        fake_template = 'A fake photo of a FE person'
        self.real_ids = self.clip_tokenizer(real_template).to(self.device)
        self.fake_ids = self.clip_tokenizer(fake_template).to(self.device)
        self.FE_ids = self.clip_tokenizer("FE").to(self.device)[0, 1].item()
        self.fake_word = self.clip_tokenizer("fake").to(self.device)[0, 1].item()
        self.real_word = self.clip_tokenizer("real").to(self.device)[0, 1].item()
        self.real_emb, self.fake_emb = self.__init_encode_text__()
        cprint("The text template has been initialized", "yellow")

    @torch.no_grad()
    def __init_encode_text__(self):
        """
        Encode the text template (from texts to embeddings)
        """
        real_text_embeddings = self.model.token_embedding(self.real_ids).to(self.model.dtype).to(self.device)
        real_text_embeddings[self.real_ids == self.FE_ids] = 0
        fake_text_embeddings = self.model.token_embedding(self.fake_ids).to(self.model.dtype).to(self.device)
        fake_text_embeddings[self.fake_ids == self.FE_ids] = 0
        return real_text_embeddings, fake_text_embeddings

    def encode_text(self, face_embeddings: torch.Tensor):
        # Get the face embeddings
        if self.face_emb_random_proj:
            face_embeddings = self.random_proj_layer(face_embeddings)

        face_embeddings = F.pad(face_embeddings, (0, self.real_emb.shape[-1] - face_embeddings.shape[-1]),
                                value=0)
        # Insert the face embeddings into the text template
        real_emb = self.real_emb.clone().detach().repeat(len(face_embeddings), 1, 1).to(self.device)
        fake_emb = self.fake_emb.clone().detach().repeat(len(face_embeddings), 1, 1).to(self.device)
        real_ids = self.real_ids.clone().detach().repeat(len(face_embeddings), 1).to(self.device)
        fake_ids = self.fake_ids.clone().detach().repeat(len(face_embeddings), 1).to(self.device)

        # real_emb[real_ids == self.FE_ids, :face_embeddings.shape[-1]] = face_embeddings.to(self.model.dtype)
        fake_emb[fake_ids == self.FE_ids, :face_embeddings.shape[-1]] = face_embeddings.to(self.model.dtype).squeeze(dim=1)
        
        # Get the final embeddings
        if self.open_clip is False:
            # The procedure for the lib "clip"
            real_x = real_emb + self.model.positional_embedding.type(self.model.dtype)
            fake_x = fake_emb + self.model.positional_embedding.type(self.model.dtype)

            real_x = real_x.permute(1, 0, 2)  # NLD -> LND
            fake_x = fake_x.permute(1, 0, 2)

            real_x = self.model.transformer(real_x)
            fake_x = self.model.transformer(fake_x)

            real_x = real_x.permute(1, 0, 2)  # LND -> NLD
            fake_x = fake_x.permute(1, 0, 2)

            real_x = self.model.ln_final(real_x).type(self.model.dtype)
            fake_x = self.model.ln_final(fake_x).type(self.model.dtype)

            real_x = real_x[torch.arange(real_x.shape[0]), self.real_ids.argmax(dim=-1)] @ self.model.text_projection
            fake_x = fake_x[torch.arange(fake_x.shape[0]), self.fake_ids.argmax(dim=-1)] @ self.model.text_projection

            out_emb = torch.cat([real_x.unsqueeze(dim=1), fake_x.unsqueeze(dim=1)], dim=1)
        else:
            # The procedure for the lib "open_clip"
            real_x = real_emb + self.model.positional_embedding.type(self.model.dtype)
            fake_x = fake_emb + self.model.positional_embedding.type(self.model.dtype)
            
            def text2feat(attn_mask, text, model, x, text_projection):
                x = model.transformer(x, attn_mask=attn_mask)
                x = model.ln_final(x)
                x, _ = text_global_pool(x, text, model.text_pool_type)            
                if text_projection is not None:
                    if isinstance(text_projection, nn.Linear):
                        x = text_projection(x)
                    else:
                        x = x @ text_projection
                return x
            real_x = text2feat(self.model.attn_mask, real_ids, self.model, real_x, self.model.text_projection)
            fake_x = text2feat(self.model.attn_mask, fake_ids, self.model, fake_x, self.model.text_projection)
            out_emb = torch.cat([real_x.unsqueeze(dim=1), fake_x.unsqueeze(dim=1)], dim=1)
            
        return out_emb

    def encode_image(self, image):
        return self.model.encode_image(image)

    def CLIP_network(self, x, face_emb):
        x_emb = self.encode_image(x)
        x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)

        # === new ===
        x_emb = x_emb.unsqueeze(dim=1)
        # txt_emb = self.encode_multi_text(face_emb)
        txt_emb = self.encode_text(face_emb)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb.permute(-3, -1, -2)
        logits = self.model.logit_scale.exp() * x_emb @ txt_emb
        logits = logits.squeeze(dim=1)
        # === new ===

        # logits = self.model.logit_scale.exp() * x_emb @ self.txt_emb.t()
        return logits

    def forward(self, image, text):
        # return self.logit_output(image, text)
        return self.CLIP_network(image, text)
    