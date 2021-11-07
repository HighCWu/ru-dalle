# -*- coding: utf-8 -*-
from math import sqrt, log

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from taming.modules.diffusionmodules.model import Encoder, Decoder


class VQGanGumbelVAE(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        model = GumbelVQ(
            ddconfig=config.model.params.ddconfig,
            n_embed=config.model.params.n_embed,
            embed_dim=config.model.params.embed_dim,
            kl_weight=config.model.params.kl_weight,
        )
        self.model = model
        self.num_layers = int(log(config.model.params.ddconfig.attn_resolutions[0]) / log(2))
        self.image_size = 256
        self.num_tokens = config.model.params.n_embed

    @torch.no_grad()
    def _get_codebook_indices(self, img):
        img = (2 * img) - 1
        _, _, [_, _, indices] = self.model.encode(img)

        # rearrange(indices, 'b h w -> b (h w)')
        return indices.reshape(indices.shape[0], -1), indices.shape[1], indices.shape[2]

    @torch.no_grad()
    def get_codebook_indices(self, img):
        if hasattr(self, 'onnx_model'):
            return self.onnx_get_codebook_indices(img)[0]
        return self._get_codebook_indices(img)[0]

    def _decode(self, img_seq, height, width):
        b = img_seq.shape[0]
        one_hot_indices = torch.nn.functional.one_hot(img_seq, num_classes=self.num_tokens).float()
        z = (one_hot_indices @ self.model.quantize.embed.weight)
        z = z.reshape(b, height, width, -1).permute(0,3,1,2) # rearrange(z, 'b (h w) c -> b c h w', h=height))
        img = self.model.decode(z)
        img = (img.clamp(-1., 1.) + 1) * 0.5
        return img

    def decode(self, img_seq):
        _, n = img_seq.shape
        h, w = int(sqrt(n)), int(sqrt(n))
        if hasattr(self, 'onnx_model'):
            return self.onnx_decode(img_seq, h, w)
        return self._decode(img_seq, h, w)

    def set_onnx(self, onnx_dir):
        device = next(self.parameters()).device.type
        from rudalle.onnx.vae import VQGanGumbelVAEONNX
        self.onnx_model = VQGanGumbelVAEONNX(onnx_dir, device)

    def onnx_get_codebook_indices(self, img):
        device = next(self.parameters()).device
        img_seq, height, width = self.onnx_model.get_codebook_indices(img.cpu().detach().numpy())
        img_seq = torch.from_numpy(img_seq).long().to(device)
        return img_seq, height, width
        
    def onnx_decode(self, img_seq, height, width):
        device = next(self.parameters()).device
        img = self.onnx_model.decode(img_seq.cpu().detach().numpy(), height, width)
        img = torch.from_numpy(img).float().to(device)
        return img


class GumbelQuantize(nn.Module):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """

    def __init__(self, num_hiddens, embedding_dim, n_embed, straight_through=True,
                 kl_weight=5e-4, temp_init=1.0, use_vqinterface=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embed = n_embed
        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight
        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(self.n_embed, self.embedding_dim)
        self.use_vqinterface = use_vqinterface

    def forward(self, z, temp=None, return_logits=False):
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp
        logits = self.proj(z)
        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)
        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()
        ind = soft_one_hot.argmax(dim=1)
        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), logits
            return z_q, diff, (None, None, ind)
        return z_q, diff, ind


class GumbelVQ(nn.Module):

    def __init__(self, ddconfig, n_embed, embed_dim, kl_weight=1e-8):
        super().__init__()
        z_channels = ddconfig['z_channels']
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = GumbelQuantize(z_channels, embed_dim, n_embed=n_embed, kl_weight=kl_weight, temp_init=1.0)
        self.quant_conv = torch.nn.Conv2d(ddconfig['z_channels'], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig['z_channels'], 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
