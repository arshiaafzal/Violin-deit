# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from curves import compute_curve_order, coords_to_index, index_to_coords_indexes
import numpy as np

def Casual_Mask_Decay(a_i , L):
    idx = torch.arange(L,device=a_i.device)
    I, J = torch.meshgrid(idx, idx, indexing='ij')
    E = (torch.abs((I-J)).float().view(1,1,L,L))
    M = torch.sigmoid(a_i).view(1,-1,1,1)**E
    return M

def Casual_Mask_Decay_Fixed(a_i , L):
    idx = torch.arange(L,device=a_i.device)
    I, J = torch.meshgrid(idx, idx, indexing='ij')
    E = (torch.abs((I-J)).float().view(1,1,L,L))
    M = (a_i).view(1,-1,1,1)**E
    return M

def max_min(x):
    print(torch.max(x).item(), torch.min(x).item())

class Violin_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., 
                 pos_emb = True, cls_tok = True, curve_list = ['s', 'sr', 'h', 'hr', 'm', 'mr','z', 'zr'],num_patches=196, qk_norm=False,mask='learned',scale=False,method='mul_v1',mask_sum='weighted'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.cls_tok = cls_tok
        self.curve_list = curve_list
        self.qk_norm = qk_norm
        self.mask = mask
        self.mask_sum = mask_sum
        self.scale = scale
        self.method = method

        self.curve_indices_inv = []

        N = num_patches 
        order = torch.range(0,N-1)
        S = int(np.sqrt(N))
        grid = order.view(S,S).clone()

        for curve in curve_list:
            if curve not in ['s', 'sr', 'h', 'hr', 'm', 'mr', 'z', 'zr']:
                raise ValueError("Invalid value for curve. Allowed values are: 's', 'sr', 'h', 'hr', 'm', 'mr', 'z', 'zr'.")
            curve_coords = compute_curve_order(grid, curve)
            self.curve_indices_inv.append(torch.tensor(index_to_coords_indexes(curve_coords, S,S)  , dtype=torch.long ))  
        
        self.num_curves = len(self.curve_indices_inv)

        if mask == 'fixed':
            self.register_buffer("ai_list", torch.stack([torch.ones(num_heads) * 0.996 for _ in range(self.num_curves )]))
        else:
            self.ai_list = nn.ParameterList([nn.Parameter(torch.empty(num_heads)) for _ in range(self.num_curves )])

        if qk_norm:
            self.q_norm = nn.LayerNorm(head_dim)
            self.k_norm = nn.LayerNorm(head_dim)

        if mask_sum == 'plain':
            self.mask_weights = torch.ones(self.num_curves ) / self.num_curves 
        elif mask_sum == 'linweight':
            self.mask_weights = nn.Linear(dim, self.num_curves * num_heads, bias=qkv_bias)
        else:
            self.mask_weights = nn.Parameter(torch.empty(self.num_curves))

        if scale:
            self.normalize = nn.Parameter(torch.empty(num_heads))
        else:
            self.normalize = torch.ones(num_heads)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        if self.qk_norm:
            q, k, v = self.q_norm(qkv[0]), self.k_norm(qkv[1]), qkv[2]   
        else:
            q, k, v = qkv[0], qkv[1], qkv[2]   
        
        attn = (q @ k.transpose(-2, -1)) * self.scale

        H = self.num_heads
        num_curves = self.num_curves
        dev = x.device
        M = torch.zeros(1,H,N,N,device=dev)

        if self.mask_sum == 'linweight':
            z = self.mask_weights(x).reshape(B, N, self.num_heads, num_curves)
            z = torch.mean(z, dim=1)
            mask_weights = torch.softmax(z, dim=-1).permute(0,2,1)
            M = torch.zeros(B,H,N,N,device=dev)
        elif self.mask_sum == 'softmax':
            mask_weights = nn.functional.softmax(self.mask_weights)
        else:
            mask_weights = self.mask_weights

        for c in range(num_curves):
            if self.mask == 'fixed':
                M_c = Casual_Mask_Decay_Fixed(self.ai_list[c].to(dev), attn.shape[-1])
            else:
                M_c = Casual_Mask_Decay(self.ai_list[c].to(dev), attn.shape[-1])
            ind = self.curve_indices_inv[c]
            M_c = M_c[:,:,ind][...,ind]
            if self.cls_tok:
                M_c = torch.cat((torch.ones((1,H,1,N), device=dev),torch.cat((torch.ones((1,H,N-1,1), device=dev),M_c),dim=-1)),dim=-2)
            if self.mask_sum == 'linweight':
                w = mask_weights[:,c]
                M += w.view(B,-1,1,1) * M_c 
            else:
                w = mask_weights[c]
                M += w * M_c 


        if self.method == 'mul_v1':        
            attn = attn * M * self.normalize.view(1,-1,1,1)
        elif self.method == 'mul_v2':  
            attn = attn * (1 + M * self.normalize.view(1,-1,1,1))
        elif self.method == 'add_v1':
            attn = attn + M * self.normalize.view(1,-1,1,1)
        elif  self.method == 'mul_after_sm':  
            pass
        elif  self.method == 'add_after_sm': 
            pass 
 

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        if self.method == 'mul_after_sm':  
            attn = attn * M * self.normalize.view(1,-1,1,1)
        elif self.method == 'add_after_sm': 
            attn = attn + M * self.normalize.view(1,-1,1,1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module):
    # Not used
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Violin_Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x 
    
class Layer_scale_init_violin_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Violin_Attention,Mlp_block=Mlp
                 ,init_values=1e-4,
                pos_emb = True, cls_tok = True, curve_list = ['s', 'sr', 'h', 'hr', 'm', 'mr','z', 'zr'], num_patches=196, qk_norm=False,mask='learned',scale=False,method='mul_v1',mask_sum='weighted'):

        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            pos_emb = pos_emb, cls_tok = cls_tok, curve_list = curve_list, num_patches=num_patches, qk_norm=qk_norm, mask=mask, scale=scale, method=method, mask_sum=mask_sum)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x
    
class violin_models(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers = Block,
                 Patch_layer=PatchEmbed,act_layer=nn.GELU,
                 Attention_block = Violin_Attention, Mlp_block=Mlp,
                dpr_constant=True,init_scale=1e-4,
                mlp_ratio_clstk = 4.0,
                pos_emb = True, cls_tok = True, curve_list = ['s', 'sr', 'h', 'hr', 'm', 'mr','z', 'zr'],qk_norm=False, mask='learned',scale=False,method='mul_v1',initialize=False,mask_sum='weighted',
                **kwargs):
        super().__init__()
        
        self.dropout_rate = drop_rate

            
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_tok = cls_tok
        self.pos_emb = pos_emb
        self.initialize = initialize

        if cls_tok:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if pos_emb:
            if cls_tok:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            else:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale,
                pos_emb = pos_emb, cls_tok = cls_tok, curve_list=curve_list, num_patches=num_patches, qk_norm=qk_norm, mask=mask, scale=scale, method=method, mask_sum=mask_sum)
            for i in range(depth)])
        

        
            
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if cls_tok:
            trunc_normal_(self.cls_token, std=.02)

        if pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = x + self.pos_embed
        
        x = torch.cat((cls_tokens, x), dim=1)
            
        for i , blk in enumerate(self.blocks):
            x = blk(x)
            
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):

        x = self.forward_features(x)
        
        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)
        
        return x

# DeiT III: Revenge of the ViT (https://arxiv.org/abs/2204.07118)

@register_model
def deit3_violin_tiny_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,   **kwargs):
    model = violin_models(
        img_size = img_size, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_violin_Block, **kwargs)
    model.default_cfg = _cfg()   
    return model
    
    
@register_model
def deit3_violin_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = violin_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_violin_Block, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def deit3_violin_medium_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False, **kwargs):
    model = violin_models(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Layer_scale_init_violin_Block, **kwargs)
    model.default_cfg = _cfg()
    return model 

@register_model
def deit3_violin_base_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = violin_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_violin_Block, **kwargs)   
    model.default_cfg = _cfg()
    return model
    
@register_model
def deit3_violin_large_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = violin_models(
        img_size = img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_violin_Block, **kwargs)
    model.default_cfg = _cfg()
    return model