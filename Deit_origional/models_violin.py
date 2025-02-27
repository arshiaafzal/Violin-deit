import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg, Mlp, HybridEmbed, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath

from curves import compute_curve_order, coords_to_index, index_to_coords_indexes
import numpy as np

__all__ = ['violin_base_patch16_224',]

def Casual_Mask_Decay(a_i , L):
    idx = torch.arange(L,device=a_i.device)
    I, J = torch.meshgrid(idx, idx, indexing='ij')
    E = (torch.abs((I-J)).float().view(1,1,L,L))
    M = torch.sigmoid(a_i).view(1,-1,1,1)**E
    return M

class Violin_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pos_emb = True, cls_tok = True, curve_list = ['s', 'sr', 'h', 'hr', 'm', 'mr'],num_patches=196):
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
        # self.curve_indices = []
        self.curve_indices_inv = []
        self.ai_list = []

        N = num_patches 
        order = torch.range(0,N-1)
        S = int(np.sqrt(N))
        grid = order.view(S,S).clone()

        for curve in curve_list:
            if curve not in ['s', 'sr', 'h', 'hr', 'm', 'mr']:
                raise ValueError("Invalid value for curve. Allowed values are: 's', 'sr', 'h', 'hr', 'm', 'mr'.")
            
            curve_coords = compute_curve_order(grid, curve)
            self.curve_indices_inv.append(torch.tensor(index_to_coords_indexes(curve_coords, S,S)  , dtype=torch.long ))  
            self.ai_list.append(nn.Parameter(torch.randn(num_heads)))


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        H = self.num_heads
        num_curves = len(self.ai_list)
        dev = x.device
        M = torch.zeros(1,H,N,N,device=dev)

        for c in range(num_curves):
            M_c = Casual_Mask_Decay(self.ai_list[c].to(dev), attn.shape[-1])
            ind = self.curve_indices_inv[c]
            M_c = M_c[:,:,ind][...,ind]
            if self.cls_tok:
                M_c = torch.cat((torch.ones((1,H,1,N), device=dev),torch.cat((torch.ones((1,H,N-1,1), device=dev),M_c),dim=-1)),dim=-2)
            M += M_c / num_curves
        attn = attn * M

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Violin_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pos_emb = True, cls_tok = True, curve_list = ['s', 'sr', 'h', 'hr', 'm', 'mr'], num_patches=196):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Violin_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, pos_emb = pos_emb, cls_tok = cls_tok, curve_list = curve_list, num_patches=num_patches)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Violin_Transformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, 
                 pos_emb = True, cls_tok = True, curve_list = ['s', 'sr', 'h', 'hr', 'm', 'mr']):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_tok = cls_tok
        self.pos_emb = pos_emb

        if cls_tok:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if pos_emb:
            if cls_tok:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            else:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Violin_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, pos_emb = pos_emb, cls_tok = cls_tok, curve_list=curve_list, num_patches=num_patches)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
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

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        if self.cls_tok:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_emb:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if self.cls_tok:
            return x[:, 0]
        else:
            return torch.mean(x, dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


##### Models with position and cls

@register_model
def violin_base_pos_cls(pretrained=False, **kwargs):
    model = Violin_Transformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pos_emb = True, cls_tok = True, curve_list = ['s', 'sr', 'h', 'hr', 'm', 'mr'], **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def violin_tiny_pos_cls(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        pos_emb = True, cls_tok = True, curve_list = ['s', 'sr', 'h', 'hr', 'm', 'mr'], **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def violin_small_pos_cls(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pos_emb = True, cls_tok = True, curve_list = ['s', 'sr', 'h', 'hr', 'm', 'mr'] ,**kwargs)
    model.default_cfg = _cfg()
    return model


##### Models with position embd and mean of class for classification


@register_model
def violin_base_pos(pretrained=False, **kwargs):
    model = Violin_Transformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pos_emb = True, cls_tok = False, curve_list = ['s', 'sr', 'h', 'hr', 'm', 'mr'], **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def violin_tiny_pos(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        pos_emb = True, cls_tok = False, curve_list = ['s', 'sr', 'h', 'hr', 'm', 'mr'], **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def violin_small_pos(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pos_emb = True, cls_tok = False, curve_list = ['s', 'sr', 'h', 'hr', 'm', 'mr'] ,**kwargs)
    model.default_cfg = _cfg()
    return model


