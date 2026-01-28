import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from functools import partial
from typing import Union, Sequence, Tuple
from collections import OrderedDict
#from model.backbone.dino.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, Attention, NestedTensorBlock as Block
from transformers import CLIPVisionModel, DistilBertConfig, DistilBertModel, CLIPModel
'''
class TextEncoder(nn.Module):
    def __init__(self, embed_dim, proj_dim):
        super().__init__()

        self.model = DistilBertModel(config=DistilBertConfig)
        self.projection = nn.Linear(embed_dim, proj_dim)
        self.layer_norm = nn.LayerNorm(proj_dim)

    def forward(self, input_ids, mask):
        
        x = self.model(input_ids, mask).last_hidden_state

        x = x[:, 0, :] #B, T[cls], E
        x = self.projection(x)

        return self.layer_norm(x)
    
class ImageEncoder(nn.Module):
    def __init__(self, base_model, embed_dim, proj_dim):
        super().__init__()

        self.model = base_model
        
        for param in self.model.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(embed_dim, proj_dim)
        self.layer_norm = nn.LayerNorm(proj_dim)

    def forward(self, x):
        x = self.projection(self.model(x))
        
        return self.layer_norm(x)
'''
class QuickGELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)
    
class LayerScale(nn.Module):
    
    '''
    From https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
    '''

    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class ResidualAttentionBlock(nn.Module):

    '''
    From https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
    '''

    def __init__(
            self,
            dim_model=384,
            num_heads=6,
            mlp_ratio=4.0,
            ls_init_value=None,
            act_layer = QuickGELU,
            is_cross_attn=False
            ):
        super().__init__()

        self.attn = nn.MultiheadAttention(dim_model, num_heads)
        self.ln_1 = nn.LayerNorm(dim_model)
        self.ls_1 = LayerScale(dim_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.mlp = nn.Sequential(OrderedDict([
            ('c_fc', nn.Linear(dim_model, dim_model * 4)),
            ('gelu', act_layer()),
            ('c_proj', nn.Linear(dim_model * 4, dim_model))
        ]))

        self.ln_2 = nn.LayerNorm(dim_model)
        self.ls_2 = LayerScale(dim_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(self, x, attn_mask= None, return_qkv=False):
        if return_qkv:
            qkv = F.linear(x, self.attn.in_proj_weight, self.attn.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)

            attn_out, attn_weight = self.attn(x, x, x, attn_mask=attn_mask, need_weights=True)
            return attn_out, (q, k, v, attn_weight)
        else:
            return self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0], None
        
    def forward(self, x, attn_mask=None, return_qkv=False):
        attn_out, qkv = self.attention(self.ln_1(x), attn_mask=attn_mask, return_qkv=return_qkv)

        x = x + self.ls_1(attn_out)
        x = x + self.ls_2(self.mlp(self.ln_2(x)))

        return x, qkv
    
class Attention(nn.Module):
    def __init__(
            self,
            embed_dim=384,
            num_heads=6,
            qkv_bias=True,
            proj_bias=True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim*3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=proj_bias)

    def forward(self, x, return_qkv=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        if return_qkv:
            return (q, k, v)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x, attn
    
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, c, h, w = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2) #[B, num_patches, embed_dim]

        return x

class Block(nn.Module):
    def __init__(
        self,
        num_embeddings,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        proj_bias=False,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=QuickGELU,
        init_values=None
    ):
        super().__init__()

        self.norm1 = norm_layer(num_embeddings)
        self.attn = Attention(num_embeddings, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias)
        self.norm2 = norm_layer(num_embeddings)

        mlp_dim = int(num_embeddings * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(num_embeddings, mlp_dim),
            act_layer(),
            nn.Linear(mlp_dim, num_embeddings)
        )

        self.ls1 = nn.Parameter(init_values * torch.ones(num_embeddings)) if init_values else None
        self.ls2 = nn.Parameter(init_values * torch.ones(num_embeddings)) if init_values else None

        self.drop_path = nn.Identity()

    def forward(self, x, return_qkv=False):
        if return_qkv:
            qkv = self.attn(self.norm1(x), return_qkv=return_qkv)
            return qkv
        
        attn_out, attn  = self.attn(self.norm1(x))
        #print('Attn shape is:', attn_out.shape)
        if self.ls1 is not None:
            x = x + self.drop_path(self.ls1 * attn_out)
            x = x + self.drop_path(self.ls2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    
class Transformer(nn.Module):
    def __init__(self, width=384, layers=12, heads=6, attn_mask=None):
        super().__init__()

        self.width = width
        self.num_layers = layers
        self.resblock = nn.ModuleList([
            ResidualAttentionBlock(width, heads) for _ in range(layers)
        ])
        self.attn_mask = attn_mask
    
    def forward(self, x, return_qkv=False):
        qkv_dict = {}

        for i, blk in enumerate(self.resblock):
            x, qkv = blk(x, attn_mask=self.attn_mask, return_qkv=return_qkv)
            if return_qkv and qkv is not None:
                qkv_dict[f'layer_{i}'] = qkv

        return x, qkv_dict if return_qkv else None

class ClipVisionTransformer(nn.Module):
    def __init__(
       self,
       img_size=224,
       patch_size=16,
       in_channels=3,
       embed_dim=384,
       depth=12,
       num_heads=6,
       mlp_ratio=4.0,
       qkv_bias=True,
       proj_bias=True,
       drop_path_rate=0.0,
       init_values=None,
       embed_layer=PatchEmbed,
       act_layer=nn.GELU,
       block_fn=Block,
       norm_layer=partial(nn.LayerNorm, eps=1e-5),
       interpolate_offsets=0.1,
       num_tokens=1     
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = embed_layer(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_channels, 
            embed_dim=embed_dim
            )
        
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + num_tokens, embed_dim))

        drop_path = [drop_path_rate] * depth

        block_list = [block_fn(
            num_embeddings=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            drop_path=drop_path[i],
            norm_layer=norm_layer,
            act_layer=act_layer,
            init_values=init_values
        ) for i in range(depth)]

        self.block = nn.ModuleList(block_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.interpolate_offsets = interpolate_offsets

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.mask_token, std=1e-6)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        n_patch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if n_patch == N and w == h:
            return self.pos_embed
        
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]

        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N)) #original size -> number of patches in each dimension
        assert N == M * M

        kwargs = {}
        if self.interpolate_offsets:
            sx = float(w0 + self.interpolate_offsets) / M
            sy = float(h0 + self.interpolate_offsets) / M
            kwargs['scale_factor'] = (sx, sy)
        else:
            kwargs['size'] = (h0, w0)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode='bicubic',
            align_corners=False,
            **kwargs
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, mask=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        '''
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        '''
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        return x
    
    def forward_features_list(self, x_list, mask_list):

        x = [self.prepare_tokens_with_masks(x, mask) for x, mask in zip(x_list, mask_list)]

        for blk in self.block:
            x = blk(x)

        all_x = x
        output = []

        for x, masks in zip(all_x, mask_list):
            x_norm = self.norm(x)

            output.append(
                {
                    'x_norm_clstoken': x_norm[:, 0],
                    'x_norm_patchtoken' : x_norm[:, 1:],
                    'x_prenorm' : x,
                    'mask' : masks,
                }
            )

        return output

    def forward_features(self, x, mask=None):

        if isinstance(x, list):
            return self.forward_features_list(x, mask)
        
        x = self.prepare_tokens_with_masks(x, mask)

        for blk in self.block:
            x = blk(x)

        x_norm = self.norm(x)

        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtoken": x_norm[:, 1:],
            "x_prenorm": x,
            "masks": mask,
        }
    def get_intermediate_layers(self, x, n_layers, return_qkv=False, return_class_token=False):
        x = self.prepare_tokens_with_masks(x)

        output, total_block_len = [], len(self.block)
        #blocks_to_take = [2, 5, 8, 11]   #blocks from 0 to 11 -> 12 total
        blocks_to_take = range(total_block_len - n_layers, total_block_len) if isinstance(n_layers, int) else n_layers
        for i, blk in enumerate(self.block):
            blk.eval()
            if return_qkv:
                qkv = blk(x, return_qkv=return_qkv)
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
            if return_qkv and i == len(self.block)-1:
                qkv_final = qkv
        if return_qkv:
            return output, qkv_final
        return output
    
    def forward(self, *args, is_training=False, **kwargs):
        feats = self.forward_features(*args, **kwargs)

        if is_training:
            return feats
        else:
            return self.head(feats["x_norm_clstoken"])
        
class CLIPFeatureExtractor(nn.Module):
    def __init__(
            self,
            embed_dim=512,
            img_size=224,
            vision_layers=12,
            vision_dim=384,
            vision_patch_size=16,
            vision_heads=6,
            context_lenght=77,
            vocab_size=49408,
            transformer_dim=512,
            transformer_heads=8,
            transformer_layers=12,
            **vision_kwargs
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.vision_dim = vision_dim
        self.context_length = context_lenght
        self.vocab_size = vocab_size
        self.transformer_dim = transformer_dim

        self.vision = ClipVisionTransformer(
            img_size=img_size,
            patch_size=vision_patch_size,
            in_channels=3,
            embed_dim=vision_dim,
            depth=vision_layers,
            num_heads=vision_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            proj_bias=True,
            **vision_kwargs
        )

        self.text = Transformer(width=transformer_dim, layers=transformer_layers, heads=transformer_heads)

        self.token_embedding = nn.Embedding(vocab_size, transformer_dim)
        self.positional_embedding = nn.Parameter(torch.empty(context_lenght, transformer_dim))
        self.layernorm_final = nn.LayerNorm(transformer_dim)

        #To joint projection embeddings
        self.text_projection = nn.Parameter(torch.empty(transformer_dim, embed_dim))
        self.vision_projection = nn.Parameter(torch.empty(vision_dim, embed_dim))
        self.log_scale = nn.Parameter(torch.ones([]) * 2.6592)

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=self.transformer_dim ** -0.5)
        nn.init.normal_(self.vision_projection, std=self.vision_dim ** -0.5)

    def encode_image(self, img, n_layers=None, return_qkv=False, normalize=True, mask=None):
        
        if n_layers is not None:
            layer_out = self.vision.get_intermediate_layers(
                img,
                n_layers=n_layers,
                return_qkv=return_qkv,
                return_class_token=False
            )

            if return_qkv:
                layer_features, qkv = layer_out
            else:
                layer_features = layer_out
                qkv = None

            final_cls = self.vision.norm(layer_features[-1])[:,0]
            embeds = final_cls @ self.vision_projection

            #print('In encode_img, the size of embeds is: ', embeds.shape)
            #print('The final_cls has shape: ', final_cls.shape)
            #print('While the size of layer features is: ', layer_features[-1].shape)

            result = {
                'embedding' : F.normalize(embeds, dim=-1) if normalize else embeds,
                'layer_features' : layer_features,
                'cls_token' : final_cls
            }

            if return_qkv:
                result['qkv_data'] = qkv
            
            #output = [out[:, 1:] for out in layer_features]

            return result

        else:

            features = self.vision.forward_features(
                img,
                mask=mask
            )
            embeds = features['x_norm_clstoken'] @ self.vision_projection

            result = {
                'embedding' : F.normalize(embeds, dim=-1) if normalize else embeds,
                'layer_features' : features,
                'cls_token' : features['x_norm_clstoken'],
                'patch_tokens' : features['x_norm_patchtoken']
            }

            if return_qkv:
                result['qkv_data'] = features.get('qkv_data', None)

            return result['embedding']
        
    def encode_text(self, text, return_qkv=False, normalize=True):

        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)

        x, qkv = self.text(x, return_qkv=return_qkv)
        x = x.permute(1, 0, 2)
        x = self.layernorm_final(x)

        #features from EoT embedding (highest token id)
        #print(' -> encode_text output before text_projection: ', x.shape)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        result = {'embedding' : F.normalize(x, dim=-1) if normalize else x}

        return result

    def forward(self, img=None, text=None, return_layers=None, return_qkv=False, mask=None, normalize=True):
        img_features, text_features = None, None

        if img is not None:
            img_features = self.encode_image(
                img, 
                n_layers=return_layers, 
                return_qkv=return_qkv, 
                normalize=normalize, 
                mask=mask
            )
            
        if text is not None:
            text_features = self.encode_text(
                text,
                return_qkv=return_qkv,
                normalize=normalize
            )
        
        if img is not None and text is not None:
            return img_features, text_features
        elif img is not None:
            return img_features
        elif text is not None:
            return text_features
        else:
            raise ValueError("Provide either image or text")

def vit_small(patch_size=14, num_register_tokens=0, **kwargs):
    model = ClipVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=Attention),
        init_values=None,
        num_tokens=1,    #2305,  #1114
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = ClipVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=Attention),
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = ClipVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=Attention),
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = ClipVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=Attention),
        **kwargs,
    )
    return model


        