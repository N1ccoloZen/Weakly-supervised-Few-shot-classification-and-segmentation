import torch
import os

import torch.nn as nn

def remap_keys_hf(state_dict):
    new_state_dict = {}

    for k, v in state_dict.items():
    
        if 'embeddings' in k:
            if k.endswith('cls_token'):
                new_state_dict['cls_token'] = v
            elif k.endswith('patch_embeddings.projection.bias'):
                new_state_dict['patch_embed.proj.bias'] = v
            elif k.endswith('patch_embeddings.projection.weight'):
                new_state_dict['patch_embed.proj.weight'] = v
            elif k.endswith('position_embeddings'):
                new_state_dict['pos_embed'] = v
        elif 'encoder.layer' in k:

            parts = k.split('.')
            n_layer = parts[3]

            if 'layernorm_before' in k:
                if k.endswith('bias'):
                    new_state_dict[f'block.{n_layer}.norm1.bias'] = v
                else:
                    new_state_dict[f'block.{n_layer}.norm1.weight'] = v
            elif 'layernorm_after' in k:
                if k.endswith('bias'):
                    new_state_dict[f'block.{n_layer}.norm2.bias'] = v
                else:
                    new_state_dict[f'block.{n_layer}.norm2.weight'] = v
            elif 'attention.output.dense' in k:
                if k.endswith('bias'):
                    new_state_dict[f'block.{n_layer}.attn.proj.bias'] = v
                else:
                    new_state_dict[f'block.{n_layer}.attn.proj.weight'] = v
            elif 'intermediate.dense' in k:
                if k.endswith('bias'):
                    new_state_dict[f'block.{n_layer}.mlp.fc1.bias'] = v
                else:
                    new_state_dict[f'block.{n_layer}.mlp.fc1.weight'] = v
            elif 'output.dense' in k:
                if k.endswith('bias'):
                    new_state_dict[f'block.{n_layer}.mlp.fc2.bias'] = v
                else:
                    new_state_dict[f'block.{n_layer}.mlp.fc2.weight'] = v
        elif 'layernorm' in k:
            if k.endswith('bias'):
                new_state_dict['norm.bias'] = v
            else:
                new_state_dict['norm.weight'] = v

    max_layers = max([int(k.split('.')[3]) for k in state_dict.keys() if 'encoder.layer' in k]) +1

    for i in range(max_layers):

        prefix = f'vit.encoder.layer.{i}'

        q_w = state_dict[f'{prefix}.attention.attention.query.weight']
        k_w = state_dict[f'{prefix}.attention.attention.key.weight']
        v_w = state_dict[f'{prefix}.attention.attention.value.weight']
        q_b = state_dict[f'{prefix}.attention.attention.query.bias']
        k_b = state_dict[f'{prefix}.attention.attention.key.bias']
        v_b = state_dict[f'{prefix}.attention.attention.value.bias']

        new_state_dict[f'block.{i}.attn.qkv.weight'] = torch.cat([q_w, k_w, v_w], dim=0)
        new_state_dict[f'block.{i}.attn.qkv.bias'] = torch.cat([q_b, k_b, v_b], dim=0)

    return new_state_dict
            
#classifier.b/w
        #vit.encoder.layer --> 
        #NUMBER.attention.attention.key.bias, key.weight, query.bias, query.weight, value.bias, value.weight
        #NUMBER.attention.output.dense.b/w
        #NUMBER.intermediate.dense.b/w
        #NUMBER.layernorm_after.b/w , layernorm_before.b/w
        #NUMBER.output.dense.b/w 

'''
        mask_token

        block.NUMBER.norm1.w/b
        block.NUMBER.attn.qkv.w/b
        block.NUMBER.attn.proj.w/b
        block.NUMBER.ls1.gamma
        block.NUMBER.norm2.w/b
        block.NUMER.mlp.fc1.w/b
        block.NUMBER.mlp.fc2.w/b
        block.NUMBER.ls2.gamma

        '''

def remap_keys_fb(state_dict):

    '''
    OLD BACKBONE

    module.positional_embedding .image_projection .text_projection .logit_scale

    module.visual.cls_token, .visual.pos_embed, 
    module.visual.patch_embed.proj.w/b 
    module.visual.blocks.NUMBER.norm1.w/b, NUMBER.attn.qkv.w/b, NUMBER.attn.proj.w/b, NUMBER.norm2.w/b, NUMBER.mlp.fc1.w/b, NUMBER.mlp.fc2.w/b 
    module.visual.norm.w/b 

    module.transformer.resblocks.NUMBER.attn.in_proj_w/b, NUMBER.attn.out_proj.w/b, NUMBER.ln_1.w/b, NUMBER.mlp.c_fc.w/b, NUMBER.mlp.c_proj.w/b, NUMBER.ln_2.w/b 
    
    module.token_embedding.w 
    module.ln_final.w/b

    NEW BACKBONE

    positional_embedding
    text_projection
    vision_projection
    log_scale

    vision.cls_token, pos_embed, mask_token
    vision.patch_embed.proj.w/b
    vision.block.NUMBER.norm1.w/b, NUMBER.attn.qkv.w/b, NUMBER.attn.proj.w/b, NUMBER.norm2.w/b, NUMBER.mlp.0.w/b, NUMBER.mlp.2.w/b
    vision.norm.w/b

    text.resblock.NUMBER.attn.in_proj_w/b, NUMBER.attn.out_proj_w/b, NUMBER.ln_1.w/b, NUMBER.mlp.c_fc.w/b, NUMBER.mlp.c_proj.w/b, NUMBER.ln_2.w/b

    token.embedding.w
    layernorm_final.w/b
    
    '''

    

    new_state_dict = {}

    for k, v in state_dict['state_dict'].items():

        new_key = k.replace('module.', '')

        new_key = new_key.replace('visual', 'vision')
        new_key = new_key.replace('blocks', 'block')
        new_key = new_key.replace('transformer', 'text')
        new_key = new_key.replace('resblocks', 'resblock')
        new_key = new_key.replace('ln_final', 'layernorm_final')
        new_key = new_key.replace('image_projection', 'vision_projection')
        new_key = new_key.replace('logit_scale', 'log_scale')

        if 'mlp.fc1' in new_key:
            new_key = new_key.replace('fc1', '0')
        elif 'mlp.fc2' in new_key:
            new_key = new_key.replace('fc2', '2')
            
        new_state_dict[new_key] = v

    return new_state_dict

    