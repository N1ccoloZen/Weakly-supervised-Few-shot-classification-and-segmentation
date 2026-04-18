from functools import reduce
from operator import add

import torch
import os 
import math
import numpy as np
import torch.nn as nn

from einops import rearrange

from model.pl_module import FSCSModule
from model.module.cst import CorrelationTransformer
from safetensors.torch import load_file
#import model.backbone.dino.vision_transformer as vits
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torch.utils.checkpoint import checkpoint
from model.backbone.clip.clip import CLIPFeatureExtractor
from model.backbone.clip.remap_keys import remap_keys_fb
from model.backbone.clip.gradcam import GradCAM
from model.backbone.segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
#from model.module.moco import ClassCondMoCoQueues
import matplotlib.pyplot as plt
import cv2



class ClfSegTransformer(FSCSModule):
    def __init__(self, args):
        super(ClfSegTransformer, self).__init__(args)

        self.backbone = CLIPFeatureExtractor()

        #https://github.com/facebookresearch/SLIP

        url = 'https://dl.fbaipublicfiles.com/slip/clip_small_25ep.pt'
    
        state_dict = torch.hub.load_state_dict_from_url(url)

        new_state_dict = remap_keys_fb(state_dict)

        self.backbone.load_state_dict(new_state_dict, strict=False)

        self.nlayer = 12
        self.nhead = 6
        self.imgsize = args.imgsize
        self.sptsize = int(int(args.imgsize // 16) // 4)
        self.layers_to_take = args.nlayers if args.nlayers <= self.backbone.vision.depth else self.backbone.vision.depth

        self.use_sam = False
        self.use_text = False
        self.distill = False
        self.distillv1 = False

        self.qry_pts = None
        self.labels = None

        if args.use_text:
            self.text_project = nn.Linear(self.backbone.embed_dim, self.backbone.vision_dim)
            #self.text_project = TextToVision(self.backbone.embed_dim, self.backbone.vision_dim)
            self.learner = CorrelationTransformer([self.nhead * self.layers_to_take], args.way, clip_dim=self.backbone.vision_dim) #embed_dim has to be used for distillW3
            #self.condition_on_episode = VisualConditionedText(self.backbone.embed_dim, 64, self.backbone.embed_dim) #https://arxiv.org/pdf/2203.05557
            #self.features_adapt = DenseFeaturesAdapter(self.backbone.vision_dim, self.nhead)
            #self.temperature = nn.Parameter(torch.ones(1) * 10.0)
            #self.bias = nn.Parameter(torch.zeros(1))

        
            self.use_text = True
        else:
            self.learner = CorrelationTransformer([self.nhead * self.layers_to_take], args.way, clip_dim=self.backbone.embed_dim)
            self.features_adapt = DenseFeaturesAdapter(self.backbone.vision_dim, self.nhead)
        
        self.sup = args.sup

        for k, v in self.backbone.named_parameters():
            v.requires_grad = False
    
        if args.use_sam:
            sam_base = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth' #baseSAMWS3, baseSAMWS2, SAMWS1, SAMWS0
            sam_large = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth' #later 
            sam_huge = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth' #for distillW0
            model_type = 'vit_b'
            if args.distill:
                model_type = 'vit_l'
                sam_state_dict = torch.hub.load_state_dict_from_url(sam_large)
                self.logits = None
                self.distill = True
            else:
                sam_state_dict = torch.hub.load_state_dict_from_url(sam_base)

            self.sam = sam_model_registry[model_type](checkpoint=sam_state_dict)
            self.sam.to(self.device)
            #self.mask_generator = SamAutomaticMaskGenerator(self.sam)
            self.predictor = SamPredictor(self.sam)

            for k, p in self.predictor.model.named_parameters():
                p.requires_grad = False
            self.predictor.model.eval()
            self.use_sam = True

        if args.debug:
            from pathlib import Path
            self.debug_dir_refine = Path("debug_outputs/maskSAMfinal")
            self.debug_dir_refine.mkdir(parents=True, exist_ok=True)

            self.debug_dir_mask = Path("debug_outputs/maskSAM")
            self.debug_dir_mask.mkdir(parents=True, exist_ok=True)

            self.debug_dir_maskgen = Path("debug_outputs/maskgenerator")
            self.debug_dir_maskgen.mkdir(parents=True, exist_ok=True)

            self.debug_dir_surgery = Path("debug_outputs/maskwithSurgery")
            self.debug_dir_surgery.mkdir(parents=True, exist_ok=True)

        '''
        self.momentum_backbone = CLIPFeatureExtractor()
        self.momentum_backbone.load_state_dict(new_state_dict, strict=False)
        self.momentum_backbone.eval()
        for p in self.momentum_backbone.parameters():
            p.requires_grad = False
        
        self.moco_queue = ClassCondMoCoQueues(
            num_classes=args.way,
            embed_dim=self.backbone.embed_dim,
            k_per_class=self.queue_size,
            temperature=0.07,
            device=self.device
        )
        self.momentum = 0.999
        '''
   
    def forward(self, batch):
        '''
        query_img.shape : [bsz, 3, H, W]
        support_imgs.shape : [bsz, way, 3, H, W]
        support_masks.shape : [bsz, way, H, W]
        '''
        '''
        qryimg_name = batch['query_name']
        print('Names of qry images: ', qryimg_name)
        sptimg_name = batch['support_names']
        print('Names of spt images', sptimg_name)
        classpresence = batch['support_classes']
        print(f'These are these support classes in the episode: \n{classpresence}')
        '''
        tokenized_labels = batch['tokenized_text'].to(self.device) #[batch, 77]
        #print('These are the dimensions of tokenized_labels: ', tokenized_labels.shape)
        tokenized_labels = tokenized_labels.view(-1, tokenized_labels.size(-1))
        #query_presence = batch['query_class_presence']
        #print(f'In the qry there are these classes: \n{query_presence}')

        spt_img = rearrange(batch['support_imgs'].squeeze(2), 'b n c h w -> (b n) c h w')
        spt_mask = None if self.sup == 'pseudo' else rearrange(batch['support_masks'].squeeze(2), 'b n h w -> (b n) h w')
        qry_img = batch['query_img']
        qry_mask = None

        spt_img = spt_img.to(memory_format=torch.channels_last)
        qry_img = qry_img.to(memory_format=torch.channels_last)
   
        B, C, H, W = qry_img.shape

        text_feats = self.backbone.encode_text(tokenized_labels)
        text_embedds = text_feats['embedding'] #[8, 512]
        text_feat = text_feats['embedding']
        #print(f'token features of size: {text_feat.shape}')

        text_embedds = F.normalize(text_embedds, p=2, dim=-1)
        text_feat = F.normalize(text_feat, p=2, dim=-1)

        if self.use_text:
            text_proj = self.text_project(text_feat)

        qry_feats_extr = self.extract_clip_feats(qry_img, return_qkv=self.sup == 'pseudo', return_attn=self.sup=='pseudo')
        spt_feats_extr = self.extract_clip_feats(spt_img, return_qkv=self.sup == 'pseudo', return_attn=self.sup=='pseudo')

        qry_feats = qry_feats_extr['layer_features']
        spt_feats = spt_feats_extr['layer_features']

        qry_feats = torch.stack(qry_feats, dim=1)
        spt_feats = torch.stack(spt_feats, dim=1)
        qry_feats = qry_feats.repeat_interleave(self.args.way, dim=0)
        
        with torch.no_grad():
            
            '''qry_feats_extr = self.extract_clip_feats(qry_img, return_qkv=self.sup == 'pseudo')
            spt_feats_extr = self.extract_clip_feats(spt_img, return_qkv=self.sup == 'pseudo')'''

            if self.sup == 'pseudo':
                
                qry_q = qry_feats_extr['qkv_data'][0]
                qry_k = qry_feats_extr['qkv_data'][1]
                qry_v = qry_feats_extr['qkv_data'][2]
                
                spt_q = spt_feats_extr['qkv_data'][0]
                spt_k = spt_feats_extr['qkv_data'][1]
                spt_v = spt_feats_extr['qkv_data'][2]
                
                qry_qkv = torch.stack([qry_q, qry_k, qry_v], dim=0)
                spt_qkv = torch.stack([spt_q, spt_k, spt_v], dim=0)

                resize = (self.imgsize, self.imgsize) if self.training else (batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item())
                
                
                if self.args.use_text:
                    with torch.enable_grad():
                        #spt_mask, _, qry_mask, _ = self.generate_pseudo_mask_value(qry_qkv, spt_qkv, class_gt=batch['query_class_presence'].flatten(), text_feat=text_feat, thr=0.4, resize=resize, text_weight=0.2)
                        #spt_mask, qry_mask = self.generate_rollout_pseudo_mask(qry_qkv, spt_qkv, class_gt=batch['query_class_presence'].flatten(), thr=0.4, resize=resize, spt_features=spt_feats_extr, qry_features=qry_feats_extr, text_feat=text_feat)
                        spt_mask, _, qry_mask, qry_points = self.generate_pseudo_mask_qkey(qry_qkv, spt_qkv, class_gt=batch['query_class_presence'].flatten(), text_feat=text_proj, thr=0.4, resize=resize, text_weight=0.3)
                else:
                    spt_mask, qry_mask = self.generate_pseudo_mask_qkey(qry_qkv, spt_qkv, class_gt=batch['query_class_presence'].flatten(), thr=0.4, resize=resize)


                if self.args.debug:
                            names = batch['support_names']
                            name = names[0][0]
                            for i in range(B):  
                                if self.training and self.global_rank == 0:
                                    self.save_debug_image(
                                        image=spt_img[i],
                                        mask=spt_mask[i],
                                        #points=qry_pts[0],
                                        #labels=labels[0],
                                        name=f"generated_sptimg{name[i]}.png",
                                        dir=self.debug_dir_surgery)
                if self.args.use_sam:

                    spt_rollout, fg_rollout, combined = self.attention_rollout(attention_maps=spt_feats_extr['attn_maps'], discard_ratio=0.9, text_feat=text_feat)
                    spt_to_combine = spt_feats_extr['layer_features']
                    combined = self.class_specific_fg_map(combined, spt_to_combine[-1][:, 1:, :], text_embedds, alpha=0.4)

                    spt_points, spt_labels = self.rollout_to_points(combined, self.imgsize, patch_size=16, n_pos=5, n_neg=3, fg_threshold=0.5)

                    spt_mask = []
                    distill_mask = []
                    mask_logits = []
                    #self.qry_pts = qry_points
                    #SAM EMBEDDINGS shape is [1, 256, 64, 64]
                    #spt_points[B, 8, 2], labels[B, 8]
                    presence = batch['query_class_presence'].flatten()
                    #print(presence)

                    for i in range(B):
                        support_img = spt_img[i, :]
                        s_img = support_img.permute(1, 2, 0).detach().cpu().numpy()
                        s_img = (s_img * 255).astype(np.uint8)

                        self.predictor.set_image(s_img)
                        #mask_to_refine = mask_s[i, :].unsqueeze(0)
                        spt_pts = spt_points[i, :].unsqueeze(0).float()
                        
                        #labels = torch.ones(spt_pts.shape[1]).unsqueeze(0).to(spt_img.device)
                        #labels[:, -3:] = 0
                        
                        labels = spt_labels[i, :].unsqueeze(0)
                        
                        if i==0 and self.args.debug:
                            print(f'Points fed into SAM are: {spt_pts.shape}')
                            print(f'first point is: {spt_pts[:, 0, :]}')
                            print(f'spt pts of {spt_pts[:, 0, :].shape}')
                            print(f'labels pts of {labels[:, 0].shape}')
                            print(f'labels are {labels}') 
                        
                        #print(f'spt pts of {spt_pts.shape}')
                        #print(f'labels pts of {spt_labels.shape}')
                    
                        first_point = spt_pts[:, :1, :]

                        masks, scores, logits = self.predictor.predict_torch(
                            point_coords=first_point,
                            point_labels=labels[:, :1],
                            multimask_output=True
                        )

                        best = scores.argmax()
                        mask_input = logits[:, best, :, :]

                        mask_sam, scores, logits = self.predictor.predict_torch(
                            point_coords=spt_pts[:, :2, :],
                            point_labels=labels[:, :2],
                            mask_input=mask_input,
                            multimask_output=True
                        )

                        best = scores.argmax()
                        mask_input = logits[:, best, :, :]

                        masks, scores, logits = self.predictor.predict_torch(
                            point_coords=spt_pts[:, :3, :],
                            point_labels=labels[:, :3],
                            mask_input=mask_input,
                            multimask_output=True
                        )

                        best = scores.argmax()
                        mask_input = logits[:, best, :, :]

                        mask_sam, scores, logits = self.predictor.predict_torch(
                            point_coords=spt_pts,
                            point_labels=labels,
                            mask_input=mask_input,
                            multimask_output=False
                        )

                        spt_mask.append(mask_sam)
                        self.predictor.reset_image()

                        if self.args.distill and not self.args.eval:
                            if presence[i]:
                                #print(f'presence is :{presence[i]}')
                                query_img = qry_img[i, :]
                                q_img = query_img.permute(1, 2, 0).detach().cpu().numpy()
                                q_img = (q_img * 255).astype(np.uint8)

                                self.predictor.set_image(q_img)
                                mask_to_refine = qry_mask[i, :].unsqueeze(0)

                                qry_pts = qry_points[i, :].unsqueeze(0).float()
                                
                                labels = torch.ones(qry_pts.shape[1]).unsqueeze(0).to(spt_img.device)
                                labels[:, -3:] = 0
                                
                                first_point = qry_pts[:, :1, :]

                                masks, scores, logits = self.predictor.predict_torch(
                                    point_coords=first_point,
                                    point_labels=labels[:, :1],
                                    multimask_output=True
                                )

                                best = scores.argmax()
                                mask_input = logits[:, best, :, :]
                                #print(f'mask input of shape {mask_input.shape}')

                                mask_sam, scores, logits = self.predictor.predict_torch(
                                    point_coords=qry_pts,
                                    point_labels=labels,
                                    mask_input=mask_input,
                                    multimask_output=False
                                )
                                self.labels = labels
                                mask_logits.append(logits)
                                distill_mask.append(mask_sam)
                                self.predictor.reset_image()
                            else:
                                distill_mask.append(qry_mask[i, :].unsqueeze(0).unsqueeze(0))

                            #generated_masks = self.mask_generator.generate(q_img)

                            if self.args.debug:
                                if self.training and self.global_rank == 0:
                                    self.save_debug_image(
                                        image=qry_img[i],
                                        mask=mask_sam,
                                        points=qry_pts[0],
                                        labels=labels[0],
                                        name=f"SAMgenerated{self.current_epoch}_query{i}.png",
                                        dir=self.debug_dir_refine)
                                    '''self.save_debug_image(
                                        image=qry_img[i],
                                        name=f'generatedMask{self.current_epoch}_query{i}.png',
                                        dir=self.debug_dir_maskgen,
                                        masks=generated_masks, 
                                        from_generator=True)'''

                    if self.args.distill and not self.args.eval:
                        distill_mask = torch.stack(distill_mask, dim=0)
                        distill_mask = distill_mask.squeeze(1)

                        #print(distill_mask.shape)

                        #mask_logits = torch.stack(mask_logits, dim=0)
                        #mask_logits = mask_logits.squeeze(1)
                        #mask_logits = F.interpolate(mask_logits, resize, mode='bilinear', align_corners=False).squeeze(1)
                        #self.logits = mask_logits
                        #print(f'sam logits of shape {mask_logits.shape}')

                        distill_mask = F.interpolate(distill_mask.float(), resize, mode='nearest').squeeze(1)
                        #print(f'distill shape: {distill_mask.shape}')
                        #qry_mask = distill_mask

                    spt_mask = torch.stack(spt_mask, dim=0)
                    spt_mask = spt_mask.squeeze(1).squeeze(1)
                    
                if self.use_text and not self.use_sam:
                    spt_mask = spt_mask

                batch['query_pmask'] = qry_mask  # used only for avg-head-pseudo-mask training
                batch['support_pmasks'] = spt_mask # only used for vis

                qry_qkv = qry_qkv.repeat_interleave(self.args.way, dim=1)

            '''qry_feats = qry_feats_extr['layer_features']
            spt_feats = spt_feats_extr['layer_features']

            qry_feats = torch.stack(qry_feats, dim=1)
            spt_feats = torch.stack(spt_feats, dim=1)
            qry_feats = qry_feats.repeat_interleave(self.args.way, dim=0)'''

            # [batch, nlayer, (1+HW), dim]
            B, L, T, C = spt_feats.shape

            h = w = int(self.imgsize // 16)
            ch = int(C // self.nhead)

            qry_feat = qry_feats.reshape(B * L, T, C)[:, 1:, :] # 1-HW token: img tokens
            spt_feat = spt_feats.reshape(B * L, T, C)[:, 1:, :] # 1-HW token: img tokens
            spt_cls = spt_feats.reshape(B * L, T, C)[:, 0, :]   # 0-th token: cls token
        
            qry_feat = rearrange(qry_feat, 'b p (n c) -> b n p c', n=self.nhead, c=ch)

            # resize support features 50x50 -> 12x12 to reduce computation
            spt_feat = rearrange(spt_feat, 'b (h w) d -> b d h w', h=h, w=w)
            spt_feat = F.interpolate(spt_feat, (self.sptsize, self.sptsize), mode='bilinear', align_corners=True)
            spt_feat = rearrange(spt_feat, 'b (n c) h w -> b n (h w) c', n=self.nhead, c=ch)

            spt_cls = rearrange(spt_cls, 'b (n c) -> b n 1 c', n=self.nhead, c=ch)
            spt_feat = torch.cat([spt_cls, spt_feat], dim=2)
            
            qry_feat = F.normalize(qry_feat, p=2, dim=-1)
            spt_feat = F.normalize(spt_feat, p=2, dim=-1)

            qry_to_emb = qry_feat
            spt_to_emb = spt_feat

            #print(f'spt feat shape: {spt_feat.shape}')
            #print(f'qry feat shape: {qry_feat.shape}')

            qry_to_emb = rearrange(qry_to_emb, '(b l) n t c -> b l t (n c)', b=B, l=self.layers_to_take) #[8, 12, 2500, 384]
            spt_to_emb = rearrange(spt_to_emb, '(b l) n t c -> b l t (n c)', b=B, l=self.layers_to_take) #[8, 12, 145, 384]

            qry_adap = rearrange(qry_to_emb, 'b l t c -> (b l) t c')
            spt_adap = rearrange(spt_to_emb, 'b l t c -> (b l) t c')
    
        qry_embeds = qry_feats_extr['embedding'] #[8, 512]
        qry_embeds = F.normalize(qry_embeds, p=2, dim=-1)

        spt_embeds = spt_feats_extr['embedding']
        spt_embeds = F.normalize(spt_embeds, p=2, dim=-1)

        #qry_feat = self.features_adapt(qry_adap)
        #spt_feat = self.features_adapt(spt_adap) #laater add these blocks

        #qry_feat = F.normalize(qry_feat, p=2, dim=-1)
        #spt_feat = F.normalize(spt_feat, p=2, dim=-1)

        #print(f'spt feat shape: {spt_feat.shape}')
        #print(f'qry feat shape: {qry_feat.shape}')

        #qry_feat = rearrange(qry_feat, 'b n (h c) -> b h n c', h=self.nhead, c=ch)
        #spt_feat = rearrange(spt_feat, 'b n (h c) -> b h n c', h=self.nhead, c=ch)
       
        headwise_corr = torch.einsum('b d q c, b d s c -> b d q s', qry_feat, spt_feat)
        headwise_corr = rearrange(headwise_corr, '(b l) d q s -> b (l d) q s', b=B, l=L)
        '''
        with torch.no_grad():

            spt_text_momentum = self.backbone.encode_text(tokenized_labels)
            spt_text_feats_m = spt_text_momentum['embedding']
            spt_text_feats_m = F.normalize(spt_text_feats_m, dim=-1)
        '''
        if self.args.use_text:
            #print(f'spt embeds shape: {spt_embeds.shape}')
            #print(f'qry embeds shape: {qry_embeds.shape}')
            #print(f'txt embeds shape: {text_feat.shape}')
            #qry_embeds = qry_embeds.repeat_interleave(self.args.way, dim=0)
            #text_feat = text_feat.repeat_interleave(self.args.way, dim=0)
            #text_proj_conditioned = self.condition_on_episode(spt_embeds.detach(), qry_embeds.detach(), text_feat)

            #headwise_corr, gate = self.modulate_corr(headwise_corr, spt_embeds.detach(), qry_embeds.detach())
            text_proj_conditioned = text_proj
        #else:
            #text_proj_conditioned = text_embedds    #text_feat, text_proj_conditioned to add later
        #if not self.args.use_text:#delete 
            #text_proj = text_feat text_proj_conditioned
        output_cls, output_masks = self.learner(headwise_corr, spt_mask, text_proj_conditioned)

        # BN, 2, H, W
        output_cls = output_cls.view(-1, self.way, 2)
        output_masks = self.upsample_logit_mask(output_masks, batch)
        output_masks = output_masks.view(-1, self.way, *output_masks.shape[1:])
        
        if not self.args.distill:
            distill_mask = qry_mask
        return output_cls, output_masks, distill_mask

    @torch.enable_grad()
    def generate_rollout_pseudo_mask(self, qry_qkv, spt_qkv, class_gt, text_feat=None, resize=(800, 800), thr=0.4, text_weight=0.2, spt_features=None, qry_features=None):
        # 0-th token: cls token
        # 1-HW token: img token
        # qry_qkv [qkv, batch, head, (1+HW), dim]
        # text_feats [batch, hdim]
        _, B, N, L, C = qry_qkv.shape
        spt_cls = spt_qkv[0, :, :, 0, :]
        spt_key = spt_qkv[1, :, :, 1:, :]
        qry_key = qry_qkv[1, :, :, 1:, :]

        h = w = int(self.imgsize // 16)
        ch = int(C // self.nhead)

        qry_key = rearrange(qry_key, 'b n (h w) c -> b n h w c', h=h, w=w)
        spt_key = rearrange(spt_key, 'b n (h w) c -> b n h w c', h=h, w=w)

        qry_key = F.normalize(qry_key, p=2, dim=-1)
        spt_key = F.normalize(spt_key, p=2, dim=-1)
        spt_cls = F.normalize(spt_cls, p=2, dim=-1)

        spt_attn = spt_features['attn_maps']
        qry_attn = qry_features['attn_maps']
        spt_feat = spt_features['layer_features'][-1][:, 1:, :]
        qry_feat = qry_features['layer_features'][-1][:, 1:, :]

        if spt_attn is not None:
                spt_rollout, _, spt_combined = self.attention_rollout(attention_maps=spt_attn, discard_ratio=0.9, attn_values=spt_qkv[2], text_feat=text_feat)
                self_corr_map = self.class_specific_fg_map(rollout_map=spt_combined, patch_features=spt_feat, text_feat=text_feat, alpha=0.4)
                self_corr_map = self_corr_map.reshape(B, 1, h, w)
                self_corr = F.interpolate(self_corr_map, resize, mode='bilinear', align_corners=True).squeeze(1)

                if qry_attn is not None:
                    qry_rollout = self.attention_rollout(attention_maps=qry_attn, head_fusion='mean', discard_ratio=0.9)
                    spt_flat = spt_rollout
                    qry_flat = qry_rollout

                    cross_score = torch.bmm(qry_flat.unsqueeze(1), spt_flat.unsqueeze(2)).squeeze()

                    cros_map = qry_rollout * cross_score.unsqueeze(-1).clamp(0, 1)
                    cros_map = cros_map.reshape(B, 1, h, w)
                    cros_corr = F.interpolate(cros_map, resize, mode='bilinear', align_corners=True).squeeze(1)
                else:
                    spt_cls_norm = spt_cls
                    cros_corr_raw = torch.einsum('b n h w c, b n c -> b n h w', qry_key, spt_cls_norm).mean(dim=1, keepdim=True)
                    cros_corr = F.interpolate(cros_corr_raw, resize, mode='bilinear', align_corners=True).squeeze(1)
                    cros_corr = (cros_corr + 1.) * .5

                self_corr = self_corr / (self_corr.flatten(1).max(dim=1).values.unsqueeze(-1).unsqueeze(-1) + 1e-6)

                cros_corr = cros_corr / (cros_corr.flatten(1).max(dim=1).values.unsqueeze(-1).unsqueeze(-1) + 1e-6)

        ret_self = (self_corr > thr).float()
        ret_cros = (cros_corr > thr).float()

        ret_cros[class_gt.squeeze(-1) == False] = 0.

        return ret_self, ret_cros
    
    @torch.enable_grad()
    def generate_pseudo_mask_value(self, qry_qkv, spt_qkv, class_gt, text_feat=None, resize=(800, 800), thr=0.4, text_weight=0.2, spt_features=None, qry_features=None):
        # 0-th token: cls token
        # 1-HW token: img token
        # qry_qkv [qkv, batch, head, (1+HW), dim]
        # text_feats [batch, hdim]
        _, B, N, L, C = qry_qkv.shape
        #spt_cls = spt_qkv[0, :, :, 0, :]
        #spt_key = spt_qkv[1, :, :, 1:, :]
        #qry_key = qry_qkv[1, :, :, 1:, :]

        spt_cls = spt_qkv[2, :, :, 0, :]
        spt_val = spt_qkv[2, :, :, 1:, :]
        qry_val = qry_qkv[2, :, :, 1:, :]

        h = w = int(self.imgsize // 16)
        ch = int(C // self.nhead)

        #qry_key = rearrange(qry_key, 'b n (h w) c -> b n h w c', h=h, w=w)
        #spt_key = rearrange(spt_key, 'b n (h w) c -> b n h w c', h=h, w=w)

        spt_val = rearrange(spt_val, 'b n (h w) c -> b n h w c', h=h, w=w)
        qry_val = rearrange(qry_val, 'b n (h w) c -> b n h w c', h=h, w=w)

        #qry_key = F.normalize(qry_key, p=2, dim=-1)
        #spt_key = F.normalize(spt_key, p=2, dim=-1)
        
        spt_cls = F.normalize(spt_cls, p=2, dim=-1)

        spt_val = F.normalize(spt_val, p=2, dim=-1)
        qry_val = F.normalize(qry_val, p=2, dim=-1)
        

        if self.use_text and not self.args.use_sam:
            text_feat = F.normalize(text_feat, p=2, dim=-1)
            text_feat = rearrange(text_feat, 'b (n d) -> b n d', n=self.nhead)

            text_qry_corr = torch.einsum('b n h w c, b n c -> b n h w', qry_val, text_feat).mean(dim=1, keepdim=True)
            text_spt_corr = torch.einsum('b n h w c, b n c -> b n h w', spt_val, text_feat).mean(dim=1, keepdim=True)

        cros_corr = torch.einsum('b n h w c, b n c -> b n h w', qry_val, spt_cls)
        self_corr = torch.einsum('b n h w c, b n c -> b n h w', spt_val, spt_cls)
        
        if not self.use_text and not self.use_sam:
            self_corr = self_corr.mean(dim=1, keepdim=True)
            cros_corr = cros_corr.mean(dim=1, keepdim=True)

            self_corr = F.interpolate(self_corr, resize, mode='bilinear', align_corners=True).squeeze(1)
            cros_corr = F.interpolate(cros_corr, resize, mode='bilinear', align_corners=True).squeeze(1)

            cros_corr_ret = (cros_corr + 1.) * .5  # [-1, 1] -> [0, 1]
            self_corr_ret = (self_corr + 1.) * .5  # [-1, 1] -> [0, 1]

            ret_self = (self_corr_ret > thr).float()
            ret_cros = (cros_corr_ret > thr).float()

            ret_cros[class_gt.squeeze(-1) == False] = 0.

            return ret_self, ret_cros
        
        cros_corr_combined = (1 - text_weight) * cros_corr + text_weight * text_spt_corr
        cros_corr_combined = cros_corr_combined.mean(dim=1, keepdim=True)

        self_corr_combined = (1 - text_weight) * self_corr + text_weight * text_qry_corr
        self_corr_combined = self_corr_combined.mean(dim=1, keepdim=True)

        self_corr = F.interpolate(self_corr_combined, resize, mode='bilinear', align_corners=True).squeeze(1)
        cros_corr = F.interpolate(cros_corr_combined, resize, mode='bilinear', align_corners=True).squeeze(1)
        p_self_corr = F.interpolate(self_corr_combined, resize, mode='bilinear', align_corners=True).squeeze(1)

        ''' Using spatially normalized score for top-k activations'''
        '''
        score_cros = cros_corr
        score_cros = score_cros - score_cros.flatten(1).mean(dim=1, keepdim=True).unsqueeze(-1)
        score_cros = score_cros / (score_cros.flatten(1).std(dim=1, keepdim=True).unsqueeze(-1) + 1e-6)
        flat_cros = score_cros.view(B, -1)

        score_self = self_corr
        score_self = score_self - score_self.flatten(1).mean(dim=1, keepdim=True).unsqueeze(-1)
        score_self = score_self / (score_self.flatten(1).std(dim=1, keepdim=True).unsqueeze(-1) + 1e-6)
        flat_self = score_self.view(B, -1)
        '''
        points_self_ret = (p_self_corr + 1.) * .5
        cros_corr_ret = (cros_corr + 1.) * .5  # [-1, 1] -> [0, 1]
        self_corr_ret = (self_corr + 1.) * .5  # [-1, 1] -> [0, 1]

        ret_self = (self_corr_ret > thr).float()
        ret_cros = (cros_corr_ret > thr).float()

        flat_cros = cros_corr_ret.view(B, -1)
        flat_self = points_self_ret.view(B, -1)

        ret_cros[class_gt.squeeze(-1) == False] = 0.
        
        vals_c, idx_c = torch.topk(flat_cros, k=3, dim=1)
        vals_s, idx_s = torch.topk(flat_self, k=1, dim=1)

        n = 3
        p = 2
        pos_points = []
        #pos_c_points = []
        neg_points = []
        neg_c_points = []
        
        for b in range(ret_self.shape[0]):
            neg_y, neg_x = torch.where(ret_self[b] == 0.)
            neg_cy, neg_cx = torch.where(ret_cros[b] == 0.)

            pos_sy, pos_sx = torch.where(ret_self[b] > thr)
            #pos_cy, pos_cx = torch.where(ret_cros[b] > thr)

            nx_s = neg_x.numel()
            nx_c = neg_cx.numel()

            px_s = pos_sx.numel()
            #px_c = pos_cx.numel()

            perm_ns = torch.randperm(nx_s)[:n]
            perm_nc = torch.randperm(nx_c)[:n]

            perm_ps = torch.randperm(px_s)[:p]
            #perm_pc = torch.randperm(px_c)[:n]

            neg_points.append(torch.stack([neg_x[perm_ns], neg_y[perm_ns]], dim=-1))
            neg_c_points.append(torch.stack([neg_cx[perm_nc], neg_cy[perm_nc]], dim=-1))

            pos_points.append(torch.stack([pos_sx[perm_ps], pos_sy[perm_ps]], dim=-1))
            #pos_c_points.append(torch.stack([pos_cx[perm_pc], pos_cy[perm_pc]], dim=-1))

        neg_points = torch.stack(neg_points, dim=0)
        neg_c_points = torch.stack(neg_c_points, dim=0)

        pos_points = torch.stack(pos_points, dim=0)
        #pos_c_points = torch.stack(pos_c_points, dim=0)

        Hc, Wc = resize
        y_c = idx_c // Wc
        x_c = idx_c % Wc

        Hs, Ws = resize
        y_s = idx_s // Ws
        x_s = idx_s % Ws

        point_s = torch.stack([x_s, y_s], dim=-1)
        points = torch.cat([point_s, pos_points, neg_points], dim=1)

        point_c = torch.stack([x_c, y_c], dim=-1)
        points_c = torch.cat([point_c, neg_c_points], dim=1)
        
        return ret_self, points, ret_cros, points_c
        
    @torch.enable_grad()
    def generate_pseudo_mask_qkey(self, qry_qkv, spt_qkv, class_gt, text_feat=None, resize=(800, 800), thr=0.4, text_weight=0.2, spt_features=None, qry_features=None, enrich_cls=False):
        # 0-th token: cls token
        # 1-HW token: img token
        # qry_qkv [qkv, batch, head, (1+HW), dim]
        # text_feats [batch, hdim]
        _, B, N, L, C = qry_qkv.shape
        spt_cls = spt_qkv[0, :, :, 0, :]
        spt_key = spt_qkv[1, :, :, 1:, :]
        qry_key = qry_qkv[1, :, :, 1:, :]
        qry_qry = qry_qkv[0, :, :, 1:, :]
        spt_qry = spt_qkv[0, :, :, 1:, :]
        spt_kcls = spt_qkv[1, :, :, 0, :]

        h = w = int(self.imgsize // 16)
        ch = int(C // self.nhead)

        qry_key = rearrange(qry_key, 'b n (h w) c -> b n h w c', h=h, w=w)
        spt_key = rearrange(spt_key, 'b n (h w) c -> b n h w c', h=h, w=w)
        spt_qry = rearrange(spt_qry, 'b n (h w) c -> b n h w c', h=h, w=w)
        qry_qry = rearrange(qry_qry, 'b n (h w) c -> b n h w c', h=h, w=w)

        qry_key = F.normalize(qry_key, p=2, dim=-1)
        spt_key = F.normalize(spt_key, p=2, dim=-1)
        spt_cls = F.normalize(spt_cls, p=2, dim=-1)
        spt_kcls = F.normalize(spt_kcls, p=2, dim=-1)
        spt_qry = F.normalize(spt_qry, p=2, dim=-1)
        qry_qry = F.normalize(qry_qry, p=2, dim=-1)

        if self.use_text:
            text_feat = F.normalize(text_feat, p=2, dim=-1)
            text_feat = rearrange(text_feat, 'b (s n) -> b n s', n=self.nhead)

            #text_qry_corr = torch.einsum('b n h w c, b n c -> b n h w', qry_qry, text_feat).mean(dim=1, keepdim=True)
            text_spt_corr = torch.einsum('b n h w c, b n s -> b n h w', spt_key, text_feat).mean(dim=1, keepdim=True)
            #cls_enriched = torch.einsum('b n c, b n c -> b n c', spt_cls, text_feat)
            '''Next, enrich spt CLS token with text token
            cls_enriched = torch.einsum('b n c, b n c -> b n c', spt_cls, text_feat)
            use it for cross and self correlation q-q https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06346.pdf
            '''

        cros_corr = torch.einsum('b n h w c, b n c -> b n h w', qry_key, spt_kcls)
        self_corr = torch.einsum('b n h w c, b n c -> b n h w', spt_key, spt_kcls)
        
        if enrich_cls or (not self.use_text and not self.use_sam):
            self_corr = self_corr.mean(dim=1, keepdim=True)
            cros_corr = cros_corr.mean(dim=1, keepdim=True)

            self_corr = F.interpolate(self_corr, resize, mode='bilinear', align_corners=True).squeeze(1)
            cros_corr = F.interpolate(cros_corr, resize, mode='bilinear', align_corners=True).squeeze(1)

            cros_corr_ret = (cros_corr + 1.) * .5  # [-1, 1] -> [0, 1]
            self_corr_ret = (self_corr + 1.) * .5  # [-1, 1] -> [0, 1]

            ret_self = (self_corr_ret > thr).float()
            ret_cros = (cros_corr_ret > thr).float()

            ret_cros[class_gt.squeeze(-1) == False] = 0.

            return ret_self, ret_cros
        
        cros_corr_combined = (1 - text_weight) * cros_corr + text_weight * text_spt_corr
        cros_corr_combined = cros_corr_combined.mean(dim=1, keepdim=True)

        self_corr_combined = (1 - text_weight) * self_corr + text_weight * text_spt_corr
        self_corr_combined = self_corr_combined.mean(dim=1, keepdim=True)

        self_corr = F.interpolate(self_corr_combined, resize, mode='bilinear', align_corners=True).squeeze(1)
        cros_corr = F.interpolate(cros_corr_combined, resize, mode='bilinear', align_corners=True).squeeze(1)
        p_self_corr = F.interpolate(self_corr_combined, resize, mode='bilinear', align_corners=True).squeeze(1)

        ''' Using spatially normalized score for top-k activations'''
        '''
        score_cros = cros_corr
        score_cros = score_cros - score_cros.flatten(1).mean(dim=1, keepdim=True).unsqueeze(-1)
        score_cros = score_cros / (score_cros.flatten(1).std(dim=1, keepdim=True).unsqueeze(-1) + 1e-6)
        flat_cros = score_cros.view(B, -1)

        score_self = self_corr
        score_self = score_self - score_self.flatten(1).mean(dim=1, keepdim=True).unsqueeze(-1)
        score_self = score_self / (score_self.flatten(1).std(dim=1, keepdim=True).unsqueeze(-1) + 1e-6)
        flat_self = score_self.view(B, -1)
        '''
        points_self_ret = (p_self_corr + 1.) * .5
        cros_corr_ret = (cros_corr + 1.) * .5  # [-1, 1] -> [0, 1]
        self_corr_ret = (self_corr + 1.) * .5  # [-1, 1] -> [0, 1]

        ret_self = (self_corr_ret > thr).float()
        ret_cros = (cros_corr_ret > thr).float()

        flat_cros = cros_corr_ret.view(B, -1)
        flat_self = points_self_ret.view(B, -1)

        ret_cros[class_gt.squeeze(-1) == False] = 0.

        if (self.args.use_text and not self.args.use_sam) or self.args.eval:
            return ret_self, None, ret_cros, None
        
        vals_c, idx_c = torch.topk(flat_cros, k=3, dim=1)
        vals_s, idx_s = torch.topk(flat_self, k=1, dim=1)

        #print(idx_c.shape)

        n = 3
        p = 2
        pos_points = []
        #pos_c_points = []
        neg_points = []
        neg_c_points = []
        
        for b in range(ret_self.shape[0]):

            neg_y, neg_x = torch.where(ret_self[b] == 0.)
            neg_cy, neg_cx = torch.where(ret_cros[b] == 0.)

            pos_sy, pos_sx = torch.where(ret_self[b] > thr)
            #pos_cy, pos_cx = torch.where(ret_cros[b] > thr)

            nx_s = neg_x.numel()
            nx_c = neg_cx.numel()

            px_s = pos_sx.numel()
            #px_c = pos_cx.numel()

            perm_ns = torch.randperm(nx_s)[:n]
            perm_nc = torch.randperm(nx_c)[:n]

            perm_ps = torch.randperm(px_s)[:p]
            #perm_pc = torch.randperm(px_c)[:n]

            neg_points.append(torch.stack([neg_x[perm_ns], neg_y[perm_ns]], dim=-1))
            neg_c_points.append(torch.stack([neg_cx[perm_nc], neg_cy[perm_nc]], dim=-1))

            pos_points.append(torch.stack([pos_sx[perm_ps], pos_sy[perm_ps]], dim=-1))
            #pos_c_points.append(torch.stack([pos_cx[perm_pc], pos_cy[perm_pc]], dim=-1))

        neg_points = torch.stack(neg_points, dim=0)
        neg_c_points = torch.stack(neg_c_points, dim=0)

        pos_points = torch.stack(pos_points, dim=0)
        #pos_c_points = torch.stack(pos_c_points, dim=0)

        Hc, Wc = resize
        y_c = idx_c // Wc
        x_c = idx_c % Wc

        Hs, Ws = resize
        y_s = idx_s // Ws
        x_s = idx_s % Ws

        point_s = torch.stack([x_s, y_s], dim=-1)
        #print("point_s:", point_s.shape)
        #print("pos_points:", pos_points.shape)
        #print("neg_points:", neg_points.shape)
        points = torch.cat([point_s, pos_points, neg_points], dim=1)

        point_c = torch.stack([x_c, y_c], dim=-1)
        points_c = torch.cat([point_c, neg_c_points], dim=1)
        
        return ret_self, points, ret_cros, points_c

    def attention_rollout(self, attention_maps, head_fusion=None, discard_ratio=0.9, fg_threshold=0.5, text_feat=None, text_weight=False, attn_values=None):
        B = attention_maps[0].shape[0]
        nhead = attention_maps[0].shape[1]
        device = attention_maps[0].device

        T = attention_maps[0].shape[-1]

        attn_last = attention_maps[-1].detach()
        cls_attn_per_head = attn_last[:, :, 0, 1:]

        rollout = torch.eye(T, device=device).unsqueeze(0).expand(B, -1, -1).clone()

        if head_fusion is None:    
            if text_feat is not None and attn_values is not None:
                
                v_patches = attn_values[:, :, 1:, :]
                v_mean = v_patches.mean(dim=2)

                vision_dim = v_mean.shape[-1]*nhead
                head_dim = v_mean.shape[-1]

                vp = self.backbone.vision_projection
                vp_per_head = vp.reshape(nhead, head_dim, vp.shape[-1])

                v_proj = torch.einsum('b n d, n d c -> b n c', v_mean, vp_per_head)
                v_proj = F.normalize(v_proj, p=2, dim=-1)

                expanded_text = text_feat.unsqueeze(1)
                head_text_sim = (v_proj * expanded_text).sum(dim=-1)

                head_w = head_text_sim.clamp(min=0).softmax(dim=-1)
                all_neg = (head_text_sim <= 0).all(dim=-1, keepdim=True)
                uniform = torch.ones_like(head_w) / nhead
                head_w = torch.where(all_neg, uniform, head_w)
                
                
                '''clip_dim = text_feat.shape[-1]
                head_dim = clip_dim // nhead
                if clip_dim % nhead == 0:
                    text_per_head = text_feat.reshape(B, nhead, head_dim)

                    attn_ent = -(cls_attn_per_head * (cls_attn_per_head + 1e-6).log()).sum(dim=-1)
                    focus_weight = (-attn_ent).softmax(dim=-1) #[B, nhead]

                    text_mag = text_per_head.norm(dim=-1)
                    text_w = text_mag.softmax(dim=-1)

                    head_w = (focus_weight * text_w)
                    head_w = head_w / (head_w.sum(dim=-1, keepdim=True) + 1e-6)'''
            else:
                attn_ent = -(cls_attn_per_head * (cls_attn_per_head + 1e-6).log()).sum(dim=-1)
                head_w = (-attn_ent).softmax(dim=-1)

            for layer_id, attn in enumerate(attention_maps):

                    w = head_w.unsqueeze(-1).unsqueeze(-1)
                    attn_fused = (attn * w).sum(dim=1)

                    if discard_ratio > 0:
                        flat = attn_fused.flatten(1)
                        threshold_idx = int(flat.shape[1] * discard_ratio)
                        #threshold_idx = max(1, min(threshold_idx, flat.shape[1]-1))
                        threshold_val = flat.kthvalue(threshold_idx, dim=1).values
                        threshold_val = threshold_val.unsqueeze(-1).unsqueeze(-1)

                        attn_fused = attn_fused * (attn_fused > threshold_val).float()

                        attn_fused = attn_fused + torch.eye(T, device=device).unsqueeze(0)
                        row_sum = attn_fused.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                        attn_fused = attn_fused / row_sum
                        rollout = torch.bmm(attn_fused, rollout)

        else:

            for attn in attention_maps:

                if head_fusion=='mean':
                    attn_fused = attn.mean(dim=1)
                elif head_fusion=='max':
                    attn_fused = attn.max(dim=1).values
                elif head_fusion=='min':
                    attn_fused = attn.min(dim=1).values
                else:
                    raise ValueError(f'Unknown head fusion: {head_fusion}')

                if discard_ratio > 0:
                    flat = attn_fused.flatten(1)
                    threshold_idx = int(flat.shape[1] * discard_ratio)
                    threshold_val = flat.kthvalue(threshold_idx, dim=1).values
                    threshold_val = threshold_val.unsqueeze(-1).unsqueeze(-1)

                    attn_fused = attn_fused * (attn_fused > threshold_val).float()

                row_sum = attn_fused.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                attn_fused = attn_fused / row_sum

                attn_fused = attn_fused + torch.eye(T, device=device).unsqueeze(0)

                row_sum = attn_fused.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                attn_fused = attn_fused / row_sum

                rollout = torch.bmm(attn_fused, rollout)

            cls_to_patches = rollout[:, 0, 1:]
            min_val = cls_to_patches.min(dim=-1, keepdim=True).values
            max_val = cls_to_patches.max(dim=-1, keepdim=True).values

            cls_to_patches = (cls_to_patches - min_val) / (max_val - min_val + 1e-6)

            return cls_to_patches

        cls_to_patches = rollout[:, 0, 1:]
        patch_to_patches = rollout[:, 1:, 1:]

        min_val = cls_to_patches.min(dim=-1, keepdim=True).values
        max_val = cls_to_patches.max(dim=-1, keepdim=True).values

        cls_to_patches = (cls_to_patches - min_val) / (max_val - min_val + 1e-6)
        fg_attn_list = []
        for b in range(B):
            fg_idx = (cls_to_patches[b] > fg_threshold).nonzero(as_tuple=False).squeeze(-1)
            if len(fg_idx) == 0:
                fg_idx = cls_to_patches[b].topk(5).indices
            fg_attn_b = patch_to_patches[b, fg_idx, :].mean(dim=0)
            fg_attn_list.append(fg_attn_b)
        
        fg_attn = torch.stack(fg_attn_list, dim=0)
        min_fg = fg_attn.min(dim=-1, keepdim=True).values
        max_fg = fg_attn.max(dim=-1, keepdim=True).values
        fg_attn = (fg_attn - min_fg) / (max_fg - min_fg + 1e-6)

        combined = (cls_to_patches * fg_attn).sqrt()

        return cls_to_patches, fg_attn, combined

    def rollout_to_points(self, rollout_map, img_size, patch_size=16, n_pos=5, n_neg=3, fg_threshold=0.5, erosion_kernel=7, dilation_kernel=11):
        B = rollout_map.shape[0]
        h = w = img_size // patch_size
        device = rollout_map.device

        spatial = rollout_map.reshape(B, 1, h, w)

        spatial_full = F.interpolate(spatial, size=(img_size, img_size), mode='bilinear', align_corners=False).squeeze(1)

        all_points = []
        all_labels = []

        for i in range(B):
            fg_map = spatial_full[i]
            fg_mask = (fg_map > fg_threshold)

            if fg_mask.sum() < n_pos:
                topk_vals, topk_idx = fg_map.flatten().topk(n_pos*4)
                fg_mask = torch.zeros_like(fg_map, dtype=torch.bool)
                rows = topk_idx // img_size
                cols = topk_idx % img_size
                fg_mask[rows, cols] = True

            pts_i, lbl_i = self.sample_point_from_mask(fg_mask, n_pos, n_neg, erosion_kernel, dilation_kernel)

            all_points.append(pts_i)
            all_labels.append(lbl_i)

        all_points = torch.stack(all_points, dim=0)
        all_labels = torch.stack(all_labels, dim=0)

        return all_points, all_labels
    
    def erode_mask(self, mask, kernel_size=5):

        pad = kernel_size // 2

        eroded = -F.max_pool2d(-mask.float().unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=pad).squeeze()

        return eroded > 0.5
    
    def dilate_mask(self, mask, kernel_size=5):
        
        pad = kernel_size // 2

        dilated = F.max_pool2d(mask.float().unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=pad).squeeze()

        return dilated > 0.5
    
    def sample_point_from_mask(self, mask, n_pos, n_neg, erosion_kernel, dilation_kernel):

        device = mask.device

        fg_confident = self.erode_mask(mask, erosion_kernel)

        fg_dilated = self.dilate_mask(mask, dilation_kernel)
        bg_confident = ~fg_confident

        def sample_region(region, n):
            coords = torch.stack(torch.where(region)).T
            if len(coords) == 0:
                coords = torch.stack(torch.where(mask > 0.5)).T

            if len(coords) == 0:
                H, W = mask.shape
                return torch.tensor([[H // 2, W // 2]], device=device).float()

            idx = torch.randperm(len(coords), device=device)[:n]
            return coords[idx].float()
        
        pos_pts = sample_region(fg_confident, n_pos)
        neg_pts = sample_region(bg_confident, n_neg)

        pos_pts = pos_pts.flip(-1)
        neg_pts = neg_pts.flip(-1)

        points = torch.cat([pos_pts, neg_pts], dim=0)
        labels = torch.cat([
            torch.ones(len(pos_pts)).to(device),
            torch.zeros(len(neg_pts)).to(device)
            ])
        
        return points, labels
    
    def class_specific_fg_map(self, rollout_map, patch_features, text_feat, alpha=0.5):
        patch_proj = patch_features @ self.backbone.vision_projection
        patch_proj = F.normalize(patch_proj, p=2, dim=-1)

        text_sim = torch.einsum('b p n, b d -> b p', patch_proj, text_feat)

        text_sim = (text_sim + 1.0) * 0.5

        fg_map = (1 - alpha) * rollout_map + alpha * text_sim

        min_val = fg_map.min(dim=-1, keepdim=True).values
        max_val = fg_map.max(dim=-1, keepdim=True).values
        fg_map = (fg_map - min_val) / (max_val - min_val + 1e-6)

        return fg_map

    
    def save_debug_image(
        self,
        image=None,          # torch.Tensor [3,H,W] o numpy [H,W,3]
        mask=None,      # torch.Tensor [H,W] opzionale
        points=None,    # torch.Tensor [N,2] opzionale
        labels=None,    # torch.Tensor [N] opzionale
        name="debug.png",
        dir=None,
        masks=None,
        from_generator=False
    ):
        # ---- sicurezza DDP ----
        if self.global_rank != 0:
            return

        # ---- image ----
        if torch.is_tensor(image):
            mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

            img = image.detach().cpu()
            img = img * std + mean
            img = img.clamp(0, 1)
            img = img.permute(1, 2, 0).numpy()
            #image = image.detach().cpu()
            #image = image.permute(1, 2, 0).numpy()

         # ---- from generator ----
        if from_generator:
            fig, ax = plt.subplots(figsize=(20,20))
            ax.imshow(img)
            self.show_anns(masks, ax)
            ax.axis('off')
            #plt.show()

            out_path = dir / name
            fig.savefig(out_path, bbox_inches="tight", dpi=150)
            plt.close(fig)

            return

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)
        ax.axis("off")

        # ---- mask ----
        if mask is not None:
            self.show_mask(mask, plt.gca())

        # ---- points ----
        if points is not None and labels is not None:
            self.show_points(points, labels, plt.gca())

        # ---- save & cleanup ----
        out_path = dir / name
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    def show_anns(self, anns, ax):
        if len(anns) == 0:
            return

        sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
        ax.set_autoscale_on(False)

        img = np.ones((
            sorted_anns[0]['segmentation'].shape[0],
            sorted_anns[0]['segmentation'].shape[1],
            4
        ))
        img[:, :, 3] = 0

        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask

        ax.imshow(img)
    
    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])

        mask = mask.detach().cpu().float().numpy()

        if mask.min() < 0.0 or mask.max() > 1.0:
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)

        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    def show_points(self, coords, labels, ax, marker_size=375):

        if torch.is_tensor(coords):
            coords = coords.detach().cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=100, edgecolor='white', linewidth=1.25)   
        
    def extract_clip_feats(self, img, return_qkv=False, return_attn=False):
        feat = self.backbone.encode_image(img, n_layers=self.layers_to_take, return_qkv=return_qkv, return_attn=return_attn) 
        return feat
    
    def modulate_corr(self, headwise_corr, spt_emb, qry_emb):

        sim = (qry_emb * spt_emb).sum(dim=-1)
        
        gate = torch.sigmoid(self.temperature * (sim * self.bias))

        gate = gate.view(-1, 1, 1, 1)

        return headwise_corr * gate, gate.squeeze()
    
    def normalize_map(self, x):
        mn = x.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        mx = x.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]

        return (x - mn) / (mx - mn + 1e-6)

    def upsample_logit_mask(self, logit_mask, batch):
        if self.training:
            spatial_size = batch['query_img'].shape[-2:]
        else:
            spatial_size = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
        return F.interpolate(logit_mask, spatial_size, mode='bilinear', align_corners=True)

    def refine_mask(self, mask, batch):
        
        qry_img = batch['query_img']
        presence = batch['query_class_presence'].flatten()
        query_pts = self.qry_pts
        labels = self.labels
        resize = (self.imgsize, self.imgsize) if self.training else (batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item())

        #print(mask.shape) [B 1 2 800 800]
        mask = F.interpolate(mask.squeeze(1), (256, 256), mode='bilinear', align_corners=False)

        masks = []
        #print(mask.shape)
        for i in range(mask.shape[0]):
            if presence[i]:
                query_img = qry_img[i, :]
                q_img = query_img.permute(1, 2, 0).detach().cpu().numpy()
                q_img = (q_img * 255).astype(np.uint8)

                self.predictor.set_image(q_img)
                
                qry_pts = query_pts[i, :].unsqueeze(0).float()
                                    
                #labels = torch.ones(qry_pts.shape[1]).unsqueeze(0).to(spt_img.device)
                #labels[:, -5:] = 0
                                    
                first_point = qry_pts[:, :1, :]
                mask_input = mask[i, :1]
                #print(mask_input.shape)
                refined_mask, scores, logits = self.predictor.predict_torch(
                                        mask_input=mask_input,
                                        point_coords=first_point,
                                        point_labels=labels[:, :1],
                                        multimask_output=False
                                    )
                
                best = scores.argmax()
                mask_to_feed = logits[:, best, :, :]

                second_point = qry_pts[:, :2, :]

                refined_mask, scores, logits = self.predictor.predict_torch(
                                        mask_input=mask_to_feed,
                                        point_coords=second_point,
                                        point_labels=labels[:, :2],
                                        multimask_output=False
                                    )
                
                best = scores.argmax()
                mask_to_feed = logits[:, best, :, :]

                third_point = qry_pts[:, :3, :]

                refined_mask, scores, logits = self.predictor.predict_torch(
                                        mask_input=mask_to_feed,
                                        point_coords=third_point,
                                        point_labels=labels[:, :3],
                                        multimask_output=False
                                    )
                
                best = scores.argmax()
                mask_to_feed = logits[:, best, :, :]

                refined_mask, scores, logits = self.predictor.predict_torch(
                                        mask_input=mask_to_feed,
                                        point_coords=qry_pts,
                                        point_labels=labels,
                                        multimask_output=False
                                    )
                
                masks.append(refined_mask.squeeze(1))
                #print(f'refined mask shape {refined_mask.shape}')
                self.predictor.reset_image()

        masks = torch.stack(masks, dim=0)
        masks = F.interpolate(masks.float(), resize, mode='nearest').squeeze(1)
        #print(f'mask to return {masks.shape}')

        return masks

    def compute_objective(self, output_cls, output_masks, gt_presence, gt_mask, pmask=None):
        
        ''' supports 1-way training 
            output_masks : [B, 2, H, W]
            
        '''

        B = gt_presence.shape[0]

        logit_cls = torch.log_softmax(output_cls, dim=2).squeeze(1)
        logit_mask = torch.log_softmax(output_masks, dim=2).squeeze(1)
        cls_loss = F.nll_loss(logit_cls, gt_presence.long().squeeze(1))
        seg_loss = F.nll_loss(logit_mask, gt_mask.long())

        if self.args.distill:
            output_2cls = output_masks.squeeze(1)
            prob = torch.softmax(output_2cls, dim=1)[:, 1]
            target = pmask.float()

            intersection = (prob * target).sum(dim=(-1, -2))
            union = prob.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2))

            dice_loss = 1 - (2 * intersection + 1e-6) / (union + 1e-6)

            weight = torch.tensor([0.3, 0.7], device=output_masks.device)
            ce = F.cross_entropy(output_2cls, pmask.long(), weight, reduction='none')

            distill_loss = ce.mean() + dice_loss.mean()
            print(f'dice loss: {dice_loss.mean().item():.4f}')
            print(f'CE loss: {ce.mean().item():.4f}')
            print(f'distillation loss: {distill_loss.item():.4f}')

        if self.distillv1:

            logits = self.logits
            
            output_2cls = output_masks.squeeze(1) #B 2 H W
            foreg_prob = torch.softmax(output_2cls, dim=1)[:, 1]
            sam_confidence = torch.sigmoid(logits)
            confidence_mask = (sam_confidence > 0.75) | (sam_confidence < 0.25)
            conf_weight = confidence_mask.float()
            target = pmask.float()
            #print(target.sum(dim=(1,2)) / 640000)
            #print("fg ratio:", target.mean().item())
            #print("conf ratio:", conf_weight.mean().item())                                 
            
            '''Weighted CE loss'''
            weight = torch.tensor([0.3, 0.7], device=output_masks.device)
            cross_ent = F.cross_entropy(output_2cls, pmask.long(), weight=weight, reduction='none')
            ce_loss = (cross_ent*conf_weight).sum() / (conf_weight.sum() + 1e-6)
            print(f'CE_loss: {ce_loss.item():.4f}')

            '''Focal loss'''
            gamma = min(2.0, self.current_epoch / 5 * 2.0)
            alpha = torch.where(pmask==1, 0.7, 0.3)
            ce = F.cross_entropy(output_2cls, pmask.long(), reduction='none')
            pt = torch.exp(-ce)
            focal = alpha * (1-pt) ** gamma * ce

            focal_loss = (focal * conf_weight).sum() / (conf_weight.sum() + 1e-6)

            print(f'focal loss: {focal_loss.item():.4f}')
            
            '''Dice loss on confident pixels'''
            p = foreg_prob * conf_weight
            t = target * conf_weight
            dice_loss = 1 - (2 * (p * t).sum((-1, -2)) + 1e-6) / (p.sum((-1, -2)) + t.sum((-1, -2)) + 1e-6)
            print(f'dice loss: {dice_loss.mean().item():.4f}')
            if self.current_epoch <= 5:
                distill_loss = ce_loss + dice_loss.mean()
            else:
                distill_loss = focal_loss + dice_loss.mean()
            print(f'dist_loss: {distill_loss.item():.4f}')

            #print(f'confident pixel ratio: {confidence_mask.float().mean():.3f}')
            #print(f'pmask-sam mean: {pmask.float().mean():.3f}')
            
        print(f"cls_loss:", cls_loss.item())
        print(f"seg_loss:", seg_loss.item())
        print("pseudo-mask mean:", gt_mask.float().mean().item())

        '''
        qry_embeds_n = F.normalize(qry_embeds, dim=-1)
        spt_keys_n = F.normalize(spt_keys_m, dim=-1)

        if support_classes.dim() > 1:
            class_ids = support_classes[: , 0].long().view(-1)
        else:
            class_ids = support_classes.long().view(-1)
        
        contrastive_loss = self.moco_queue.compute_contrastive(
            qry_embeds_n, spt_keys_n, class_ids=class_ids, gt_presence=gt_presence.squeeze(1)
        )
        '''
        '''
        pos_img_target = gt_presence.float().squeeze(-1)
        gt_mask_patched = F.interpolate(gt_mask.unsqueeze(1).float(), size=(50, 50), mode='bilinear').squeeze(1)
        pos_targets = gt_mask_patched.flatten(start_dim=1).float()
        pos_targets = torch.where(pos_targets>0.5, pos_targets, 0.)
        #pos_targets = gt_mask.flatten(start_dim=1).float()
        
        qry = qry_embeds #F.normalize(qry_embeds, dim=-1)
        spt = spt_embeds #F.normalize(spt_embeds, dim=-1)
        txt = text_embeds.unsqueeze(1).unsqueeze(2) #F.normalize(text_embeds, dim=-1).unsqueeze(1).unsqueeze(2)
        '''
        '''Patch-level contrast'''
        '''
        proto = F.normalize(spt_embeds.mean(dim=(1,2)), dim=-1)
        qry_flat = F.normalize(qry_embeds.flatten(2).transpose(1,2), dim=-1) #B, N, D

        patch_loss = 0
        for i in range(B):
            sims = qry_flat[i] @ proto.T #N, B
            pos_mask = torch.zeros(B, device=qry_embeds.device)

            if gt_presence[i] == 1:
                pos_mask[i] = 1.
            targets = pos_mask.unsqueeze(0).expand_as(sims) #N, B
            patch_loss += F.binary_cross_entropy_with_logits(sims, targets)

        patch_loss /= B
        '''
        
        '''Image-level contrast'''
        '''
        qry_glob = F.normalize(qry_embeds.mean(dim=(1,2)), dim=-1)
        spt_glob = F.normalize(spt_embeds.mean(dim=(1,2)), dim=-1)
        txt_glob = F.normalize(text_embeds, dim=-1)

        global_qs = (qry_glob * spt_glob).sum(dim=-1)
        global_qt = (qry_glob * txt_glob).sum(dim=-1)

        target = gt_presence.float().squeeze(1)

        global_qs_loss = F.binary_cross_entropy_with_logits(global_qs, target)
        global_qt_loss = F.binary_cross_entropy_with_logits(global_qt, target)

        global_loss = (global_qs_loss + global_qt_loss) * 0.5
        '''
        '''InfoNCE Loss'''
        '''
        logits_qt = qry_glob @ txt_glob.T * self.temperature.exp()
        labels = torch.arange(B, device=qry_embeds.device)
        loss_qt = F.cross_entropy(logits_qt, labels)

        logits_qs = qry_glob @ spt_glob.T * self.temperature.exp()
        loss_qs = F.cross_entropy(logits_qs, labels)

        infoNCE_loss = (loss_qs + loss_qt) * 0.5  
        '''
        #in ws2v1 scale = 0.2 and seg_loss * 1 
        #in ws2v2 scale = ramp and seg decreases to 0.2
        if self.args.distill:
            scale = min(0.2, self.current_epoch / 15 * 0.2)
            if self.current_epoch <= 11:
                dim = 1.0
            elif self.current_epoch > 19:
                dim=0.2
            else:
                dim = 1 - ((self.current_epoch - 11)/10)
            print(f'distill scale is: {scale}')
            print(f'seg_loss applied at scale: {dim}')
            return cls_loss * 0.1 + distill_loss * scale + seg_loss * dim
        else:
            return cls_loss * 0.1 + seg_loss

    def predict_cls(self, output_cls):
        with torch.no_grad():
            logit_cls = torch.softmax(output_cls, dim=2)
            pred_cls = logit_cls[:, :, 1] > 0.5
        return pred_cls

    def predict_mask(self, output_masks):
        with torch.no_grad():
            logit_seg = torch.softmax(output_masks, dim=2)
            max_fg_val, max_fg_idx = logit_seg[:, :, 1].max(dim=1)
            max_fg_idx = max_fg_idx + 1  # smallest idx should be 1
            max_fg_idx[max_fg_val < 0.5] = 0  # set it as bg
            pred_seg = max_fg_idx
        return pred_seg

    def predict_cls_seg(self, batch, nshot):
        logit_mask_agg = 0
        cls_score_agg = 0
        support_imgs = batch['support_imgs'].clone()
        support_masks = batch['support_masks'].clone()

        for s_idx in range(nshot):
            batch['support_imgs'] = support_imgs[:, :, s_idx]
            batch['support_masks'] = support_masks[:, :, s_idx]
            output_cls, output_masks, _ = self.forward(batch)
            cls_score_agg += torch.softmax(output_cls, dim=2).clone()
            logit_mask_agg += torch.softmax(output_masks, dim=2).clone()

        pred_cls = self.predict_cls(cls_score_agg / float(nshot))
        pred_seg = self.predict_mask(logit_mask_agg / float(nshot))

        return pred_cls, pred_seg
    
    def sam_loss(output_mask, sam_mask):
        def dice_loss(pred, target, eps=1e-6):
            pred = torch.sigmoid(pred)
            intersection = (pred * target).sum(dim=(-1, -2))
            union = pred.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2))

            return 1 - (2*intersection+eps) / (union+eps)

        BCELoss = F.binary_cross_entropy_with_logits(output_mask, sam_mask.float())
        return BCELoss + dice_loss(output_mask, sam_mask)

    '''
    @torch.no_grad()
    def update_momentum_encoder(self):
        for p_q, p_k in zip(self.backbone.parameters(), self.momentum_backbone.parameters()):
            p_k.data = p_k.data * self.momentum + p_q.data * (1. - self.momentum)

    def on_before_zero_grad(self, optimizer):
        return self.update_momentum_encoder()
    '''
    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging

    def configure_optimizers(self):
        if not self.args.use_text:
            return torch.optim.Adam([{"params": self.learner.parameters(), "lr": self.args.lr, "weight_decay": 1e-3},
                                    #{"params": self.text_project.parameters(), "lr": self.args.lr * 10, "weight_decay": 1e-3},
                                    #{"params": self.condition_on_episode.parameters(), "lr": self.args.lr, "weight_decay": 1e-3},
                                    {"params": self.features_adapt.parameters(), "lr": self.args.lr * 10, "weight_decay": 1e-2}
                                    ])
        else:
            return torch.optim.Adam([{"params": self.learner.parameters(), "lr": self.args.lr, "weight_decay": 1e-3},
                                    {"params": self.text_project.parameters(), "lr": self.args.lr * 10, "weight_decay": 1e-3},
                                    {"params": self.condition_on_episode.parameters(), "lr": self.args.lr, "weight_decay": 1e-3},
                                    {"params": self.features_adapt.parameters(), "lr": self.args.lr * 10, "weight_decay": 1e-2}
                                    ])
    

class VisualConditionedText(nn.Module):
    #https://arxiv.org/pdf/2203.05557
    #https://arxiv.org/pdf/2112.10003
    def __init__(self, clip_dim, hidden_dim, out_dim):
        super().__init__()

        self.visual_to_text = nn.Sequential(
            nn.Linear(clip_dim*3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, spt_feat, qry_feat, text_feat):

        combined = torch.cat([text_feat, spt_feat, qry_feat], dim=-1)
        delta = self.visual_to_text(combined)

        return F.normalize(text_feat + delta * self.gate, p=2, dim=-1)
    

class DenseFeaturesAdapter(nn.Module):
    #https://openaccess.thecvf.com/content/CVPR2022/papers/Rao_DenseCLIP_Language-Guided_Dense_Prediction_With_Context-Aware_Prompting_CVPR_2022_paper.pdf
    #https://arxiv.org/pdf/2304.05653
    def __init__(self, dim, reduction):
        super().__init__()

        self.net_adapter = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, dim)
        )

        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, embeds):
        return embeds + self.gate * self.net_adapter(embeds)
    

class TextToVision(nn.Module):
    def __init__(self, text_dim, vision_dim):
        super().__init__()

        self.adapter = nn.Sequential(
            nn.Linear(text_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, vision_dim)
        )

    def forward(self, x):

        return self.adapter(x)