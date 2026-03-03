from functools import reduce
from operator import add

import torch
import os 
import math
import numpy as np

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
from model.backbone.segment_anything import sam_model_registry, SamPredictor
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

        if args.use_text:
            self.text_project = torch.nn.Linear(self.backbone.embed_dim, self.backbone.vision_dim) #[512->384]
            self.use_text = True

        #self.temperature = torch.nn.Parameter(torch.ones(1) * np.log(1/0.07)).to(self.device)
        
        #self.thr = 0.4
        self.sup = args.sup
        #self.backbone.eval()
        #self.queue_size = 4096

        for k, v in self.backbone.named_parameters():
            v.requires_grad = False
        self.learner = CorrelationTransformer([self.nhead * self.layers_to_take], args.way, clip_dim=self.backbone.embed_dim)
        
        if args.use_sam:
            sam_url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
            model_type = 'vit_b'

            sam_state_dict = torch.hub.load_state_dict_from_url(sam_url)

            self.sam = sam_model_registry[model_type](checkpoint=sam_state_dict)
            self.sam.to(self.device)

            for k, p in self.sam.named_parameters():
                p.requires_grad = False

            self.predictor = SamPredictor(self.sam)
            self.use_sam = True

        if args.debug:
            from pathlib import Path
            self.debug_dir_refine = Path("debug_outputs/mask_to_refine")
            self.debug_dir_refine.mkdir(parents=True, exist_ok=True)

            self.debug_dir_mask = Path("debug_outputs/mask_GradCAM")
            self.debug_dir_mask.mkdir(parents=True, exist_ok=True)

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

        spt_img = spt_img.to(memory_format=torch.channels_last)
        qry_img = qry_img.to(memory_format=torch.channels_last)
   
        B, C, H, W = qry_img.shape
        
        with torch.no_grad():

            qry_feats_extr = self.extract_clip_feats(qry_img, return_qkv=self.sup == 'pseudo')
            spt_feats_extr = self.extract_clip_feats(spt_img, return_qkv=self.sup == 'pseudo')

            text_feats = self.backbone.encode_text(tokenized_labels)
            text_feat = text_feats['embedding'] #[8, 512]

            text_feat = F.normalize(text_feat, p=2, dim=-1)

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
                
                if self.use_text:
                    with torch.enable_grad():
                        mask_s, spt_points, qry_mask, qry_points = self.generate_pseudo_mask(qry_qkv, spt_qkv, text_feat, class_gt=batch['query_class_presence'].flatten(), thr=0.4, resize=resize, text_weight=0.2, qry_img=qry_img, spt_img=spt_img)
                else:
                    spt_mask, qry_mask = self.generate_pseudo_mask(qry_qkv, spt_qkv, class_gt=batch['query_class_presence'].flatten(), thr=0.4, resize=resize)

                if self.use_sam:

                    spt_mask = []

                    #SAM EMBEDDINGS shape is [1, 256, 64, 64]

                    for i in range(B):
                        #q_img = qry_img[i, :]
                        support_img = spt_img[i, :]
                        s_img = support_img.permute(1, 2, 0).detach().cpu().numpy()
                        s_img = (s_img * 255).astype(np.uint8)

                        self.predictor.set_image(s_img)
                        mask_to_refine = mask_s[i, :].unsqueeze(0)
                        #img_w, img_h = mask_to_refine.shape[-2:]

                        #print(f'\nmask to refine has shape: {mask_to_refine.shape}')
                        
                        spt_pts = spt_points[i, :].unsqueeze(0).float()
                        
                        #print(f'feeding into sam spt_pts of shape: {spt_pts.shape}')
                        labels = torch.ones(spt_pts.shape[1]).unsqueeze(0).to(spt_img.device)
                        labels[:, -2:] = 0
                        
                        #print('These are the labels: ', labels.shape)
                        
                        #spt_pts = self.predictor.transform.apply_coords_torch(spt_pts, original_size=(img_h, img_w)) DO NOT USE
                        '''
                        if i==0:
                            print(f'Points fed into SAM are: {spt_pts.shape}')
                            print(f'first point is: {spt_pts[:, 0, :]}')
                            print(f'spt pts of {spt_pts[:, 0, :].shape}')
                            print(f'labels pts of {labels[:, 0].shape}')
                        '''
                    
                        first_point = spt_pts[:, 0, :].unsqueeze(0)
                        #print(f'first point is: {first_point.shape}')
                        

                        masks, scores, logits = self.predictor.predict_torch(
                            point_coords=first_point,
                            point_labels=labels[:, 0:1],
                            multimask_output=True
                        )

                        best = scores.argmax()
                        #print(f'best is {best}')
                        #print(f'logit of shape {logits.shape}')
                        mask_input = logits[:, best, :, :]
                        #print(f'Mask input is of shape: {masks.shape}')

                        masks, scores, logits = self.predictor.predict_torch(
                            point_coords=spt_pts,
                            point_labels=labels,
                            mask_input=mask_input,
                            multimask_output=False
                        )

                        spt_mask.append(masks)
                        
                        #masks = logits[torch.argmax(scores), :, :]

                        #print(f'mask from SAM has a score: {scores}')
                    

                    '''
                        mask_to_refine = mask_to_refine.unsqueeze(1)
                        mask_to_refine = F.interpolate(mask_to_refine, resize, mode='bilinear', align_corners=False).squeeze(1)
                        if self.training and self.global_rank == 0:
                            self.save_debug_image(
                                image=support_img,
                                mask=mask_to_refine,
                                points=spt_pts[0],
                                labels=labels[0],
                                name=f"_wSAM5neg2_epoch{self.current_epoch+i}_support.png",
                                dir=self.debug_dir_refine
                            )
                            self.save_debug_image(
                                image=support_img,
                                mask=mask_input,
                                points=spt_pts[0],
                                labels=labels[0],
                                name=f"_wSAM5neg2_epoch{self.current_epoch+i}_support.png",
                                dir=self.debug_dir_mask
                            )
                            '''
            
                        #mask = torch.from_numpy(mask[0]).to(spt_img.device)
                        #print(f'generated mask of shape: {masks.shape}')
                    
                    spt_mask = torch.stack(spt_mask, dim=0)
                    spt_mask = spt_mask.squeeze(1).squeeze(1)
                    
                if self.use_text and not self.use_sam:
                    spt_mask = mask_s

            batch['query_pmask'] = qry_mask  # used only for avg-head-pseudo-mask training
            batch['support_pmasks'] = spt_mask #.float()  # only used for vis

            qry_qkv = qry_qkv.repeat_interleave(self.args.way, dim=1)
                #print(f'final mask of shape: {spt_mask.shape}')

            qry_feats = qry_feats_extr['layer_features']
            spt_feats = spt_feats_extr['layer_features']

            qry_feats = torch.stack(qry_feats, dim=1)
            spt_feats = torch.stack(spt_feats, dim=1)
            qry_feats = qry_feats.repeat_interleave(self.args.way, dim=0)

            # [batch, nlayer, (1+HW), dim]
            B, L, T, C = spt_feats.shape

            h = w = int(self.imgsize // 16)
            ch = int(C // self.nhead)

            #print("** Query Feature size before reshaping: ", qry_feats.shape)
            #print("** Support Feature size before reshaping: ", spt_feats.shape)

            qry_feat = qry_feats.reshape(B * L, T, C)[:, 1:, :] # 1-HW token: img tokens
            spt_feat = spt_feats.reshape(B * L, T, C)[:, 1:, :] # 1-HW token: img tokens
            spt_cls = spt_feats.reshape(B * L, T, C)[:, 0, :]   # 0-th token: cls token
            #print("** Query Feature size after reshaping: ", qry_feat.shape)
            #print('** spt_feat size after reshaping: ', spt_feat.shape)
        
            qry_feat = rearrange(qry_feat, 'b p (n c) -> b n p c', n=self.nhead, c=ch)
            #print("** Query Feature size after rearrangement: ", qry_feat.shape)
            #print("** Support Feature size after reshaping: ", spt_feat.shape)

            # resize support features 50x50 -> 12x12 to reduce computation
            spt_feat = rearrange(spt_feat, 'b (h w) d -> b d h w', h=h, w=w)
            spt_feat = F.interpolate(spt_feat, (self.sptsize, self.sptsize), mode='bilinear', align_corners=True)
            spt_feat = rearrange(spt_feat, 'b (n c) h w -> b n (h w) c', n=self.nhead, c=ch)

            #print('spt_feat after rearrangement is: ', spt_feat.shape)

            spt_cls = rearrange(spt_cls, 'b (n c) -> b n 1 c', n=self.nhead, c=ch)
            spt_feat = torch.cat([spt_cls, spt_feat], dim=2)

            #print("** Query Feature size after rearrangement: ", qry_feat.shape)
            #print("** Support Feature size after reshaping: ", spt_feat.shape)

            qry_feat = F.normalize(qry_feat, p=2, dim=-1)
            spt_feat = F.normalize(spt_feat, p=2, dim=-1)
            #print("** Query Feature size after rearrangement: ", qry_feat.shape)
            #print("** Support Feature size after reshaping: ", spt_feat.shape)

            qry_to_emb = qry_feat
            spt_to_emb = spt_feat

            qry_to_emb = rearrange(qry_to_emb, '(b l) n t c -> b l t (n c)', b=B, l=self.layers_to_take) #[8, 12, 2500, 384]
            spt_to_emb = rearrange(spt_to_emb, '(b l) n t c -> b l t (n c)', b=B, l=self.layers_to_take) #[8, 12, 145, 384]
    
            qry_embeds = qry_feats_extr['embedding'] #[8, 512]
            qry_embeds = F.normalize(qry_embeds, p=2, dim=-1)

            spt_embeds = spt_feats_extr['embedding']
            spt_embeds = F.normalize(spt_embeds, p=2, dim=-1)
       
            headwise_corr = torch.einsum('b d q c, b d s c -> b d q s', qry_feat, spt_feat)
            headwise_corr = rearrange(headwise_corr, '(b l) d q s -> b (l d) q s', b=B, l=L)
        '''
        with torch.no_grad():

            spt_text_momentum = self.backbone.encode_text(tokenized_labels)
            spt_text_feats_m = spt_text_momentum['embedding']
            spt_text_feats_m = F.normalize(spt_text_feats_m, dim=-1)
        '''

        output_cls, output_masks = self.learner(headwise_corr, spt_mask, text_feat)

        # BN, 2, H, W
        output_cls = output_cls.view(-1, self.way, 2)
        output_masks = self.upsample_logit_mask(output_masks, batch)
        output_masks = output_masks.view(-1, self.way, *output_masks.shape[1:])

        #print('This is the shape of output_cls: ', output_cls.shape)
        #print('This is what actually is output_cls:', output_cls)
        #print('This is the shape of output_mask: ', output_masks.shape)

        return output_cls, output_masks #qry_feat, spt_feat, text_feat#spt_text_feats_m

    def save_debug_image(
        self,
        image,          # torch.Tensor [3,H,W] o numpy [H,W,3]
        mask=None,      # torch.Tensor [H,W] opzionale
        points=None,    # torch.Tensor [N,2] opzionale
        labels=None,    # torch.Tensor [N] opzionale
        name="debug.png",
        dir=None
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

    
    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])

        mask = mask.detach().cpu().float().numpy()
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
        
    def extract_clip_feats(self, img, return_qkv=False):
        feat = self.backbone.encode_image(img, n_layers=self.layers_to_take, return_qkv=return_qkv) 
        return feat
    
    def normalize_map(self, x):
        mn = x.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        mx = x.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]

        return (x - mn) / (mx - mn + 1e-6)

    @torch.enable_grad()
    def generate_pseudo_mask(self, qry_qkv, spt_qkv, class_gt, text_feat=None, resize=(800, 800), thr=0.4, text_weight=0.2, 
                             qry_img=None, spt_img=None):
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

        if self.use_text:
            text_feat = self.text_project(text_feat)
            text_feat = F.normalize(text_feat, p=2, dim=-1)
            text_feat = rearrange(text_feat, 'b (n d) -> b n d', n=self.nhead)

            text_qry_corr = torch.einsum('b n h w c, b n c -> b n h w', qry_key, text_feat).mean(dim=1, keepdim=True)
            text_spt_corr = torch.einsum('b n h w c, b n c -> b n h w', spt_key, text_feat).mean(dim=1, keepdim=True)

        cros_corr = torch.einsum('b n h w c, b n c -> b n h w', qry_key, spt_cls)
        self_corr = torch.einsum('b n h w c, b n c -> b n h w', spt_key, spt_cls)
        
        if not self.use_text:
            self_corr = self_corr.mean(dim=1, keepdim=True)
            cros_corr = cros_corr.mean(dim=1, keepdim=True)

            self_corr = F.interpolate(self_corr_combined, resize, mode='bilinear', align_corners=True).squeeze(1)
            cros_corr = F.interpolate(cros_corr_combined, resize, mode='bilinear', align_corners=True).squeeze(1)

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

        '''fig, ax = plt.sublplots(figsize=(8, 8))

        ax.imshow(self_corr[0].detach().cpu())
        ax.colorbar()
        name = f'self_corr mask'
        fig.savefig(self.debug_dir+name, bbox_inches="tight", dpi=150)
        plt.close(fig)
        '''

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
        
        #flat_cros = cros_corr.view(B, -1)
        #flat_self = self_corr.view(B, -1)

        #print(f'\nFlat self is of shape: {flat_self.shape}')
        #print(f'Because it was flattened a tensor of shape: {points_self_ret.shape} and not from {ret_self.shape}')

        k = 3 

        vals_c, idx_c = torch.topk(flat_cros, k=k, dim=1)
        vals_s, idx_s = torch.topk(flat_self, k=k, dim=1)

        #print(f'After topk, we obtain indices of shape: {idx_s.shape}')
        #print(f'These indices are: {idx_s}')

        n = 2
        neg_points = []
        #print(f'perhaps there is a mistake.. {ret_self.shape}')
        for b in range(ret_self.shape[0]):
            neg_x, neg_y = torch.where(ret_self[b] == 0.)
            #print(f'negative samples are in {neg_x}, {neg_y}')
            x = neg_x.numel()
            perm = torch.randperm(x)[:n]
            neg_points.append(torch.stack([neg_x[perm], neg_y[perm]], dim=-1))

        neg_points = torch.stack(neg_points, dim=0)

        #print('Neg points of shape', neg_points.shape)

        Hc, Wc = resize
        y_c = idx_c // Wc
        x_c = idx_c % Wc

        Hs, Ws = resize#self_corr.shape[-2:]
        y_s = idx_s // Ws
        x_s = idx_s % Ws

        point_s = torch.stack([x_s, y_s], dim=-1)
        #print('support points of shape: ', point_s.shape)
        points = torch.cat([point_s, neg_points], dim=1)
        point_c = torch.stack([x_c, y_c], dim=-1)
        #print(f'spt_points returned are of shape: {points.shape}')
        #print(f'spt points returned for the first image are: {points[0]}')
        '''
        labels = torch.ones(points.shape[1]).unsqueeze(0).to(spt_img.device)
        labels[:, -2:] = 0
        print(f'the labels are: {labels}')
        mask_to_refine = ret_self.unsqueeze(1)
        mask_to_refine = F.interpolate(mask_to_refine, resize, mode='bilinear', align_corners=False).squeeze(1)
        if self.training and self.global_rank == 0:
            self.save_debug_image(
            image=spt_img[0],
            mask=mask_to_refine[0],
            points=points[0],
            labels=labels[0],
            name=f"firstimage_SAM5neg2_epoch{self.current_epoch}_support.png"
            )
        '''
        return ret_self, points, ret_cros, point_c

    def upsample_logit_mask(self, logit_mask, batch):
        if self.training:
            spatial_size = batch['query_img'].shape[-2:]
        else:
            spatial_size = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
        return F.interpolate(logit_mask, spatial_size, mode='bilinear', align_corners=True)

    def compute_objective(self, output_cls, output_masks, gt_presence, gt_mask): #support_classes, spt_keys_m):
        
        ''' supports 1-way training '''
        B = gt_presence.shape[0]

        logit_cls = torch.log_softmax(output_cls, dim=2).squeeze(1)
        logit_mask = torch.log_softmax(output_masks, dim=2).squeeze(1)
        cls_loss = F.nll_loss(logit_cls, gt_presence.long().squeeze(1))
        seg_loss = F.nll_loss(logit_mask, gt_mask.long())

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
        return cls_loss * 0.1 + seg_loss #+ contrastive_loss * 0.1
        #+ (infoNCE_loss + (global_loss * patch_contrastive_loss) * 0.5) * 0.05

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
            output_cls, output_masks = self.forward(batch)
            cls_score_agg += torch.softmax(output_cls, dim=2).clone()
            logit_mask_agg += torch.softmax(output_masks, dim=2).clone()

        pred_cls = self.predict_cls(cls_score_agg / float(nshot))
        pred_seg = self.predict_mask(logit_mask_agg / float(nshot))

        return pred_cls, pred_seg
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
        return torch.optim.Adam([{"params": self.learner.parameters(), "lr": self.args.lr, "weight_decay": 1e-3},
                                 #{"params": self.text_project.parameters(), "lr": self.args.lr * .1, "weight_decay": 1e-4}
                                 ])