
import torch
import torch.nn.functional as F
from einops import rearrange


class GradCAM:
    def __init__(self, model):
        self.model = model.eval()
        #self.feature = None
        #self.gradient = None
        self.tokens = None
        #self.forward_handler = None
        #self.backward_handle = None
        self.hook_handle = None
        #self.target = target
        self._register_hooks()

    def _get_features_hook(self, module, input, output):
        if isinstance(output, tuple):
            output = output[0]

        self.feature = output[:, 1:, :]

    def _get_grads_hook(self, module, input_grad, output_grad):
        grad = output_grad[0]
        if isinstance(grad, tuple):
            grad = grad[0]

        #B, N, C = grad.shape
        #H, W = int((N-1)**0.5)

        self.gradient = grad[:, 1:, :]

    def _register_hooks(self):
        def save_tokens(module, input, output):
            if isinstance(output, tuple):
                output = output[0]

            if not output.requires_grad:
                output.requires_grad_(True)

            self.tokens = output
            self.tokens.retain_grad()

        return self.model.vision.block[-1].register_forward_hook(save_tokens)
        
        '''
        self.forward_handler = self.target.register_forward_hook(
            lambda m, i, o: self._get_features_hook(m, i, o)
        )
        self.backward_handle = self.target.register_full_backward_hook(
            lambda m, gin, gout: self._get_grads_hook(m, gin, gout)
        )
        '''
        '''
        def forward_hook(module, input, output):
            self._get_features_hook(module, input, output)
            if output.requires_grad:
                output.register_hook(self._get_grads_hook)

        self.forward_handler = self.target.register_forward_hook(forward_hook)
        '''

    def remove_hooks(self):
        if self.forward_handler:
            self.forward_handler.remove()
            self.forward_handler = None
    
    @torch.enable_grad()
    def __call__(self, inputs, text, resize=(800, 800)):

        hook = self._register_hooks()
        B, _, H, W = inputs.shape
        patch_size = self.model.vision.patch_size
        grid_h = H // patch_size
        grid_w = W // patch_size

        #self.model.zero_grad()
        inputs = inputs.clone().detach().requires_grad_(True)

        orig_requires_grad = {}
        for name, param in self.model.named_parameters():
            orig_requires_grad[name] = param.requires_grad
            param.requires_grad = True
        try:

            output = self.model(inputs)
            #print('Out shape', output.shape)
            tokens = self.tokens
            patch_tokens = tokens[:, 1:, :]

            #patch_feat = self.feature v1
            patch_feat_raw = patch_tokens @ self.model.vision_projection
            patch_feat = F.normalize(patch_feat_raw, dim=-1)
            #logits = (patch_feat * text.unsqueeze(1)).sum(dim=-1) v2
            #logits = (patch_feat.mean(dim=1) * text).sum(dim=-1) #v1
            logits = (patch_feat @ text.T)
            #target = logits.mean()
            #target = logits.sum() #v1
            #print(target.requires_grad, target.grad_fn)

            #index = torch.argmax(output, dim=1)
            target = logits.max(dim=1)[0].sum()
            target.backward()

            grad = self.tokens.grad[:, 1:, :]
            grad = grad @ self.model.vision_projection

            #gradient = self.gradient v1
            #print('grad shape',gradient.shape) in v1 is dim=1
            weight = grad.mean(dim=(1), keepdim=True)#.mean(dim=3, keepdim=True)
            #weight = F.relu(grad).mean(dim=1, keepdim=True)
            #feature = self.feature v1

            #cam = (feature * weight).sum(dim=-1)v1
            cam = (patch_feat_raw * weight).sum(dim=-1)
            cam = F.relu(cam)

            #print('Cam Shape', cam.shape)

            #cam_min = cam.min(dim=1, keepdim=True)[0] v1
            #cam_max = cam.max(dim=1, keepdim=True)[0] v1
            cam_min = cam.min(dim=(-2), keepdim=True)[0].min(dim=(-1), keepdim=True)[0]
            cam_max = cam.max(dim=(-2), keepdim=True)[0].min(dim=(-1), keepdim=True)[0]
            cam = (cam - cam_min) / (cam_max + 1e-6)

            #cam = cam.view(B, grid_h, grid_w)
            cam = rearrange(cam, 'b (h w) -> b h w', h=grid_h, w=grid_w).unsqueeze(1)
            #print(f'cam shape: {cam.shape}')
            cam = F.interpolate(cam, resize, mode='bilinear', align_corners=False)
            cam = cam.squeeze(1)

            #print('Final CAM Shape', cam.shape)
            print(
                "CAM stats:",
                cam.min().item(),
                cam.max().item(),
                cam.mean().item()
            )

            mask = (cam > 0.3).float()
            cam = cam.detach()
        finally:
            
            for name, params in self.model.named_parameters():
                params.requires_grad = orig_requires_grad[name]
        hook.remove()
        self.tokens = None
        self.feature = None
        self.gradient = None

        return cam, mask.detach()
