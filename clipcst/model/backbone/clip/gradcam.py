
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target):
        self.model = model.eval()
        self.feature = None
        self.gradient = None
        self.forward_handler = None
        self.backward_handle = None
        self.target = target
        self._register_hooks()

    def _get_features_hook(self, module, input, output):
        if isinstance(output, tuple):
            output = output[0]

        self.feature = output[:, 1:, :].detach()

    def _get_grads_hook(self, module, input_grad, output_grad):
        grad = output_grad[0]
        if isinstance(grad, tuple):
            grad = grad[0]

        #B, N, C = grad.shape
        #H, W = int((N-1)**0.5)

        self.gradient = grad[:, 1:, :].detach()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._get_features_hook(module, input, output)
            if output.requires_grad:
                output.register_hook(self._get_grads_hook)

        self.forward_handler = self.target.register_forward_hook(forward_hook)

    def remove_hooks(self):
        if self.forward_handler:
            self.forward_handler.remove()
            self.forward_handler = None
    
    @torch.enable_grad()
    def __call__(self, inputs, text, resize=(800, 800)):
        B, _, H, W = inputs.shape
        patch_size = self.model.vision.patch_size
        grid_h = H // patch_size
        grid_w = W // patch_size

        self.model.zero_grad()
        inputs = inputs.clone().detach().requires_grad_(True)

        orig_requires_grad = {}
        for name, param in self.model.named_parameters():
            orig_requires_grad[name] = param.requires_grad
            param.requires_grad = True
        try:

            output = self.model(inputs)
            #print('Out shape', output.shape)

            patch_feat = self.feature
            patch_feat = patch_feat @ self.model.vision_projection
            patch_feat = F.normalize(patch_feat, dim=-1)
            logits = (patch_feat.mean(dim=1) * text).sum(dim=-1)
            
            target = logits.sum()
            print(target.requires_grad, target.grad_fn)

            #index = torch.argmax(output, dim=1)
            #target = output[:, index].sum()
            target.backward()

            gradient = self.gradient
            print('grad shape',gradient.shape)
            weight = gradient.mean(dim=1, keepdim=True)#.mean(dim=3, keepdim=True)
            #weight = F.relu(weight)
            feature = self.feature

            cam = (feature * weight).sum(dim=-1)
            cam = F.relu(cam)

            print('Cam Shape', cam.shape)

            cam_min = cam.min(dim=1, keepdim=True)[0]
            cam_max = cam.max(dim=1, keepdim=True)[0]
            cam = (cam - cam_min) / (cam_max + 1e-6)

            cam = cam.view(B, grid_h, grid_w)
            cam = F.interpolate(cam, resize, mode='bilinear', align_corners=True)
            cam = cam.squeeze(1)

            print('Final CAM Shape', cam.shape)

            mask = (cam > 0.3).float()
        finally:
            for name, params in self.model.named_parameters():
                params.requires_grad = orig_requires_grad[name]

        self.feature = None
        self.gradient = None

        return mask
