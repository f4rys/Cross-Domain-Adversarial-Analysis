"""CW (Carlini & Wagner) attack implementation for PyTorch."""
import torch
import torch.optim as optim


class CW:
    def __init__(self, model, c, kappa, steps, lr, clip_min=None, clip_max=None):
        self.model = model
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.device = next(model.parameters()).device
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __repr__(self):
        return f"CW(c={self.c}, kappa={self.kappa}, steps={self.steps}, lr={self.lr})"

    def __call__(self, images, labels):
        return self._cw_l2_attack(images, labels)

    def _cw_l2_attack(self, images, labels):
        images = images.to(self.device).clone().detach()
        labels = labels.to(self.device)
        batch_size = images.shape[0]

        # If using custom clipping, skip normalization logic
        if self.clip_min is not None and self.clip_max is not None:
            images_orig_scale = images
        else:
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            images_orig_scale = images * std + mean
            images_orig_scale = torch.clamp(images_orig_scale, 0, 1)

        w = torch.atanh(torch.clamp(images_orig_scale * 2 - 1, -1 + 1e-6, 1 - 1e-6)).detach().requires_grad_(True)
        optimizer = optim.AdamW([w], lr=self.lr)
        best_adv_images = images.clone().detach()
        best_l2 = torch.full((batch_size,), float('inf'), device=self.device)
        self.model.eval()
        for step in range(self.steps):
            optimizer.zero_grad()
            adv_images_orig_scale = 0.5 * (torch.tanh(w) + 1)
            if self.clip_min is not None and self.clip_max is not None:
                adv_images_orig_scale = torch.clamp(adv_images_orig_scale, min=self.clip_min, max=self.clip_max)
                adv_images_norm_scale = adv_images_orig_scale
            else:
                mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
                adv_images_norm_scale = (adv_images_orig_scale - mean) / std
            outputs = self.model(adv_images_norm_scale)
            one_hot_labels = torch.eye(outputs.shape[1], device=self.device)[labels]
            real = torch.sum(one_hot_labels * outputs, dim=1)
            other = torch.max((1 - one_hot_labels) * outputs - one_hot_labels * 10000, dim=1)[0]
            f_loss = torch.clamp(real - other + self.kappa, min=0)
            l2_loss = torch.sum((adv_images_orig_scale - images_orig_scale).pow(2), dim=(1, 2, 3))
            loss = self.c * f_loss + l2_loss
            loss.sum().backward()
            optimizer.step()
            with torch.no_grad():
                pred_adv = torch.argmax(outputs, dim=1)
                attack_success = (pred_adv != labels)
                update_mask = (attack_success & (l2_loss < best_l2))
                if update_mask.any():
                    best_l2[update_mask] = l2_loss[update_mask]
                    best_adv_images[update_mask] = adv_images_norm_scale[update_mask].detach()
        return best_adv_images.detach()
