"""CW (Carlini & Wagner) attack implementation for PyTorch."""
import torch
import torch.optim as optim


class CW:
    def __init__(self, model, c, kappa, steps, lr):
        self.model = model
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.device = next(model.parameters()).device

    def __repr__(self):
        return f"CW(c={self.c}, kappa={self.kappa}, steps={self.steps}, lr={self.lr})"

    def __call__(self, images, labels):
        return self._cw_l2_attack(images, labels)

    def _cw_l2_attack(self, images, labels):
        images = images.to(self.device).clone().detach()
        labels = labels.to(self.device)
        batch_size = images.shape[0]

        # We need the mean and std used in the original normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        images_orig_scale = images * std + mean

        # Clamp to ensure valid [0, 1] range after potential float inaccuracies
        images_orig_scale = torch.clamp(images_orig_scale, 0, 1)

        # Initialize perturbation variable 'w' in arctanh space
        # w = arctanh(2*x - 1)  => x = 0.5 * (tanh(w) + 1)
        # Add small epsilon to avoid infinity for 0 and 1
        w = torch.atanh(torch.clamp(images_orig_scale * 2 - 1, -1 + 1e-6, 1 - 1e-6)).detach().requires_grad_(True)

        optimizer = optim.AdamW([w], lr=self.lr)
        best_adv_images_norm_scale = images.clone().detach() # Store best normalized images
        best_l2 = torch.full((batch_size,), float('inf'), device=self.device)

        self.model.eval() # Ensure model is in eval mode

        for step in range(self.steps):
            optimizer.zero_grad()

            # Transform w back to image space [0, 1]
            adv_images_orig_scale = 0.5 * (torch.tanh(w) + 1)

            # Re-normalize for the model
            adv_images_norm_scale = (adv_images_orig_scale - mean) / std

            outputs = self.model(adv_images_norm_scale)
            one_hot_labels = torch.eye(outputs.shape[1], device=self.device)[labels]

            # Calculate CW loss components
            real = torch.sum(one_hot_labels * outputs, dim=1)
            # Max logit of non-target classes
            other = torch.max((1 - one_hot_labels) * outputs - one_hot_labels * 10000, dim=1)[0]

            # CW Loss for untargeted attack: f(x') = max(0, Z(x')_t - max_{i!=t}{Z(x')_i} + kappa)
            # We want to maximize 'other' and minimize 'real' (or make 'real' smaller than 'other')
            f_loss = torch.clamp(real - other + self.kappa, min=0)

            # L2 distance loss (between original [0,1] and adversarial [0,1])
            l2_loss = torch.sum((adv_images_orig_scale - images_orig_scale).pow(2), dim=(1, 2, 3))

            # Total loss = c * f(x') + ||delta||_2^2
            # We minimize this total loss
            loss = self.c * f_loss + l2_loss

            # Sum losses over the batch for backward pass
            loss.sum().backward()
            optimizer.step()

            # Update best adversarial images found so far
            with torch.no_grad():
                # Check if the attack succeeded for each image in the batch
                pred_adv = torch.argmax(outputs, dim=1)
                attack_success = (pred_adv != labels)

                # Update images where the attack is successful *and* L2 is lower
                # Note: l2_loss is already squared L2 norm
                update_mask = (attack_success & (l2_loss < best_l2))
                if update_mask.any():
                    best_l2[update_mask] = l2_loss[update_mask]
                    best_adv_images_norm_scale[update_mask] = adv_images_norm_scale[update_mask].detach()

        # Return the best adversarial images found (in normalized scale)
        # If attack never succeeded for an image, it returns the original normalized image
        return best_adv_images_norm_scale.detach()
