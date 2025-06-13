"""
FGSM (Fast Gradient Sign Method) untargeted attack implementation for PyTorch.

This implementation follows the original FGSM algorithm (Goodfellow et al., 2014)
which uses the sign of gradients to create adversarial examples in a single step.
This is the untargeted variant that maximizes the loss for the true label.
"""
import torch
from torch import nn


class FGSM:
    def __init__(self, model, eps, clip_min=None, clip_max=None):
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.eps = eps
        self.device = next(model.parameters()).device
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __repr__(self):
        return f"FGSM Attack (eps={self.eps})"

    def __call__(self, images, labels):
        return self._fgsm_attack(images, labels)

    def _fgsm_attack(self, images, labels):
        # Keep a reference to the original images for clipping
        original_images = images.to(self.device).clone().detach()
        labels = labels.to(self.device)

        # Clone images to avoid modifying the original tensor and ensure requires_grad is set
        images_for_grad = original_images.clone().detach().requires_grad_(True)

        # Ensure model is in evaluation mode
        self.model.eval() # Correctly placed

        # Calculate loss
        outputs = self.model(images_for_grad)
        cost = self.loss_fn(outputs, labels)

        # Zero all existing gradients
        self.model.zero_grad()

        # Calculate gradients of loss w.r.t input images
        cost.backward()

        # Check if gradients were computed for the image
        if images_for_grad.grad is None:
             raise RuntimeError("Gradient for images is None. Ensure requires_grad=True is set on the input tensor.")

        # Collect the sign of the gradients
        sign_data_grad = images_for_grad.grad.sign()

        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = images_for_grad + self.eps * sign_data_grad

        # Unified clamping logic
        if self.clip_min is not None and self.clip_max is not None:
            perturbed_image = torch.clamp(perturbed_image, min=self.clip_min, max=self.clip_max)
        else:
            min_val = original_images - self.eps
            max_val = original_images + self.eps
            perturbed_image = torch.clamp(perturbed_image, min=min_val, max=max_val)

        # Return the perturbed image, detached from the computation graph
        return perturbed_image.detach()
