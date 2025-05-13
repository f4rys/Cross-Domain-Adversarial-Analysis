"""PGD (Projected Gradient Descent) attack implementation for PyTorch."""
import torch
import torch.nn as nn


class PGD:
    def __init__(self, model, eps, alpha, steps):
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = next(model.parameters()).device

    def __repr__(self):
        return f"PGD(eps={self.eps}, alpha={self.alpha}, steps={self.steps})"

    def __call__(self, images, labels):
        return self._pgd_attack(images, labels)

    def _pgd_attack(self, images, labels):
        images = images.to(self.device).clone().detach()
        labels = labels.to(self.device)

        # Keep a reference to the original images for projection
        original_images = images.clone().detach()

        adv_images = images.clone().detach() # Start from original image

        self.model.eval() # Ensure model is in eval mode

        for _ in range(self.steps):
            adv_images.requires_grad_(True)

            # Calculate loss and gradients using self.model
            outputs = self.model(adv_images)
            cost = self.loss_fn(outputs, labels)

            # Zero gradients for self.model
            self.model.zero_grad()
            cost.backward()

            # Check if gradients were computed
            if adv_images.grad is None:
                # If no gradient, it might mean the loss didn't depend on the input
                # or the model part processing the input didn't run.
                # In this context, it likely means the image is already maximally wrong
                # or something unexpected happened. We can stop iterating for this image/batch.
                # print(f"Warning: Gradient for adv_images is None in PGD step. Stopping iteration.") # Optional warning
                break # Exit the loop for this batch

            # Collect the sign of the gradients
            sign_data_grad = adv_images.grad.sign()

            # Update adversarial images (gradient ascent step)
            # Detach adv_images before the update to prevent interference with gradient calculation in the next step
            adv_images = adv_images.detach() + self.alpha * sign_data_grad

            # Project back into the epsilon-ball around the original image
            # Calculate perturbation relative to original
            perturbation = adv_images - original_images

            # Clip perturbation to [-eps, eps] (L-infinity norm)
            perturbation = torch.clamp(perturbation, -self.eps, self.eps)

            # Apply clipped perturbation to original image to ensure it stays within the L-infinity ball
            adv_images = original_images + perturbation

            # Optional: Clip final image values to be valid (e.g., if inputs were [0,1] before normalization)
            # This depends on whether the original 'images' input to the function are expected
            # to be in a specific range (like [0,1]) or are already normalized.
            # Since the notebook uses normalized images, clipping based on normalized bounds might be needed
            # if the perturbation pushes values outside the valid normalized range.
            # However, the standard PGD often clips only the perturbation magnitude (as done above).
            # If clipping the final image is desired:
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            min_val = (0 - mean) / std
            max_val = (1 - mean) / std
            adv_images = torch.clamp(adv_images, min=min_val, max=max_val)


        # Return the final perturbed image, detached from the computation graph
        return adv_images.detach()
