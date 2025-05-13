"""DeepFool attack implementation for PyTorch."""
import torch
import torch.autograd.functional as F


class DeepFool:
    def __init__(self, model, steps, overshoot, num_classes):
        self.model = model
        self.steps = steps
        self.overshoot = overshoot
        self.num_classes = num_classes
        self.device = next(model.parameters()).device

        # Calculate normalized bounds based on ImageNet normalization
        # These are the min/max values in the normalized space corresponding to [0, 1] pixel values
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.min_val = (0 - mean) / std
        self.max_val = (1 - mean) / std


    def __repr__(self):
        return f"DeepFool(steps={self.steps}, overshoot={self.overshoot}, num_classes={self.num_classes})"

    def __call__(self, images, labels=None):
        # Note: labels are not used in untargeted DeepFool
        # but kept for consistency with the evaluation loop structure.
        return self._deepfool_attack(images)

    def _deepfool_attack(self, images):
        images = images.to(self.device).clone().detach()
        batch_size = images.shape[0]

        # Store original images
        original_images = images.clone().detach()
        # This will store the final adversarial images for the batch
        final_adv_images = images.clone().detach()

        self.model.eval()

        # Get initial predictions (needed to know when attack succeeds)
        with torch.no_grad():
            initial_outputs = self.model(images)
            initial_preds = torch.argmax(initial_outputs, dim=1)

        # Loop over images in the batch
        for idx in range(batch_size):
            # Process one image [1, C, H, W]
            # Start each image's attack from its original state
            img = original_images[idx:idx+1].clone().detach()
            img_initial_pred = initial_preds[idx].item()

            # Perturbation accumulated for *this* image across steps
            current_img_pert = torch.zeros_like(img, device=self.device)

            for step in range(self.steps):
                # Need gradients for the current perturbed image
                # Create a version of the image for this step's gradient calculation
                img_for_grad = img.clone().detach().requires_grad_(True)

                outputs = self.model(img_for_grad) # Shape [1, num_classes]
                current_pred = torch.argmax(outputs, dim=1).item()

                # If already misclassified relative to the original prediction, stop.
                if current_pred != img_initial_pred:
                    break

                # Optimized Gradient Calculation using Jacobian
                def model_logits(x):
                    return self.model(x)[0] # Output shape [num_classes]

                try:
                    grads_jacobian = F.jacobian(model_logits, img_for_grad, create_graph=False, strict=False)
                except Exception as e:
                    print(f"Jacobian calculation failed for image {idx} at step {step}: {e}")
                    break # Stop processing this image if Jacobian fails

                # Squeeze the batch dimension potentially added by jacobian: (num_classes, C, H, W)
                # Handle potential extra dimensions if jacobian behaves differently
                if grads_jacobian.dim() > 4: # e.g., (1, num_classes, C, H, W)
                    grads = grads_jacobian.squeeze(dim=0).squeeze(dim=1)
                elif grads_jacobian.dim() == 4: # (num_classes, C, H, W)
                     grads = grads_jacobian.squeeze(dim=1) # Should not happen with squeeze(0) above, but safer
                else:
                    print(f"Unexpected Jacobian shape: {grads_jacobian.shape} for image {idx} at step {step}")
                    break


                # Get logit and gradient for the current predicted class (still the initial pred at this stage)
                f_pred = outputs[0, current_pred].detach().item() # Logit of current (initial) prediction
                w_pred = grads[current_pred] # Gradient for current (initial) prediction

                min_pert_ratio = float('inf')
                closest_boundary_l = -1
                closest_w_diff = None # Initialize closest_w_diff

                # Find the closest class boundary (minimum perturbation ratio)
                for k in range(self.num_classes):
                    if k == current_pred: # Compare against the current prediction
                        continue

                    f_k = outputs[0, k].detach().item() # Logit of other class k
                    w_k = grads[k] # Gradient for other class k

                    # Difference in logits and gradients
                    f_diff = f_k - f_pred
                    w_diff = w_k - w_pred # shape [C, H, W]

                    # Use sum of squares for squared L2 norm
                    # Add small epsilon for numerical stability
                    w_diff_norm_sq = torch.sum(w_diff**2) + 1e-9

                    # Perturbation ratio: abs(f_diff) / ||w_diff||_2^2
                    pert_ratio_k = abs(f_diff) / w_diff_norm_sq

                    if pert_ratio_k < min_pert_ratio:
                        min_pert_ratio = pert_ratio_k
                        closest_boundary_l = k
                        closest_w_diff = w_diff # Store the w_diff for the closest boundary

                # If no boundary found (e.g., gradients were zero or identical)
                if closest_boundary_l == -1 or closest_w_diff is None:
                    print(f"Warning: DeepFool couldn't find a boundary for image {idx} at step {step}.")
                    break # Stop iterating steps for this image

                # Calculate the minimal perturbation vector for this step (L2)
                r_i = min_pert_ratio * closest_w_diff

                # Accumulate perturbation for this image
                current_img_pert += r_i

                # Update the image *for the next iteration* by applying the accumulated perturbation
                img = (original_images[idx:idx+1] + current_img_pert).detach()

                # Check if the classification has changed *after* this step's perturbation
                with torch.no_grad():
                    check_outputs = self.model(img)
                    check_pred = torch.argmax(check_outputs, dim=1).item()

                if check_pred != img_initial_pred:
                    break

            # After loop finishes (max steps reached or boundary crossed)
            # Apply overshoot to the total accumulated perturbation
            final_pert = (1 + self.overshoot) * current_img_pert
            # Calculate the final adversarial image
            adv_img = original_images[idx:idx+1] + final_pert

            # Clip the final adversarial image to the valid normalized range
            # This corresponds to clipping pixel values to [0, 1]
            final_adv_images[idx] = torch.clamp(adv_img, min=self.min_val, max=self.max_val).detach().squeeze(0)


        # Return the batch of final adversarial images
        return final_adv_images
