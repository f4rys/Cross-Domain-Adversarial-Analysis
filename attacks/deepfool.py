"""DeepFool attack implementation for PyTorch."""
import torch
import torch.autograd.functional as F


class DeepFool:
    def __init__(self, model, steps, overshoot, num_classes, clip_min=None, clip_max=None):
        self.model = model
        self.steps = steps
        self.overshoot = overshoot
        self.num_classes = num_classes
        self.device = next(model.parameters()).device
        self.clip_min = clip_min
        self.clip_max = clip_max
        if self.clip_min is None or self.clip_max is None:
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            self.min_val = (0 - mean) / std
            self.max_val = (1 - mean) / std
        else:
            self.min_val = self.clip_min
            self.max_val = self.clip_max

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
                current_pred_label = torch.argmax(outputs, dim=1).item() # Renamed to avoid clash

                # If already misclassified relative to the original prediction, stop.
                if current_pred_label != img_initial_pred:
                    break

                # Optimized Gradient Calculation using Jacobian
                def model_logits(x):
                    return self.model(x)[0] # Output shape [num_classes]

                try:
                    # Jacobian of model_logits (output: num_classes) w.r.t. img_for_grad (input: 1,C,H,W)
                    # Expected shape: (num_classes, 1, C, H, W)
                    grads_jacobian = F.jacobian(model_logits, img_for_grad, create_graph=False, strict=False)
                except Exception as e:
                    print(f"Jacobian calculation failed for image {idx} at step {step}: {e}")
                    break # Stop processing this image if Jacobian fails

                # Assuming grads_jacobian shape is (num_classes, 1, C, H, W)
                if grads_jacobian.dim() == 5 and grads_jacobian.shape[1] == 1:
                    grads = grads_jacobian.squeeze(1) # Shape: (num_classes, C, H, W)
                else:
                    # Fallback or error for unexpected shape
                    # This replaces the more complex if/elif structure previously
                    print(f"Unexpected Jacobian shape: {grads_jacobian.shape} for image {idx} at step {step}. Expected (num_classes, 1, C, H, W).")
                    break

                # Get logit and gradient for the current predicted class
                f_k_all = outputs[0]  # Logits for all classes, Shape [num_classes]
                
                f_pred_val = f_k_all[current_pred_label].detach() # Scalar tensor, logit of current prediction
                w_pred_val = grads[current_pred_label]      # Gradient for current prediction, Shape [C, H, W]

                # Calculate differences for all k (vectorized)
                f_diffs = f_k_all.detach() - f_pred_val  # Shape [num_classes]
                # Unsqueeze w_pred_val to enable broadcasting with grads (w_k_all)
                w_diffs = grads - w_pred_val.unsqueeze(0) # Shape [num_classes, C, H, W]

                # Calculate L2 norm squared for all w_diffs (sum over C, H, W dimensions)
                # Add small epsilon for numerical stability
                w_diffs_norm_sq = torch.sum(w_diffs.pow(2), dim=(1, 2, 3)) + 1e-9 # Shape [num_classes]
                
                # Avoid division by zero if w_diffs_norm_sq is zero (e.g. if w_diff is zero)
                # This can happen if grads for a class k are identical to grads for current_pred_label
                # or if a gradient is zero.
                # Set pert_ratios to infinity where w_diffs_norm_sq is too small (or zero)
                # This also handles cases where f_diffs is zero for a class k.
                pert_ratios = torch.abs(f_diffs) / w_diffs_norm_sq # Shape [num_classes]

                # Mask out the current predicted class by setting its ratio to infinity
                pert_ratios[current_pred_label] = float('inf')

                # Find the minimum perturbation ratio and corresponding class
                min_pert_ratio, closest_boundary_l = torch.min(pert_ratios, dim=0)

                # If no valid boundary found (e.g., all pert_ratios are inf)
                if torch.isinf(min_pert_ratio) or closest_boundary_l.item() == current_pred_label:
                    print(f"Warning: DeepFool couldn't find a valid boundary for image {idx} at step {step}.")
                    break # Stop iterating steps for this image
                
                # Get the w_diff for the closest boundary
                closest_w_diff = w_diffs[closest_boundary_l]

                # Calculate the minimal perturbation vector for this step (L2)
                # min_pert_ratio is a 0-dim tensor here
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
