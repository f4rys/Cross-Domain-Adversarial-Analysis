import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.helpers import inv_tensor_transform, get_class_name


def plot_accuracy_vs_epsilon(epsilons, accuracies, title='Accuracy vs. Epsilon'):
    """Plots accuracy vs. epsilon."""
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, accuracies, marker='o')
    plt.title(title)
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy (%)')
    plt.xticks(epsilons, rotation=45)
    plt.grid(True)
    plt.show()

def visualize_attack_examples_epsilon_sweep(num_samples_to_show, epsilons, attack_results, clean_data, imagenet_classes, attack_name="Attack"):
    """Visualizes clean and adversarial examples for an epsilon sweep."""
    fig, axs = plt.subplots(num_samples_to_show, len(epsilons) + 1, figsize=(18, 3 * num_samples_to_show))
    fig.suptitle(f"{attack_name} Examples vs. Epsilon", fontsize=16)

    for i in range(num_samples_to_show):
        if i >= len(clean_data['images']): continue # Skip if not enough images

        # Display clean image
        clean_img_tensor = clean_data['images'][i]
        clean_pred_idx = clean_data['preds'][i]
        true_label_idx = clean_data['labels'][i]
        clean_conf = clean_data['confidences'][i]
        clean_img_pil = inv_tensor_transform(clean_img_tensor)

        ax = axs[i, 0]
        ax.imshow(clean_img_pil)
        title_clean = f"Clean\nPred: {get_class_name(clean_pred_idx, imagenet_classes)} ({clean_conf:.2f})\nTrue: {get_class_name(true_label_idx, imagenet_classes)}"
        ax.set_title(title_clean, color=("green" if clean_pred_idx == true_label_idx else "red"), fontsize=8)
        ax.axis('off')

        # Display adversarial images for each epsilon
        for j, eps in enumerate(epsilons):
            if eps not in attack_results or i >= len(attack_results[eps]['images']):
                ax = axs[i, j + 1]
                ax.text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.set_title(f"{attack_name} (eps={eps})\nPred: N/A", fontsize=8)
                ax.axis('off')
                continue

            adv_img_tensor = attack_results[eps]['images'][i]
            adv_pred_idx = attack_results[eps]['preds'][i]
            adv_conf = attack_results[eps]['confidences'][i]
            adv_img_pil = inv_tensor_transform(adv_img_tensor)

            ax = axs[i, j + 1]
            ax.imshow(adv_img_pil)
            title_adv = f"{attack_name} (eps={eps})\nPred: {get_class_name(adv_pred_idx, imagenet_classes)} ({adv_conf:.2f})"
            ax.set_title(title_adv, color=("green" if adv_pred_idx == true_label_idx else "red"), fontsize=8)
            ax.axis('off')

    plt.tight_layout(h_pad=3, rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_accuracy_vs_c(c_values, accuracies, title='CW Accuracy vs. C'):
    """Plots CW accuracy vs. C values."""
    plt.figure(figsize=(8, 5))
    plt.plot(c_values, accuracies, marker='o')
    plt.title(title)
    plt.xlabel('C Value')
    plt.ylabel('Accuracy (%)')
    plt.xticks(c_values)
    plt.grid(True)
    plt.show()

def visualize_attack_examples_c_sweep(num_samples_to_show, c_values, attack_results, clean_data, imagenet_classes, attack_name="Attack", params_info=""):
    """Visualizes clean and adversarial examples for a C value sweep (e.g., for CW attack)."""
    fig, axs = plt.subplots(num_samples_to_show, len(c_values) + 1, figsize=(18, 3 * num_samples_to_show))
    fig.suptitle(f"{attack_name} Examples vs. C {params_info}", fontsize=16)

    for i in range(num_samples_to_show):
        if i >= len(clean_data['images']): continue

        clean_img_tensor = clean_data['images'][i]
        clean_pred_idx = clean_data['preds'][i]
        true_label_idx = clean_data['labels'][i]
        clean_conf = clean_data['confidences'][i]
        clean_img_pil = inv_tensor_transform(clean_img_tensor)

        ax = axs[i, 0]
        ax.imshow(clean_img_pil)
        title_clean = f"Clean\nPred: {get_class_name(clean_pred_idx, imagenet_classes)} ({clean_conf:.2f})\nTrue: {get_class_name(true_label_idx, imagenet_classes)}"
        ax.set_title(title_clean, color=("green" if clean_pred_idx == true_label_idx else "red"), fontsize=8)
        ax.axis('off')

        for j, c_val in enumerate(c_values):
            ax = axs[i, j + 1] # Define ax here for consistent access
            if c_val in attack_results and i < len(attack_results[c_val]['images']):
                adv_img_tensor = attack_results[c_val]['images'][i]
                adv_pred_idx = attack_results[c_val]['preds'][i]
                adv_conf = attack_results[c_val]['confidences'][i]
                adv_img_pil = inv_tensor_transform(adv_img_tensor)
                
                ax.imshow(adv_img_pil)
                title_adv = f"{attack_name} (c={c_val})\nPred: {get_class_name(adv_pred_idx, imagenet_classes)} ({adv_conf:.2f})"
                ax.set_title(title_adv, color=("green" if adv_pred_idx == true_label_idx else "red"), fontsize=8)
            else:
                ax.text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.set_title(f"{attack_name} (c={c_val})\nPred: N/A", fontsize=8)
            ax.axis('off')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def visualize_pgd_heatmaps(pgd_results, pgd_epsilons, pgd_alphas, pgd_steps):
    """Visualizes PGD accuracy heatmaps for different step counts on a single figure."""
    num_heatmaps = len(pgd_steps)
    if num_heatmaps == 0:
        return

    max_cols = 3
    num_cols = min(num_heatmaps, max_cols)
    num_rows = (num_heatmaps + num_cols - 1) // num_cols  # Ceiling division

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows), squeeze=False)
    # squeeze=False ensures axs is always a 2D array

    for idx, steps_val in enumerate(pgd_steps):
        row = idx // num_cols
        col = idx % num_cols
        ax = axs[row, col]

        heatmap_data = pd.DataFrame(index=pgd_alphas, columns=pgd_epsilons)
        for params, result in pgd_results.items():
            eps, alpha, steps = params
            if steps == steps_val:
                heatmap_data.loc[alpha, eps] = result['accuracy']
        
        heatmap_data = heatmap_data.apply(pd.to_numeric, errors='coerce')

        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis_r", 
                    linewidths=.5, cbar_kws={'label': 'Accuracy (%)'}, ax=ax)
        ax.set_title(f'PGD Accuracy (%) for {steps_val} Steps')
        ax.set_xlabel('Epsilon')
        ax.set_ylabel('Alpha')

    # Hide any unused subplots
    for idx in range(num_heatmaps, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        fig.delaxes(axs[row, col])

    plt.tight_layout()
    plt.show()

def visualize_comparison_results(num_samples_to_show, results, attacks, imagenet_classes):
    """Visualizes a comparison of clean images vs. images perturbed by different attacks."""
    # Visualize sample results
    num_samples_to_show = min(5, results['clean']['total'])

    # Adjust figsize if titles become too long
    fig, axs = plt.subplots(num_samples_to_show, len(attacks) + 1, figsize=(18, 3.5 * num_samples_to_show))
    fig.suptitle("Clean vs Adversarial Examples (with Confidence)", fontsize=16)

    for i in range(num_samples_to_show):
        # Get clean image and info
        clean_img_tensor = results['clean']['images'][i]
        clean_pred_idx = results['clean']['preds'][i]
        true_label_idx = results['clean']['labels'][i]
        clean_conf = results['clean']['confidences'][i]

        # Denormalize for display
        clean_img_pil = inv_tensor_transform(clean_img_tensor)

        # Display clean image
        ax = axs[i, 0]
        ax.imshow(clean_img_pil)
        # Add confidence to title
        title_clean = f"Clean\nPred: {get_class_name(clean_pred_idx, imagenet_classes)} ({clean_conf:.2f})\nTrue: {get_class_name(true_label_idx, imagenet_classes)}"
        ax.set_title(title_clean,
                        color=("green" if clean_pred_idx == true_label_idx else "red"),
                        fontsize=9)
        ax.axis('off')

        # Display adversarial images
        col_idx = 1
        for attack_key, _ in attacks.items(): # Iterate over keys and values, but only use the key
            # Check if adv image exists for this index
            if i < len(results[attack_key]['images']):
                adv_img_tensor = results[attack_key]['images'][i]
                adv_pred_idx = results[attack_key]['preds'][i]
                adv_conf = results[attack_key]['confidences'][i]

                # Denormalize
                adv_img_pil = inv_tensor_transform(adv_img_tensor)

                ax = axs[i, col_idx]
                ax.imshow(adv_img_pil)
                # Add confidence to title
                title_adv = f"{attack_key}\nPred: {get_class_name(adv_pred_idx, imagenet_classes)} ({adv_conf:.2f})"
                ax.set_title(title_adv,
                                color=("green" if adv_pred_idx == true_label_idx else "red"),
                                fontsize=9)
                ax.axis('off')
            else:
                # Handle cases where attack might have failed or skipped
                ax = axs[i, col_idx]
                ax.text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.set_title(f"{attack_key}\nPred: N/A", fontsize=9)
                ax.axis('off')

            col_idx += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_accuracy_vs_deepfool_param(param_values, accuracies, param_label, title='DeepFool Accuracy vs. Parameter'):
    """Plots DeepFool accuracy vs. a given parameter (e.g., Steps or Overshoot)."""
    plt.figure(figsize=(8, 5))
    plt.plot(param_values, accuracies, marker='o')
    plt.title(title)
    plt.xlabel(param_label)
    plt.ylabel('Accuracy (%)')
    plt.xticks(param_values) # Assuming param_values are suitable for direct ticking
    plt.grid(True)
    plt.show()

def visualize_deepfool_examples_param_sweep(
    num_samples_to_show, param_values, param_name_display, 
    attack_results_subset, clean_data, imagenet_classes, 
    attack_name="DeepFool"
):
    """Visualizes clean and DeepFool adversarial examples for a parameter sweep."""
    fig, axs = plt.subplots(num_samples_to_show, len(param_values) + 1, figsize=(18, 3 * num_samples_to_show))
    fig.suptitle(f"{attack_name} Examples vs. {param_name_display}", fontsize=16)

    for i in range(num_samples_to_show):
        if i >= len(clean_data['images']): continue 

        clean_img_tensor = clean_data['images'][i]
        clean_pred_idx = clean_data['preds'][i]
        true_label_idx = clean_data['labels'][i]
        clean_conf = clean_data['confidences'][i]
        clean_img_pil = inv_tensor_transform(clean_img_tensor)

        ax = axs[i, 0]
        ax.imshow(clean_img_pil)
        title_clean = f"Clean\nPred: {get_class_name(clean_pred_idx, imagenet_classes)} ({clean_conf:.2f})\nTrue: {get_class_name(true_label_idx, imagenet_classes)}"
        ax.set_title(title_clean, color=("green" if clean_pred_idx == true_label_idx else "red"), fontsize=8)
        ax.axis('off')

        for j, p_val in enumerate(param_values):
            ax = axs[i, j + 1] 
            if p_val in attack_results_subset and i < len(attack_results_subset[p_val]['images']):
                adv_img_tensor = attack_results_subset[p_val]['images'][i]
                adv_pred_idx = attack_results_subset[p_val]['preds'][i]
                adv_conf = attack_results_subset[p_val]['confidences'][i]
                adv_img_pil = inv_tensor_transform(adv_img_tensor)
                
                ax.imshow(adv_img_pil)
                title_adv = f"{attack_name} ({param_name_display}={p_val})\nPred: {get_class_name(adv_pred_idx, imagenet_classes)} ({adv_conf:.2f})"
                ax.set_title(title_adv, color=("green" if adv_pred_idx == true_label_idx else "red"), fontsize=8)
            else:
                ax.text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.set_title(f"{attack_name} ({param_name_display}={p_val})\nPred: N/A", fontsize=8)
            ax.axis('off')
            
    plt.tight_layout(h_pad=3, rect=[0, 0.03, 1, 0.95])
    plt.show()

def visualize_deepfool_heatmaps(deepfool_results, df_steps, df_overshoot):
    """
    Visualizes DeepFool accuracy heatmaps for different steps and overshoot values.
    Rows: overshoot, Columns: steps.
    """
    if not df_steps or not df_overshoot:
        return

    heatmap_data = pd.DataFrame(index=df_overshoot, columns=df_steps)
    for params, result in deepfool_results.items():
        steps, overshoot = params
        heatmap_data.loc[overshoot, steps] = result['accuracy']

    heatmap_data = heatmap_data.apply(pd.to_numeric, errors='coerce')

    plt.figure(figsize=(6, 5))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis_r",
                linewidths=.5, cbar_kws={'label': 'Accuracy (%)'})
    plt.title('DeepFool Accuracy (%)')
    plt.xlabel('Steps')
    plt.ylabel('Overshoot')
    plt.tight_layout()
    plt.show()
