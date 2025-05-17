import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.helpers import inv_tensor_transform, get_class_name


def plot_accuracy_vs_param(param_values, accuracies, param_label, title):
    """Plots accuracy vs. a specified parameter."""
    plt.figure(figsize=(8, 5))
    plt.plot(param_values, accuracies, marker='o')
    plt.title(title)
    plt.xlabel(param_label)
    plt.ylabel('Accuracy (%)')
    # Attempt to use param_values directly for ticks, good for numeric/categorical
    try:
        plt.xticks(param_values, rotation=45)
    except TypeError:
        # Fallback for non-numeric types if direct use fails, though less ideal
        plt.xticks(range(len(param_values)), labels=[str(pv) for pv in param_values], rotation=45)
    plt.grid(True)
    plt.show()

def visualize_generic_sweep(num_samples_to_show, param_values, attack_results,
                            clean_data, imagenet_classes, attack_name,
                            param_label_in_subplot_title, param_display_name_in_suptitle,
                            suptitle_extra_info=""):
    """Helper function to visualize clean and adversarial examples for a parameter sweep."""
    fig, axs = plt.subplots(num_samples_to_show, len(param_values) + 1, figsize=(18, 3 * num_samples_to_show))
    full_suptitle = f"{attack_name} Examples vs. {param_display_name_in_suptitle}"
    if suptitle_extra_info:
        full_suptitle += f" {suptitle_extra_info}"
    fig.suptitle(full_suptitle, fontsize=16)

    for i in range(num_samples_to_show):
        if i >= len(clean_data['images']): 
            continue

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

        # Display adversarial images for each parameter value
        for j, param_val in enumerate(param_values):
            ax = axs[i, j + 1]
            if param_val not in attack_results or i >= len(attack_results[param_val]['images']):
                ax.text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.set_title(f"{attack_name} ({param_label_in_subplot_title}={param_val})\nPred: N/A", fontsize=8)
                ax.axis('off')
                continue

            adv_img_tensor = attack_results[param_val]['images'][i]
            adv_pred_idx = attack_results[param_val]['preds'][i]
            adv_conf = attack_results[param_val]['confidences'][i]
            adv_img_pil = inv_tensor_transform(adv_img_tensor)

            ax.imshow(adv_img_pil)
            title_adv = f"{attack_name} ({param_label_in_subplot_title}={param_val})\nPred: {get_class_name(adv_pred_idx, imagenet_classes)} ({adv_conf:.2f})"
            ax.set_title(title_adv, color=("green" if adv_pred_idx == true_label_idx else "red"), fontsize=8)
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def visualize_accuracy_heatmap(results_dict, primary_param_values, secondary_param_values, 
                               primary_param_name, secondary_param_name, title_prefix,
                               fixed_params_dict=None):
    """
    Visualizes accuracy heatmaps for attacks with two varying parameters, 
    potentially with other parameters fixed (multi-level heatmaps if fixed_params_dict is used).
    """
    if not primary_param_values or not secondary_param_values:
        print("Primary or secondary parameter values list is empty. Cannot generate heatmap.")
        return

    if fixed_params_dict and any(fixed_params_dict.values()):
        # Determine the parameter to iterate over for creating multiple heatmaps
        fixed_param_iter_name = list(fixed_params_dict.keys())[0]
        fixed_param_iter_values = fixed_params_dict[fixed_param_iter_name]
        num_heatmaps = len(fixed_param_iter_values)

        max_cols = 3
        num_cols = min(num_heatmaps, max_cols)
        num_rows = (num_heatmaps + num_cols - 1) // num_cols

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows), squeeze=False)

        for idx, fixed_val in enumerate(fixed_param_iter_values):
            row = idx // num_cols
            col = idx % num_cols
            ax = axs[row, col]

            heatmap_data = pd.DataFrame(index=secondary_param_values, columns=primary_param_values)
            for params_tuple, result_data in results_dict.items():
                if len(params_tuple) > 2 and params_tuple[2] == fixed_val:
                    primary_val, secondary_val = params_tuple[0], params_tuple[1]
                    if primary_val in primary_param_values and secondary_val in secondary_param_values:
                         heatmap_data.loc[secondary_val, primary_val] = result_data['accuracy']
            
            heatmap_data = heatmap_data.apply(pd.to_numeric, errors='coerce').sort_index(axis=0).sort_index(axis=1)
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis_r", 
                        linewidths=.5, cbar_kws={'label': 'Accuracy (%)'}, ax=ax)
            ax.set_title(f'{title_prefix} (%) for {fixed_param_iter_name}={fixed_val}')
            ax.set_xlabel(primary_param_name)
            ax.set_ylabel(secondary_param_name)

        for idx in range(num_heatmaps, num_rows * num_cols):
            fig.delaxes(axs[idx // num_cols, idx % num_cols])
        plt.tight_layout()

    else: # Single heatmap
        heatmap_data = pd.DataFrame(index=secondary_param_values, columns=primary_param_values)
        for params_tuple, result_data in results_dict.items():
            if len(params_tuple) == 2:
                primary_val, secondary_val = params_tuple
                if primary_val in primary_param_values and secondary_val in secondary_param_values:
                    heatmap_data.loc[secondary_val, primary_val] = result_data['accuracy']

        heatmap_data = heatmap_data.apply(pd.to_numeric, errors='coerce').sort_index(axis=0).sort_index(axis=1)
        plt.figure(figsize=(6, 5))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis_r",
                    linewidths=.5, cbar_kws={'label': 'Accuracy (%)'})
        plt.title(f'{title_prefix} (%)')
        plt.xlabel(primary_param_name)
        plt.ylabel(secondary_param_name)
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
