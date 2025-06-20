"""Utility functions for visualizing results of adversarial attacks and performance."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.helpers import get_class_name, inv_tensor_transform


def plot_accuracy_vs_param(param_values, accuracies, param_label, title):
    """Plots accuracy vs. a specified parameter."""
    plt.figure(figsize=(8, 5))
    plt.plot(param_values, accuracies, marker="o")
    plt.title(title)
    plt.xlabel(param_label)
    plt.ylabel("Accuracy (%)")
    # Attempt to use param_values directly for ticks, good for numeric/categorical
    try:
        plt.xticks(param_values, rotation=45)
    except TypeError:
        # Fallback for non-numeric types if direct use fails, though less ideal
        plt.xticks(range(len(param_values)), labels=[str(pv) for pv in param_values], rotation=45)
    plt.grid(True)
    plt.show()


def visualize_accuracy_heatmap(results_dict, primary_param_values, secondary_param_values,
                               primary_param_name, secondary_param_name, title_prefix,
                               fixed_params_dict=None):
    """
    Visualizes accuracy heatmaps for attacks with two varying parameters,
    potentially with other parameters fixed (multi-level heatmaps if fixed_params_dict is used).
    """
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
                         heatmap_data.loc[secondary_val, primary_val] = result_data["accuracy"]

            heatmap_data = heatmap_data.apply(pd.to_numeric, errors="coerce").sort_index(axis=0).sort_index(axis=1)
            # Create a DataFrame of formatted strings for annotations
            annot_data = heatmap_data.map(lambda x: f"{x:.0f}%" if pd.notnull(x) else "")
            sns.heatmap(heatmap_data, annot=annot_data, fmt="", cmap="viridis_r",
                        linewidths=.5, cbar_kws={"label": "Accuracy (%)"}, ax=ax, vmin=0, vmax=100)
            ax.set_title(f"{title_prefix} for {fixed_param_iter_name}={fixed_val}")
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
                    heatmap_data.loc[secondary_val, primary_val] = result_data["accuracy"]

        heatmap_data = heatmap_data.apply(pd.to_numeric, errors="coerce").sort_index(axis=0).sort_index(axis=1)
        # Create a DataFrame of formatted strings for annotations
        annot_data = heatmap_data.map(lambda x: f"{x:.0f}%" if pd.notnull(x) else "")
        plt.figure(figsize=(6, 5))
        sns.heatmap(heatmap_data, annot=annot_data, fmt="", cmap="viridis_r",
                    linewidths=.5, cbar_kws={"label": "Accuracy (%)"}, vmin=0, vmax=100)
        plt.title(f"{title_prefix} (%)")
        plt.xlabel(primary_param_name)
        plt.ylabel(secondary_param_name)
        plt.tight_layout()

    plt.show()


def visualize_adversarial_grid(num_samples_to_show, clean_data, column_definitions,
                               imagenet_classes, figure_suptitle):
    """
    Visualizes a grid of images: clean examples in the first column,
    and various adversarial examples in subsequent columns.
    """
    num_adv_cols = len(column_definitions)
    # Ensure axs is always a 2D array using squeeze=False
    fig, axs = plt.subplots(num_samples_to_show, 1 + num_adv_cols,
                            figsize=(3.5 * (1 + num_adv_cols), 3.5 * num_samples_to_show),
                            squeeze=False)

    fig.suptitle(figure_suptitle, fontsize=16)

    for i in range(num_samples_to_show):
        # Display clean image
        clean_img_tensor = clean_data["images"][i]
        clean_pred_idx = clean_data["preds"][i]
        true_label_idx = clean_data["labels"][i]
        clean_conf = clean_data["confidences"][i]
        clean_img_pil = inv_tensor_transform(clean_img_tensor)

        ax_clean = axs[i, 0]
        ax_clean.imshow(clean_img_pil)
        title_clean = f"Clean\nPred: {get_class_name(clean_pred_idx, imagenet_classes)} ({clean_conf:.2f})\nTrue: {get_class_name(true_label_idx, imagenet_classes)}"
        ax_clean.set_title(title_clean, color=("green" if clean_pred_idx == true_label_idx else "red"), fontsize=9)
        ax_clean.axis("off")

        # Display adversarial images for each definition
        for j, col_def in enumerate(column_definitions):
            ax_adv = axs[i, j + 1]
            adv_results = col_def["data_results"]

            # Ensure the sample index is valid for the current adversarial result
            if adv_results and i < len(adv_results.get("images", [])):
                adv_img_tensor = adv_results["images"][i]
                adv_pred_idx = adv_results["preds"][i]
                adv_conf = adv_results["confidences"][i]
                # true_label_idx is the same as for the clean image for comparison

                adv_img_pil = inv_tensor_transform(adv_img_tensor)
                ax_adv.imshow(adv_img_pil)
                title_adv = f"{col_def['title_segment']}\nPred: {get_class_name(adv_pred_idx, imagenet_classes)} ({adv_conf:.2f})"
                ax_adv.set_title(title_adv, color=("green" if adv_pred_idx == true_label_idx else "red"), fontsize=9)
            else:
                # Handle cases where adversarial data might be missing for this sample or param
                ax_adv.text(0.5, 0.5, "N/A", horizontalalignment="center", verticalalignment="center", transform=ax_adv.transAxes)
                ax_adv.set_title(f"{col_def['title_segment']}\nPred: N/A", fontsize=9)
            ax_adv.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_model_comparison_bar_chart(all_model_results, attack_names, metric="accuracy"):
    """
    Plots a bar chart comparing models based on a specified metric (e.g., accuracy, confidence)
    for different adversarial attacks, including clean accuracy.
    """
    model_names = list(all_model_results.keys())
    num_models = len(model_names)

    # Categories to plot: 'Clean' first, then the specified adversarial attacks
    plot_categories = ["Clean"] + attack_names
    num_categories = len(plot_categories)

    # Prepare data for plotting
    metric_values = np.zeros((num_models, num_categories))

    for i, model_name in enumerate(model_names):
        for j, category_name in enumerate(plot_categories):
            if category_name == "Clean":
                if "clean" in all_model_results[model_name]:
                    source_results = all_model_results[model_name]["clean"]
                    if metric == "accuracy":
                        if "correct" in source_results and "total" in source_results and source_results["total"] > 0:
                            metric_values[i, j] = (source_results["correct"] / source_results["total"]) * 100
                        else:
                            metric_values[i, j] = 0
                    elif metric in source_results: # For other metrics like confidence
                        metric_values[i, j] = source_results[metric]
                    else:
                        metric_values[i, j] = 0 # Metric not found for clean
                else:
                    metric_values[i, j] = 0 # No clean data
            else: # Adversarial attack
                if category_name in all_model_results[model_name]:
                    source_results = all_model_results[model_name][category_name]
                    if metric == "accuracy":
                        if "correct" in source_results and "total" in source_results and source_results["total"] > 0:
                            metric_values[i, j] = (source_results["correct"] / source_results["total"]) * 100
                        else:
                            metric_values[i, j] = 0
                    elif metric in source_results:
                        metric_values[i, j] = source_results[metric]
                    else:
                        metric_values[i, j] = 0 # Metric not found for attack
                else:
                    metric_values[i, j] = 0 # Attack data not found

    # Plotting
    _, ax = plt.subplots(figsize=(10 + num_categories * 1.5, 7))

    bar_width = 0.8 / num_categories
    index = np.arange(num_models)

    for j, category_name in enumerate(plot_categories):
        bars = ax.bar(index + j * bar_width, metric_values[:, j], bar_width, label=category_name)
        # Add annotations for each bar
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Model")
    ax.set_ylabel(metric.replace("_", " ").title() + (" (%)" if metric == "accuracy" else ""))
    ax.set_title(f'Model Comparison: {metric.replace("_", " ").title()} under Various Conditions')
    ax.set_xticks(index + bar_width * (num_categories - 1) / 2)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend(title="Condition")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()
