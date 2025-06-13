"""Utility functions for evaluating adversarial attacks on a model."""
import numpy as np
import torch


def evaluate_attack(attack_instance, test_loader, model, device, num_images_to_process):
    """Evaluates a single attack instance on the test loader."""
    model.eval()
    attack_results = {"correct": 0, "total": 0, "images": [], "preds": [], "confidences": [], "time": 0.0}
    clean_results = {"images": [], "preds": [], "labels": [], "confidences": []}
    total_processed = 0

    for _, (images, labels) in enumerate(test_loader):
        if total_processed >= num_images_to_process:
            break

        images, labels = images.to(device), labels.to(device)
        current_batch_size = images.shape[0]

        # Store clean info for comparison
        with torch.no_grad():
            outputs_clean = model(images)
            preds_clean = torch.argmax(outputs_clean, dim=1)
            softmax_clean = torch.softmax(outputs_clean, dim=1)
            confidences_clean = softmax_clean.gather(
                1, preds_clean.unsqueeze(1)).squeeze(1).cpu().tolist()
            if isinstance(confidences_clean, float):
                confidences_clean = [confidences_clean]

        clean_results["images"].extend(images.cpu())
        clean_results["preds"].extend(preds_clean.cpu().tolist())
        clean_results["labels"].extend(labels.cpu().tolist())
        clean_results["confidences"].extend(confidences_clean)

        # Generate and Evaluate Adversarial Images
        adv_images = attack_instance(images, labels)

        with torch.no_grad():
            outputs_adv = model(adv_images)
            preds_adv = torch.argmax(outputs_adv, dim=1)
            softmax_adv = torch.softmax(outputs_adv, dim=1)
            confidences_adv = softmax_adv.gather(
                1, preds_adv.unsqueeze(1)).squeeze(1).cpu().tolist()
            if isinstance(confidences_adv, float):
                confidences_adv = [confidences_adv]

        correct_adv = (preds_adv == labels).sum().item()
        attack_results["correct"] += correct_adv
        attack_results["total"] += current_batch_size
        attack_results["images"].extend(adv_images.cpu())
        attack_results["preds"].extend(preds_adv.cpu().tolist())
        attack_results["confidences"].extend(confidences_adv)

        total_processed += current_batch_size
        if total_processed >= num_images_to_process:
             print(f"  Processed {total_processed} images. Stopping.")
             break
        print(f"  Processed {total_processed}/{num_images_to_process} images...", end="\r")

    attack_results["accuracy"] = (attack_results["correct"] / attack_results["total"]) * 100 if attack_results["total"] > 0 else 0

    print(f"Attack evaluation finished. Accuracy: {attack_results['accuracy']:.2f}%")
    return attack_results, clean_results

def evaluate_all_attacks(attacks, test_loader, model, device, num_images, imagenet_classes):
    """Evaluates all specified attacks and clean images."""
    results_summary = {"clean": {"correct": 0, "total": 0, "images": [], "preds": [], "labels": [], "confidences": []}}
    for attack_name in attacks.keys():
        results_summary[attack_name] = {"correct": 0, "total": 0, "images": [], "preds": [], "confidences": []}

    print("\nStarting evaluation...")
    total_processed = 0

    for i, (images, labels) in enumerate(test_loader):
        if total_processed >= num_images:
            break

        images_device, labels_device = images.to(device), labels.to(device)
        current_batch_size = images_device.shape[0]

        # Evaluate on clean images
        with torch.no_grad():
            outputs_clean = model(images_device)
            preds_clean = torch.argmax(outputs_clean, dim=1)
            softmax_clean = torch.softmax(outputs_clean, dim=1)
            confidences_clean_tensor = softmax_clean.gather(
                1, preds_clean.unsqueeze(1)).squeeze(1)
            confidences_clean_list = confidences_clean_tensor.cpu().tolist()
            if isinstance(confidences_clean_list, float):
                confidences_clean_list = [confidences_clean_list]

        correct_clean = (preds_clean == labels_device).sum().item()
        results_summary["clean"]["correct"] += correct_clean
        results_summary["clean"]["total"] += current_batch_size
        results_summary["clean"]["images"].extend(images.cpu()) # Store original images from dataloader (not on device)
        results_summary["clean"]["preds"].extend(preds_clean.cpu().tolist())
        results_summary["clean"]["labels"].extend(labels.cpu().tolist()) # Store original labels
        results_summary["clean"]["confidences"].extend(confidences_clean_list)

        print(f"  Batch {i+1}/{len(test_loader)}: Processing {current_batch_size} images...")

        # Generate and evaluate adversarial images
        for attack_name, attack_fn in attacks.items():
            adv_images = attack_fn(images_device, labels_device) # Pass images and labels on device

            with torch.no_grad():
                outputs_adv = model(adv_images) # adv_images are already on device from attack
                preds_adv = torch.argmax(outputs_adv, dim=1)
                softmax_adv = torch.softmax(outputs_adv, dim=1)
                confidences_adv_tensor = softmax_adv.gather(1, preds_adv.unsqueeze(1)).squeeze(1)
                confidences_adv_list = confidences_adv_tensor.cpu().tolist()
                if isinstance(confidences_adv_list, float):
                    confidences_adv_list = [confidences_adv_list]

            correct_adv = (preds_adv == labels_device).sum().item() # Compare with labels on device
            results_summary[attack_name]["correct"] += correct_adv
            results_summary[attack_name]["total"] += current_batch_size
            results_summary[attack_name]["images"].extend(adv_images.cpu()) # Store adv images (now on CPU)
            results_summary[attack_name]["preds"].extend(preds_adv.cpu().tolist())
            results_summary[attack_name]["confidences"].extend(confidences_adv_list)

            print(f"    {attack_name} finished. Accuracy: {correct_adv}/{current_batch_size}")

        total_processed += current_batch_size
        if total_processed >= num_images:
            print(f"\nReached target number of images ({num_images}). Stopping.")
            break

    print("\nEvaluation finished.")
    return results_summary

def print_accuracy_confidence(results_summary, attacks):
    """Prints accuracy and average confidence for clean and adversarial results."""
    print("-" * 30)
    print("Accuracy and Average Confidence Results:")
    print("-" * 30)

    acc_clean = (results_summary["clean"]["correct"] / results_summary["clean"]["total"]) * 100 if results_summary["clean"]["total"] > 0 else 0
    avg_conf_clean = np.mean(results_summary["clean"]["confidences"]) if results_summary["clean"]["confidences"] else 0

    print(f"Clean Accuracy: {results_summary['clean']['correct']}/{results_summary['clean']['total']} ({acc_clean:.2f}%)")
    print(f"Clean Avg Confidence: {avg_conf_clean:.4f}")
    print("-" * 30)

    for attack_name in attacks.keys():
        total_adv = results_summary[attack_name]["total"]
        correct_adv = results_summary[attack_name]["correct"]
        confidences_adv = results_summary[attack_name]["confidences"]

        acc_adv = (correct_adv / total_adv) * 100 if total_adv > 0 else 0
        avg_conf_adv = np.mean(confidences_adv) if confidences_adv else 0

        print(f"{attack_name} Accuracy: {correct_adv}/{total_adv} ({acc_adv:.2f}%)")
        print(f"{attack_name} Avg Confidence: {avg_conf_adv:.4f}")
        print("-" * 30)
