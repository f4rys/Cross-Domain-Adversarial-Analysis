# Adversarial Attacks Analysis

This repository explores the vulnerabilities of Convolutional Neural Networks (CNNs) and hyperspectral model to adversarial attacks. It provides experiments and analysis on standard CNN architectures as well as the HybridSN model for hyperspectral image classification.

## Project Overview
- **Attacks** This project contains implementations of FGSM (Fast Gradient Sign Method), PGD (Projected Gradient Descent), CW (Carlini & Wagner) and DeepFool (untargeted).
- **Adversarial Attacks on CNNs:** Evaluate and compare the robustness of CNNs against various adversarial attack methods (e.g., FGSM, PGD, DeepFool, CW) using the ImageNet validation set.
- **Hyperspectral Model Analysis:** Analyze the impact of adversarial attacks on the HybridSN model using the Indian Pines dataset.
- **Notebooks:**
  - `model_comparison.ipynb` and `parameter_impact.ipynb`: Experiments on CNNs and adversarial attacks.
  - `hyperspectral.ipynb`: Experiments on the HybridSN model with hyperspectral data.

## Environment Setup

This repository provides two environment YAML files for easy setup with Conda:

- **For CNN experiments (`model_comparison.ipynb`, `parameter_impact.ipynb`):**
  ```sh
  conda env create -f env_cnns.yml
  conda activate CNNs
  ```
- **For hyperspectral experiments (`hyperspectral.ipynb`):**
  ```sh
  conda env create -f env_hybridsn.yml
  conda activate HybridSN
  ```

**Note:**
The provided environments install PyTorch **without CUDA support** by default. If you have a compatible GPU and want to enable CUDA acceleration, you must reinstall PyTorch with CUDA. See [PyTorch installation instructions](https://pytorch.org/get-started/locally/) and select the appropriate CUDA version for your system.

## Attribution

This repository makes use of the following datasets and external resources:
- [Indian Pines dataset](https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes#Indian_Pines) on CC0 1.0 Universal License
- [Subset of ImageNet validation set](https://www.kaggle.com/datasets/titericz/imagenet1k-val) on CC0 1.0 Universal License 
- [HybridSN implementation and pretrained weights](https://github.com/Pancakerr/HybridSN) on MIT License
