import torch
import torchvision.transforms as transforms
import numpy as np  # Added import for np.random

# Standard normalization for ImageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# Inverse transform to display images later
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

inv_tensor_transform = transforms.Compose([
    inv_normalize,
    transforms.ToPILImage()
])

def load_imagenet_classes(label_map_file):
    """Loads ImageNet class labels from a file."""
    with open(label_map_file, "r") as f:
        lines = f.readlines()
    
    temp_classes = {}
    for line in lines:
        line = line.strip()
        parts = line.split(',', 1)
        idx = int(parts[0].strip())
        name = parts[1].strip().replace("'", "").replace('"', '')
        # Take only the main name if multiple are given
        name = name.split(',')[0].strip()
        temp_classes[idx] = name
    return temp_classes

def get_class_name(class_idx, imagenet_classes):
    """Helper function to get class name from index using the loaded map."""
    if isinstance(class_idx, torch.Tensor):
        class_idx = class_idx.item()
    class_idx = int(class_idx)
    return imagenet_classes.get(class_idx, f"Unknown Index {class_idx}")

def set_seed(seed, device):
    """Sets random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed_all(seed)

def get_fixed_parameter(params):
    return params[len(params) // 2]
