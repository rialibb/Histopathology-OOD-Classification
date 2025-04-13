import torch
import random
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
import os

from utils import precompute
from datasets import BaselineDataset




def preprocessor(name):
    """
    Returns preprocessing transforms (with and without augmentation) based on the given model name.

    Inputs:
        - name (str): The name of the feature extractor. Should be one of ['ctranspath', 'kimianet', 'provgigapath'].

    Outputs:
        - preprocessor_tr (torchvision.transforms.Compose): Training transform with data augmentations.
        - preprocessor_val (torchvision.transforms.Compose): Validation/test transform without augmentations.
    """
    
    if name in ['ctranspath', 'kimianet']:
        preprocessor_tr = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        
        preprocessor_val = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        return preprocessor_tr, preprocessor_val

    elif name == 'provgigapath':
        preprocessor_tr = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        preprocessor_val = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        
        return preprocessor_tr, preprocessor_val

    else:
        raise ValueError(f"Feature extractor {name} is invalid.")
    
    
    


def preprocess_dataset(preprocessing_tr, preprocessing_val, feature_extractor, tr_path, val_path, PREPROCESSED_FOLDER, train_file, val_file, bs, seed, device):
    """
    Preprocesses and extracts features from training and validation datasets, then saves them to disk.

    Inputs:
        - preprocessing_tr (Compose): Transformations with augmentations applied to training data.
        - preprocessing_val (Compose): Transformations applied to validation data.
        - feature_extractor (nn.Module): Pretrained model used to extract features.
        - tr_path (str): Path to the training dataset file (e.g., HDF5).
        - val_path (str): Path to the validation dataset file.
        - PREPROCESSED_FOLDER (str): Directory where precomputed features will be saved.
        - train_file (str): File path for saving precomputed training features and labels.
        - val_file (str): File path for saving precomputed validation features and labels.
        - bs (int): Batch size for data loading.
        - seed (int): Random seed for reproducibility.
        - device (torch.device): Device to perform computation on.

    Outputs:
        - None (saves precomputed features and labels to disk).
    """
    torch.random.manual_seed(seed)
    random.seed(seed)

    train_dataset = BaselineDataset(tr_path, preprocessing_tr, 'train')
    val_dataset = BaselineDataset(val_path, preprocessing_val, 'train')

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=bs)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=bs)

    # Precompute
    train_features, train_labels = precompute(train_dataloader, feature_extractor, device)
    val_features, val_labels = precompute(val_dataloader, feature_extractor, device)
    # Save to disk
    if not os.path.exists(PREPROCESSED_FOLDER):
        os.makedirs(PREPROCESSED_FOLDER)
    torch.save({'features': train_features, 'labels': train_labels}, train_file)
    torch.save({'features': val_features, 'labels': val_labels}, val_file)





