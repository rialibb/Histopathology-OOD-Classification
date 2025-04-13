from torch.utils.data import Dataset
import h5py
import torch
import numpy as np



class BaselineDataset(Dataset):
    """
    Custom PyTorch Dataset for loading images and labels from an HDF5 file, with optional preprocessing.

    Inputs:
        - dataset_path (str): Path to the HDF5 file containing image patches and (optionally) labels.
        - preprocessing (Callable): Transformations to apply to the images (e.g., augmentations, normalization).
        - mode (str): Either 'train' (loads labels) or another mode (e.g., 'test', skips labels).

    Outputs (per item):
        - image (Tensor): Preprocessed image tensor.
        - label (int or None): Corresponding label if in training mode, otherwise None.
    """
    def __init__(self, dataset_path, preprocessing, mode):
        super(BaselineDataset, self).__init__()
        self.dataset_path = dataset_path
        self.preprocessing = preprocessing
        self.mode = mode
        
        with h5py.File(self.dataset_path, 'r') as hdf:        
            self.image_ids = list(hdf.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        with h5py.File(self.dataset_path, 'r') as hdf:
            img = torch.tensor(hdf.get(img_id).get('img'))
            label = np.array(hdf.get(img_id).get('label')) if self.mode == 'train' else None
        return self.preprocessing(img).float(), label
    
    
    
    

class PrecomputedDataset(Dataset):
    """
    Dataset for loading precomputed features and labels for training or evaluation.

    Inputs:
        - features (Tensor): Tensor of shape [num_samples, feature_dim], containing extracted features.
        - labels (Tensor): Tensor of shape [num_samples], containing binary labels (0 or 1).

    Outputs (per item):
        - feature (Tensor): Feature vector for a single sample.
        - label (Tensor): Corresponding label as a float tensor of shape [1].
    """
    def __init__(self, features, labels):
        super(PrecomputedDataset, self).__init__()
        self.features = features
        self.labels = labels.unsqueeze(-1)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].float()
    
    


    
    
    
    