import torch
from tqdm import tqdm
import numpy as np




def precompute(dataloader, model, device):
    """
    Extracts and returns features and labels from a dataloader using a pretrained model.

    Inputs:
        - dataloader (DataLoader): PyTorch DataLoader providing batches of input images and labels.
        - model (nn.Module): Feature extractor model used to compute embeddings.
        - device (torch.device): Device on which to run the model (e.g., 'cuda' or 'cpu').

    Outputs:
        - features (torch.Tensor): Tensor of extracted features with shape [num_samples, feature_dim].
        - labels (torch.Tensor): Tensor of corresponding labels with shape [num_samples].
    """
    xs, ys = [], []
    for x, y in tqdm(dataloader, leave=False):
        with torch.no_grad():
            xs.append(model(x.to(device)).detach().cpu().numpy())
        ys.append(y.numpy())
    xs = np.vstack(xs)
    ys = np.hstack(ys)
    return torch.tensor(xs), torch.tensor(ys)