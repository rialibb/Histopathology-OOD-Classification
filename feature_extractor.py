from huggingface_hub import login
import timm
import os
from torchvision import models
import torch
import torch.nn as nn





def extractor_model(name, device):
    """
    Loads a specified pretrained feature extractor model and returns it along with its feature dimension.

    Inputs:
        - name (str): The name of the model to load. Supported options: 'provgigapath', 'kimianet', 'ctranspath'.
        - device (torch.device): Device to which the model will be moved (e.g., 'cuda' or 'cpu').

    Outputs:
        - feature_extractor (nn.Module): Pretrained model with the classification head removed, ready for feature extraction.
        - feature_dim (int): Dimensionality of the output feature vectors (1536 for provgigapath, 1024 for kimianet, 768 for ctranspath).
    """
    
    if name=='provgigapath':
        
        # 1. Get token from environment variable
        hf_token = os.getenv("HF_TOKEN")

        if hf_token is None:
            raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")

        # 2. Login securely
        login(token=hf_token)

        # 3. Load the model
        feature_extractor = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).to(device)
        # 4. Remove classifier head and set to eval
        feature_extractor.reset_classifier(0)
        feature_extractor.eval()
        
        return feature_extractor, 1536
        
    elif name=='kimianet':

        # Load DenseNet121 backbone
        backbone = models.densenet121(pretrained=True)
        features = backbone.features

        feature_extractor = nn.Sequential(
            features,
            nn.ReLU(inplace=True),  # optional but common in DenseNet usage
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        ).to(device)
        
        # Load pretrained KimiaNet weights, but skip the classification head
        state_dict = torch.load('pretrained_models/KimiaNetPyTorchWeights.pth')
        # Remove classifier weights (those with final layer keys)
        filtered_state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
        # Load filtered weights (non-strict to avoid key mismatch)
        feature_extractor.load_state_dict(filtered_state_dict, strict=False)
        # set model to eval
        feature_extractor.eval()
        
        return feature_extractor, 1024
        
    elif name =='ctranspath':
        from models import ctranspath
        
        feature_extractor = ctranspath().to(device)

        state_dict = torch.load('pretrained_models/ctranspath.pth', map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        feature_extractor.load_state_dict(state_dict, strict=False)

        feature_extractor.reset_classifier(0)
        feature_extractor.eval()
        
        return feature_extractor, 768
        
    else:
        
        raise ValueError(f"Feature extractor {name} is invalid.")
    
    