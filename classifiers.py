import torch.nn as nn



class SEHead(nn.Module):
    """
    Classification head with a Squeeze-and-Excitation (SE) block for binary prediction from feature vectors.

    Inputs:
        - in_features (int): Dimensionality of the input features.
        - reduction (int): Reduction ratio for the SE block (default: 16).

    Outputs (from forward pass):
        - output (Tensor): Predicted probability tensor of shape [B, 1], with values in [0, 1].
    """
    def __init__(self, in_features, reduction=16):
        super(SEHead, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(256, 64)
        self.relu2 = nn.ReLU()

        # Squeeze-and-Excitation block after the 64-dim layer
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.se_fc1 = nn.Linear(64, 64 // reduction)
        self.se_relu = nn.ReLU()
        self.se_fc2 = nn.Linear(64 // reduction, 64)
        self.se_sigmoid = nn.Sigmoid()

        self.out = nn.Linear(64, 1)
        self.out_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))     # [B, 256]
        x = self.relu2(self.fc2(x))     # [B, 64]

        # Squeeze and Excitation block
        se = self.global_avg_pool(x.unsqueeze(-1)).squeeze(-1)  # [B, 64] -> [B, 64]
        se = self.se_fc1(se)
        se = self.se_relu(se)
        se = self.se_fc2(se)
        se = self.se_sigmoid(se)
        x = x * se  # channel-wise scaling

        x = self.out(x)
        x = self.out_activation(x)
        return x

        
    
    
class ResidualMLP(nn.Module):
    """
    Residual multi-layer perceptron (MLP) for binary classification from feature vectors.

    Inputs:
        - dim (int): Dimensionality of the input features.

    Outputs (from forward pass):
        - output (Tensor): Predicted probability tensor of shape [B, 1], with values in [0, 1].
    """
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, dim)
        self.out = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x += residual
        return self.out(x)
    
    
    

class FullyConnectedBinary(nn.Module):
    """
    Simple fully connected binary classifier for predicting probabilities from feature vectors.

    Inputs:
        - num_ftrs (int): Dimensionality of the input feature vectors.

    Outputs (from forward pass):
        - output (Tensor): Predicted probability tensor of shape [B, 1], with values in [0, 1].
    """
    def __init__(self, num_ftrs):
        super(FullyConnectedBinary, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.fc(x)
        return out
    
    
    
    

def choose_classifier(name):
    """
    Returns the appropriate classifier head class based on the name of the feature extractor.

    Inputs:
        - name (str): Name of the feature extractor. Should be one of ['provgigapath', 'kimianet', 'ctranspath'].

    Outputs:
        - classifier_class (nn.Module): Corresponding classifier class to be instantiated later.

    Raises:
        - ValueError: If the provided name is not a supported feature extractor.
    """
    if name=='provgigapath':
        return ResidualMLP
    elif name=='kimianet':
        return FullyConnectedBinary
    elif name=='ctranspath':
        return SEHead
    else:
        raise ValueError(f"Feature extractor {name} is invalid.")