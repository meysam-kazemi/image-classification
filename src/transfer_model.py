import torch.nn as nn
from torchvision import models

def build_transfer_model(num_classes=10):
    """
    Builds a transfer learning model using a pre-trained MobileNetV2.
    """
    # Load a pre-trained model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze all the parameters in the pre-trained model
    for param in model.parameters():
        param.requires_grad = False
    
    # Get the number of input features for the classifier
    num_ftrs = model.classifier[1].in_features
    
    # Replace the final classifier layer with our own
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)
    )
    
    return model
