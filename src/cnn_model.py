import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        # Conv layer 1: 1 input channel (grayscale), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv layer 2: 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        
        # Flattening and fully connected layers
        # The input features to fc1 needs calculation based on conv/pool layers
        # Input 28x28 -> conv1(3x3) -> 26x26 -> pool(2x2) -> 13x13
        # 13x13 -> conv2(3x3) -> 11x11 -> pool(2x2) -> 5x5
        # So, the flattened size is 64 channels * 5 * 5 = 1600
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Apply layers sequentially with activation functions
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the tensor for the dense layers
        x = x.view(-1, 64 * 5 * 5)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Output raw logits
        return x
