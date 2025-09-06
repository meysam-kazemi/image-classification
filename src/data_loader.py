import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders_cnn(batch_size=64):
    """
    Creates data loaders for the custom CNN.
    - Converts images to tensors.
    - Normalizes pixel values.
    """
    # Transformation pipeline
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts images to PyTorch Tensors
        transforms.Normalize((0.1307,), (0.3081,)) # Mean and std dev of MNIST
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    print("Data loaders for CNN created.")
    return train_loader, test_loader

def get_dataloaders_transfer(batch_size=64, img_size=32):
    """
    Creates data loaders for the transfer learning model.
    - Resizes images.
    - Converts to 3 channels for pre-trained models.
    - Converts to tensors and normalizes.
    """
    # Transformation pipeline for pre-trained models
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3), # Convert to 3 channels
        transforms.ToTensor(),
        # Normalization values for models pre-trained on ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    print("\nData loaders for Transfer Learning created.")
    return train_loader, test_loader
