import os
import torch
import torch.optim as optim
import torch.nn as nn

from src.data_loader import get_dataloaders_cnn, get_dataloaders_transfer
from src.cnn_model import CNNModel
from src.transfer_model import build_transfer_model
from src.train import train_model
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

def main():
    """Main function to run the training and comparison."""
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(config.get('MODELS', 'dir'), exist_ok=True) 

    # Train and evaluate the Custom CNN Model
    print("="*50)
    print("Handling Custom CNN Model")
    print("="*50)
    
    train_loader_cnn, test_loader_cnn = get_dataloaders_cnn()
    cnn_model = CNNModel()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    
    cnn_accuracy = train_model(
        model=cnn_model,
        train_loader=train_loader_cnn,
        test_loader=test_loader_cnn,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        model_name=config.get('MODEL', 'cnn_name'),
        model_path=config.get('MODEL', 'cnn_path')
    )

    #  Train and evaluate the Transfer Learning Model 
    print("\n" + "="*50)
    print("Handling Transfer Learning Model")
    print("="*50)
    
    train_loader_tl, test_loader_tl = get_dataloaders_transfer()
    transfer_model = build_transfer_model()

    criterion_tl = nn.CrossEntropyLoss()

    optimizer_tl = optim.Adam(filter(lambda p: p.requires_grad, transfer_model.parameters()), lr=0.001)
    
    transfer_accuracy = train_model(
        model=transfer_model,
        train_loader=train_loader_tl,
        test_loader=test_loader_tl,
        optimizer=optimizer_tl,
        criterion=criterion_tl,
        device=device,
        model_name=config.get("MODEL", "transfer_name"),
        model_path=config.get('MODEL', 'transfer_path')
    )

    #  Comparison 
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    print(f"Custom CNN Model Accuracy: {cnn_accuracy:.2f}%")
    print(f"Transfer Learning Model Accuracy: {transfer_accuracy:.2f}%\n")
    
    print(" Trainable Parameters ")
    cnn_trainable_params = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)
    print(f"Custom CNN: {cnn_trainable_params:,}")
    
    tl_trainable_params = sum(p.numel() for p in transfer_model.parameters() if p.requires_grad)
    tl_total_params = sum(p.numel() for p in transfer_model.parameters())
    print(f"Transfer Learning (Trainable only): {tl_trainable_params:,}")
    print(f"Transfer Learning (Total): {tl_total_params:,}")

if __name__ == '__main__':
    main()
