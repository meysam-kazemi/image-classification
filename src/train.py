import torch
import torch.nn as nn

def train_model(model, train_loader, test_loader, optimizer, criterion, device, model_name, model_path, epochs=5):
    """
    Handles the training and evaluation loop for a PyTorch model.
    """
    model.to(device)
    print(f"\n--- Training {model_name} on {device} ---")

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Move data to the selected device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # --- Evaluation ---
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad(): # No need to calculate gradients during evaluation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Save the model's state dictionary
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return accuracy
