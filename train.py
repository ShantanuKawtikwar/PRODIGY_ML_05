import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataset import get_dataloaders
from tqdm import tqdm

def train_model(data_dir, num_epochs=5, batch_size=32, learning_rate=0.001):
    # Setup Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else 
        "cpu"
    )
    print(f"Using device: {device}")

    # Ensure dataset exists before trying to load it
    food101_root = os.path.join(data_dir, 'food-101')
    if not os.path.exists(food101_root):
        print(f"Dataset not found at {food101_root}. Please run download_data.py first.")
        return

    # Data Loaders
    print("Initializing Data Loaders...")
    # On Windows num_workers>0 can sometimes cause broken pipes if not wrapped in __main__, using 0 for safety in script format
    train_loader, test_loader, classes = get_dataloaders(food101_root, batch_size=batch_size, num_workers=0)
    num_classes = len(classes)
    print(f"Found {num_classes} classes.")

    # Model Definition
    print("Loading pretrained ResNet50...")
    # Loading torchvision ResNet50. Warning: weights=ResNet50_Weights.DEFAULT is newer PyTorch syntax
    try:
        from torchvision.models import ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    except ImportError:
        # Fallback for older torchvision
        model = models.resnet50(pretrained=True)

    # Freeze base model layers (optional, but speeds up early training)
    for param in model.parameters():
        param.requires_grad = False

    # Replace classification head
    num_ftrs = model.fc.in_features
    # We unfreeze the final fully connected layer to train on the 101 new classes
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # We only optimize the final layer initially since we froze the rest
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_acc = 0.0
    best_model_wts = model.state_dict()

    print("Starting Training...")
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = test_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            pbar = tqdm(dataloader, desc=f"{phase.capitalize()} Phase")
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward (track history if only in train)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update bar
                pbar.set_postfix({'loss': loss.item()})
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model if it's the best one so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, 'best_food_model.pth')
                print(">>> Saved new best model!")

    time_elapsed = time.time() - start_time
    print(f"\\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

if __name__ == '__main__':
    # Define dataset root path internally, usually downloaded to 'data'
    train_model(data_dir='./data', num_epochs=5, batch_size=32)
