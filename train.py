import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import json
import argparse
from tqdm import tqdm

class MultiCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Collect all image paths recursively
        self.image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        
        if not self.image_paths:
            raise ValueError(f"No images found in {root_dir}")
            
        # Extract class names from parent directory
        self.classes = sorted(list(set([os.path.basename(os.path.dirname(p)) for p in self.image_paths])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        class_name = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_idx[class_name]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_dataloaders(root_dir, batch_size=32, num_workers=4):
    # Common transformations for ResNet
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # For simplicity, we create one dataset and use random_split. 
    # For a real hold-out, we'd ideally have separate folders.
    full_dataset = MultiCancerDataset(root_dir, transform=None)
    
    # Save class mapping for inference
    with open('class_mapping.json', 'w') as f:
        json.dump(full_dataset.class_to_idx, f)
    
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    # Apply transforms
    train_dataset.dataset.transform = transform_train
    val_dataset.dataset.transform = transform_val
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, full_dataset.classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=r"Multi Cancer\Multi Cancer", help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading data...")
    train_loader, val_loader, classes = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    num_classes = len(classes)
    print(f"Found {len(train_loader.dataset)} training images and {len(val_loader.dataset)} validation images across {num_classes} classes.")

    print("Initializing model...")
    # Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Replace final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix(loss=loss.item(), acc=100.*correct/total)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100. * correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_bar.set_postfix(loss=loss.item())
                
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = 100. * val_correct / val_total

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.2f}%")

        if val_epoch_acc > best_val_acc:
            print(f"Validation accuracy improved from {best_val_acc:.2f}% to {val_epoch_acc:.2f}%. Saving model...")
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), 'model_best.pth')

if __name__ == '__main__':
    main()
