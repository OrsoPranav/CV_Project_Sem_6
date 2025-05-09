# FixMatch Implementation for ISIC 2017 Dataset
# This implementation follows the FixMatch paper (Sohn et al., 2020)
# for semi-supervised learning on skin lesion images

# Cell 1: Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Ensure reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Cell 2: Set up the device and paths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths to the ISIC2017 dataset
base_dir = "isic2017"
train_dir = os.path.join(base_dir, "ISIC-2017_Training_Data")
train_gt_dir = os.path.join(base_dir, "ISIC-2017_Training_Part1_GroundTruth")
valid_dir = os.path.join(base_dir, "ISIC-2017_Validation_Data")
valid_gt_dir = os.path.join(base_dir, "ISIC-2017_Validation_Part1_GroundTruth")
test_dir = os.path.join(base_dir, "ISIC-2017_Test_v2_Data")
test_gt_dir = os.path.join(base_dir, "ISIC-2017_Test_v2_Part1_GroundTruth")

# Cell 3: Create a custom dataset class for ISIC2017
class ISIC2017Dataset(Dataset):
    def __init__(self, data_dir, gt_dir, transform=None, return_name=False):
        """
        Args:
            data_dir: Directory with all the images
            gt_dir: Directory with ground truth masks
            transform: Optional transform to be applied to the images
            return_name: Whether to return the image filename
        """
        self.data_dir = data_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.return_name = return_name
        
        # Get list of all image files
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.jpg')])
        
        # Extract label from ground truth files (for classification task)
        # Note: For ISIC 2017, we'll convert the segmentation masks to classification labels
        # (presence of lesion or not) for simplicity
        self.labels = []
        for img_file in self.image_files:
            # Get corresponding mask filename (ISIC2017 naming convention)
            img_id = img_file.split('.')[0]
            mask_file = f"{img_id}_segmentation.png"
            
            if os.path.exists(os.path.join(gt_dir, mask_file)):
                # Check if the mask has any positive pixels (lesion present)
                mask = Image.open(os.path.join(gt_dir, mask_file)).convert("L")
                mask_np = np.array(mask)
                # If any pixel is > 0, consider it as positive class (1), else negative (0)
                has_lesion = 1 if np.sum(mask_np > 0) > 0 else 0
                self.labels.append(has_lesion)
            else:
                # If no mask exists, assume no lesion (class 0)
                self.labels.append(0)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        if self.return_name:
            return image, label, self.image_files[idx]
        return image, label

# Cell 4: Define the weak and strong augmentations for FixMatch
class RandAugment:
    """RandAugment implementation for strong augmentation in FixMatch"""
    def __init__(self, n=2, m=10):
        """
        Args:
            n: Number of augmentations to apply
            m: Magnitude of the augmentations (0-10)
        """
        self.n = n
        self.m = m
        self.augment_list = [
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.RandomPosterize(bits=4, p=0.5),
            transforms.RandomEqualize(p=0.5),
            transforms.RandomSolarize(threshold=128, p=0.5)
        ]
    
    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
            img = op(img)
        return img

# Define the image transformations
# Weak augmentation
weak_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Strong augmentation
strong_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    RandAugment(n=2, m=10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Transform for validation/test
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Cell 5: Load the datasets and create labeled/unlabeled splits
# Load the full training dataset
full_train_dataset = ISIC2017Dataset(train_dir, train_gt_dir, transform=None)

# Split into labeled and unlabeled data
# For FixMatch, we'll use a small portion of labeled data (e.g., 10%)
labeled_ratio = 0.1
num_train = len(full_train_dataset)
num_labeled = int(labeled_ratio * num_train)
num_unlabeled = num_train - num_labeled

# Generate random indices for the split
indices = list(range(num_train))
random.shuffle(indices)
labeled_indices = indices[:num_labeled]
unlabeled_indices = indices[num_labeled:]

print(f"Total training data: {num_train}")
print(f"Labeled data: {num_labeled} ({labeled_ratio*100:.1f}%)")
print(f"Unlabeled data: {num_unlabeled} ({(1-labeled_ratio)*100:.1f}%)")

# Cell 6: Create FixMatch-specific dataset classes
class LabeledDataset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

class UnlabeledDataset(Dataset):
    def __init__(self, dataset, indices, weak_transform=None, strong_transform=None):
        self.dataset = dataset
        self.indices = indices
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        image, _ = self.dataset[self.indices[idx]]
        
        # Apply both weak and strong augmentations
        weak_image = self.weak_transform(image)
        strong_image = self.strong_transform(image)
        
        return weak_image, strong_image

# Create the labeled and unlabeled datasets
labeled_dataset = LabeledDataset(full_train_dataset, labeled_indices, transform=weak_transform)
unlabeled_dataset = UnlabeledDataset(full_train_dataset, unlabeled_indices, 
                                   weak_transform=weak_transform, 
                                   strong_transform=strong_transform)

# Create the validation and test datasets
valid_dataset = ISIC2017Dataset(valid_dir, valid_gt_dir, transform=test_transform)
test_dataset = ISIC2017Dataset(test_dir, test_gt_dir, transform=test_transform)

# Cell 7: Create data loaders
batch_size = 16
labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size*7, shuffle=True, num_workers=2, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Cell 8: Define the model for FixMatch
class FixMatchModel(nn.Module):
    def __init__(self, num_classes=2):
        super(FixMatchModel, self).__init__()
        # Use ResNet50 pretrained on ImageNet as the backbone
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

# Create the model and move it to the device
model = FixMatchModel().to(device)

# Cell 9: Define the FixMatch training function
def train_fixmatch(model, labeled_loader, unlabeled_loader, optimizer, scheduler, 
                  num_epochs=100, threshold=0.95, lambda_u=1.0):
    """
    Train using the FixMatch algorithm
    
    Args:
        model: The neural network model
        labeled_loader: DataLoader for labeled data
        unlabeled_loader: DataLoader for unlabeled data
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        threshold: Confidence threshold for pseudo-labeling
        lambda_u: Weight for the unsupervised loss
    """
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    train_losses, val_accs = [], []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        unlabeled_iter = iter(unlabeled_loader)
        
        with tqdm(labeled_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, (inputs_x, targets_x) in enumerate(pbar):
                try:
                    # Get unlabeled batch
                    (inputs_u_w, inputs_u_s) = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    (inputs_u_w, inputs_u_s) = next(unlabeled_iter)
                
                # Move data to device
                inputs_x, targets_x = inputs_x.to(device), targets_x.to(device)
                inputs_u_w, inputs_u_s = inputs_u_w.to(device), inputs_u_s.to(device)
                
                batch_size = inputs_x.shape[0]
                
                # Forward pass for labeled data
                outputs_x = model(inputs_x)
                
                # Forward pass for unlabeled data (weak and strong augmentation)
                with torch.no_grad():
                    outputs_u_w = model(inputs_u_w)
                    # Generate pseudo-labels using the model's predictions on weakly augmented images
                    probs_u_w = torch.softmax(outputs_u_w, dim=1)
                    max_probs, pseudo_labels = torch.max(probs_u_w, dim=1)
                    # Create mask for confident predictions
                    mask = max_probs.ge(threshold).float()
                
                # Forward pass for unlabeled data with strong augmentation
                outputs_u_s = model(inputs_u_s)
                
                # Calculate losses
                # Supervised loss on labeled data
                loss_x = criterion(outputs_x, targets_x)
                
                # Unsupervised loss on unlabeled data (only for confident predictions)
                loss_u = torch.mean(
                    mask * F.cross_entropy(outputs_u_s, pseudo_labels, reduction='none')
                )
                
                # Combined loss
                loss = loss_x + lambda_u * loss_u
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update progress bar
                epoch_loss += loss.item()
                pbar.set_postfix({"Loss": epoch_loss / (batch_idx + 1),
                                 "Labeled": loss_x.item(),
                                 "Unlabeled": loss_u.item(),
                                 "Mask": mask.mean().item()})
        
        # Evaluate on validation set
        val_acc = evaluate(model, valid_loader)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "fixmatch_best_model.pth")
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss / len(labeled_loader):.4f}, "
              f"Val Acc: {val_acc:.4f}, Best Val Acc: {best_val_acc:.4f}")
        
        train_losses.append(epoch_loss / len(labeled_loader))
    
    # Plot training curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('fixmatch_training_curve.png')
    plt.show()
    
    return train_losses, val_accs

# Cell 10: Define evaluation function
def evaluate(model, dataloader):
    """Evaluate the model on the given dataloader"""
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_preds)
    return accuracy

# Cell 11: Set up optimizer and scheduler
# FixMatch typically uses SGD with momentum and weight decay
optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4, nesterov=True)

# Learning rate scheduler (cosine annealing)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# Cell 12: Train the model with FixMatch
num_epochs = 100
threshold = 0.95  # Confidence threshold for pseudo-labeling
lambda_u = 1.0  # Weight for unsupervised loss

train_losses, val_accs = train_fixmatch(
    model=model,
    labeled_loader=labeled_loader,
    unlabeled_loader=unlabeled_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=num_epochs,
    threshold=threshold,
    lambda_u=lambda_u
)

# Cell 13: Load the best model and evaluate on test set
model.load_state_dict(torch.load("fixmatch_best_model.pth"))
test_acc = evaluate(model, test_loader)
print(f"Test accuracy: {test_acc:.4f}")

# Cell 14: Detailed evaluation with additional metrics
def detailed_evaluation(model, dataloader):
    """Evaluate the model with multiple metrics"""
    model.eval()
    all_preds, all_targets = [], []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    print("Evaluation Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs
    }

# Run detailed evaluation on the test set
test_results = detailed_evaluation(model, test_loader)

# Cell 15: Save the model and results
# Save the model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_accuracy': val_accs[-1],
    'test_accuracy': test_acc,
    'threshold': threshold,
    'lambda_u': lambda_u,
}, 'fixmatch_isic2017_final.pth')

# Save the training history
np.savez('fixmatch_training_history.npz', 
         train_losses=train_losses, 
         val_accs=val_accs,
         test_metrics=test_results)

print("Training complete! Model and results saved.")