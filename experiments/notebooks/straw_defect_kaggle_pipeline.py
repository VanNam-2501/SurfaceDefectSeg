# %% [markdown]
# # Juice Box Straw Defect Classification Pipeline
# This notebook demonstrates the full pipeline for classifying defects in juice box straws.
# It trains and evaluates three models: ResNet18, EfficientNet-B0, and MobileNetV3-Small.

# %% [markdown]
# ## a. Setup & Install dependencies

# %%
# Install required packages (if not already installed)
# !pip install -q torch torchvision scikit-learn pandas matplotlib seaborn

import os
import time
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# Configuration
DATA_DIR = "/kaggle/input/juice-box-straw-defects"
OUTPUT_DIR = "/kaggle/working/results"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")

# %% [markdown]
# ## b. Dataset exploration & visualization

# %%
def explore_dataset(data_dir):
    data_path = Path(data_dir)
    classes = [d.name for d in data_path.iterdir() if d.is_dir()]
    print(f"Classes found: {classes}")
    
    class_counts = {}
    sample_images = {}
    
    for cls in classes:
        cls_dir = data_path / cls
        images = list(cls_dir.glob("*.jpg"))
        if not images:
            images = list(cls_dir.glob("*.png"))
            
        class_counts[cls] = len(images)
        if images:
            sample_images[cls] = str(images[0])
            
    print("\nImages per class:")
    for cls, count in class_counts.items():
        print(f" - {cls}: {count}")
        
    # Plot samples
    plt.figure(figsize=(15, 5))
    for i, (cls, img_path) in enumerate(sample_images.items()):
        plt.subplot(1, len(classes), i + 1)
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(f"{cls}\n({img.size})")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if os.path.exists(DATA_DIR):
    explore_dataset(DATA_DIR)
else:
    print(f"Dataset path {DATA_DIR} does not exist. (Skipping exploration)")

# %% [markdown]
# ## c. Data preparation

# %%
class StrawDefectDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def prepare_dataloaders(data_dir, batch_size=32, image_size=224, train_split=0.7, val_split=0.15):
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Data path {data_dir} does not exist. Returning empty loaders.")
        return {}, []
        
    classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    all_data = []
    for cls in classes:
        cls_dir = data_path / cls
        images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        for img in images:
            all_data.append((str(img), class_to_idx[cls]))
            
    # Shuffle and split
    random.shuffle(all_data)
    n = len(all_data)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    
    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train+n_val]
    test_data = all_data[n_train+n_val:]
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    datasets = {
        'train': StrawDefectDataset(train_data, transform=train_transform),
        'val': StrawDefectDataset(val_data, transform=val_transform),
        'test': StrawDefectDataset(test_data, transform=val_transform)
    }
    
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, num_workers=2),
        'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, num_workers=2)
    }
    
    return dataloaders, classes

dataloaders, class_names = prepare_dataloaders(DATA_DIR, BATCH_SIZE, IMAGE_SIZE)
print(f"Classes: {class_names}")

# %% [markdown]
# ## d. Model definition

# %%
def get_model(model_name, num_classes):
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Unknown model {model_name}")
    return model

# %% [markdown]
# ## Common Training Loop

# %%
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                val_acc_history.append(epoch_acc.item())
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, val_acc_history

# Global evaluation function
def evaluate_model(model, test_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    return acc, cm, report

results = {}
trained_models = {}

# %% [markdown]
# ## e. Training ResNet18

# %%
if dataloaders:
    print("Training ResNet18...")
    resnet = get_model('resnet18', len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.parameters(), lr=LEARNING_RATE)
    
    resnet, _ = train_model(resnet, dataloaders, criterion, optimizer, num_epochs=EPOCHS)
    trained_models['resnet18'] = resnet
    
    torch.save(resnet.state_dict(), os.path.join(OUTPUT_DIR, 'resnet18_best.pth'))

# %% [markdown]
# ## f. Training EfficientNet-B0

# %%
if dataloaders:
    print("Training EfficientNet-B0...")
    effnet = get_model('efficientnet_b0', len(class_names)).to(DEVICE)
    optimizer = optim.Adam(effnet.parameters(), lr=LEARNING_RATE)
    
    effnet, _ = train_model(effnet, dataloaders, criterion, optimizer, num_epochs=EPOCHS)
    trained_models['efficientnet_b0'] = effnet
    
    torch.save(effnet.state_dict(), os.path.join(OUTPUT_DIR, 'efficientnet_b0_best.pth'))

# %% [markdown]
# ## g. Training MobileNetV3-Small

# %%
if dataloaders:
    print("Training MobileNetV3-Small...")
    mobilenet = get_model('mobilenet_v3_small', len(class_names)).to(DEVICE)
    optimizer = optim.Adam(mobilenet.parameters(), lr=LEARNING_RATE)
    
    mobilenet, _ = train_model(mobilenet, dataloaders, criterion, optimizer, num_epochs=EPOCHS)
    trained_models['mobilenet_v3_small'] = mobilenet
    
    torch.save(mobilenet.state_dict(), os.path.join(OUTPUT_DIR, 'mobilenet_v3_small_best.pth'))

# %% [markdown]
# ## h. Evaluation & Comparison

# %%
confusion_matrices = {}

if dataloaders:
    for name, model in trained_models.items():
        print(f"\nEvaluating {name} on test set...")
        acc, cm, report = evaluate_model(model, dataloaders['test'], class_names)
        results[name] = {
            'Accuracy': acc,
            'Macro F1': report['macro avg']['f1-score'],
            'Weighted F1': report['weighted avg']['f1-score']
        }
        confusion_matrices[name] = cm
        print(f"Accuracy: {acc:.4f}")

# %% [markdown]
# ## i. Confusion matrices for all models

# %%
if confusion_matrices:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (name, cm) in enumerate(confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=class_names, yticklabels=class_names)
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'))
    plt.show()

# %% [markdown]
# ## j. Final results summary table

# %%
if results:
    df_results = pd.DataFrame(results).T
    print("\nFinal Results Summary:")
    print("="*50)
    print(df_results)
    
    # Save to CSV
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'))
    
    # Plot comparison
    df_results.plot(kind='bar', figsize=(10, 6))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison_chart.png'))
    plt.show()
