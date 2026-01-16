# CIFAR-10 ResNet18 Baseline Model
# AI Final Project - Baseline Implementation

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import json

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.1
EPOCHS = 50
NUM_CLASSES = 10
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

# CIFAR-10 normalization constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

print("\n" + "="*60)
print("BASELINE RESNET18 ON CIFAR-10")
print("="*60)

# Data transformations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

# Load CIFAR-10
print("\nLoading CIFAR-10...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(f"Training samples: {len(trainset)}")
print(f"Test samples: {len(testset)}")

# Model
print("\nBuilding ResNet18...")
model = torchvision.models.resnet18(weights=None, num_classes=NUM_CLASSES)

# Initialize weights
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

model.apply(init_weights)
model = model.to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total

# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total

# Training loop
train_losses, train_accs = [], []
test_losses, test_accs = [], []
best_acc = 0

print("\n" + "="*60)
print("TRAINING")
print("="*60)

start_time = time.time()

for epoch in range(EPOCHS):
    print(f'\nEpoch {epoch+1}/{EPOCHS}')

    train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, testloader, criterion, device)
    scheduler.step()

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'baseline_best.pth')

    print(f'Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Best: {best_acc:.2f}%')

training_time = (time.time() - start_time) / 60

# Results
print("\n" + "="*60)
print("BASELINE RESULTS")
print("="*60)
print(f"Best Test Accuracy: {best_acc:.2f}%")
print(f"Final Test Accuracy: {test_accs[-1]:.2f}%")
print(f"Training Time: {training_time:.1f} minutes")
print("="*60)

# Save results
results = {
    'model': 'Baseline ResNet18',
    'best_test_acc': float(best_acc),
    'final_test_acc': float(test_accs[-1]),
    'final_train_acc': float(train_accs[-1]),
    'training_time_min': float(training_time),
    'test_accs': [float(x) for x in test_accs],
    'train_accs': [float(x) for x in train_accs]
}

with open('baseline_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label='Train', linewidth=2)
ax1.plot(test_losses, label='Test', linewidth=2)
ax1.set_title('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(train_accs, label='Train', linewidth=2)
ax2.plot(test_accs, label='Test', linewidth=2)
ax2.set_title(f'Accuracy (Best: {best_acc:.2f}%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('baseline_results.png', dpi=300)
plt.show()

# Download
from google.colab import files
files.download('baseline_results.json')
files.download('baseline_results.png')

print("\n✅ Baseline complete! Download the files.")