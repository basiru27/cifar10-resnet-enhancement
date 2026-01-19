import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import json
import matplotlib.pyplot as plt

# 1. Reproducibility & Device
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Hyperparameters (Increased Epochs for Convergence)
BATCH_SIZE = 128
LEARNING_RATE = 0.1
EPOCHS = 50
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

# CIFAR-10 stats
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# 3. Enhanced Data Transformations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

# 4. Load Data
print("Downloading and loading CIFAR-10 dataset...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train samples: {len(trainset)}, Test samples: {len(testset)}")

# 5. ENHANCED MODEL ARCHITECTURE
def get_enhanced_resnet18():
    # Load standard ResNet18
    model = torchvision.models.resnet18(weights=None, num_classes=10)

    # ENHANCEMENT 1: Modify the first conv layer for CIFAR-10
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # ENHANCEMENT 2: Remove the MaxPool layer
    model.maxpool = nn.Identity()

    return model

model = get_enhanced_resnet18().to(device)
print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# 6. Optimizer and Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    steps_per_epoch=len(trainloader),
    epochs=EPOCHS
)

# 7. Training & Evaluation Functions
def train(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    correct, total, running_loss = 0, 0, 0.0
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss/len(loader), 100.*correct/total

def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss/len(loader), 100.*correct/total

# 8. Main Training Loop
print("\n" + "="*60)
print("Starting training...")
print("="*60 + "\n")

best_acc = 0
train_losses = []
test_losses = []
train_accs = []
test_accs = []
BASELINE_ACC = 80.0

start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()

    train_loss, train_acc = train(model, trainloader, criterion, optimizer, scheduler, device)
    test_loss, test_acc = evaluate(model, testloader, criterion, device)

    # Store metrics
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'enhanced_best.pth')

    epoch_time = time.time() - epoch_start
    print(f'Epoch {epoch+1:2d}/{EPOCHS} | Train Acc: {train_acc:6.2f}% | Test Acc: {test_acc:6.2f}% | Best: {best_acc:6.2f}% | Time: {epoch_time:.1f}s')

total_time = time.time() - start_time
print(f"\n{'='*60}")
print(f"✅ Training Complete!")
print(f"   Peak Accuracy: {best_acc:.2f}%")
print(f"   Total Time: {total_time/60:.1f} minutes")
print(f"{'='*60}\n")

# 9. Save Results to JSON
results = {
    'best_accuracy': best_acc,
    'final_train_acc': train_accs[-1],
    'final_test_acc': test_accs[-1],
    'epochs': EPOCHS,
    'total_training_time_minutes': total_time/60,
    'train_losses': train_losses,
    'test_losses': test_losses,
    'train_accs': train_accs,
    'test_accs': test_accs,
    'hyperparameters': {
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'momentum': MOMENTUM,
        'epochs': EPOCHS
    },
    'model_architecture': 'Enhanced ResNet18 for CIFAR-10',
    'enhancements': [
        'Modified conv1: 3x3 stride 1 (instead of 7x7 stride 2)',
        'Removed MaxPool layer',
        'OneCycleLR scheduler',
        'RandomErasing augmentation'
    ]
}

with open('enhanced_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Results saved to enhanced_results.json")

# 10. Create Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
ax1.plot(range(1, EPOCHS+1), train_losses, label='Train Loss', linewidth=2, marker='o', markersize=3)
ax1.plot(range(1, EPOCHS+1), test_losses, label='Test Loss', linewidth=2, marker='s', markersize=3)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Accuracy plot
ax2.plot(range(1, EPOCHS+1), train_accs, label='Train Acc', linewidth=2, marker='o', markersize=3)
ax2.plot(range(1, EPOCHS+1), test_accs, label='Test Acc', linewidth=2, marker='s', markersize=3)
ax2.axhline(y=BASELINE_ACC, color='r', linestyle='--', linewidth=2, label=f'Baseline: {BASELINE_ACC:.1f}%')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title(f'Training and Test Accuracy (Best: {best_acc:.2f}%)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 100])

plt.tight_layout()
plt.savefig('enhanced_results.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved to enhanced_results.png")
plt.show()

# 11. Download files (for Google Colab)
try:
    from google.colab import files
    print("\n📥 Downloading files...")
    files.download('enhanced_results.json')
    files.download('enhanced_results.png')
    files.download('enhanced_best.pth')
    print("✅ All files downloaded successfully!")
except ImportError:
    print("\n⚠️  Not running in Google Colab - files saved locally")
    print("   - enhanced_results.json")
    print("   - enhanced_results.png")
    print("   - enhanced_best.pth")

print("\n" + "="*60)
print("✅ Enhanced model training complete!")
print("="*60)