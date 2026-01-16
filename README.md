# Enhanced ResNet18 for CIFAR-10

AI Final Project - Option A: Enhance an Existing AI Model

## Goal
Improve ResNet18 performance on CIFAR-10 through systematic enhancements.

## Structure
- `data/` - CIFAR-10 dataset
- `models/` - Saved checkpoints
- `results/` - Metrics and visualizations
- `notebooks/` - Training code
- `src/` - Source code
- `docs/` - Final report

## 🛠️ Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cifar10-resnet-enhancement.git
cd cifar10-resnet-enhancement

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Option 1: Google Colab (Recommended)

1. Upload `baseline_model.py` to Google Colab
2. Run all cells
3. Download results and models

### Option 2: Local Training
```bash
# Train baseline model
python src/baseline_model.py

# Train enhanced model
python src/enhanced_model.py
```

## 📈 Training Details

- **Dataset:** CIFAR-10 (60,000 32x32 images)
- **Architecture:** ResNet18
- **Training Time:** ~25-35 minutes on Tesla T4 GPU
- **Epochs:** 50
- **Batch Size:** 128

## 📝 Report

Full project report available in `docs/final_report.md`

## 👥 Team Members

- Amadou Basiru Jallow
- Yusupha Conta
- Muhammed Sillah

## 🙏 Acknowledgments

- CIFAR-10 dataset by Alex Krizhevsky
- PyTorch team for the framework
- ResNet architecture by He et al. (2016)
