# MLP-MFF: Lightweight Pyramid Fusion MLP for Ultra-Efficient End-to-End Multi-focus Image Fusion

<p align="center">
  <img src="https://img.shields.io/badge/python-3.7%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-1.7%2B-orange" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey" alt="Platform">
</p>

---

## 📖 Introduction | 简介

**MLP-MFF** is a lightweight, end-to-end multi-focus image fusion framework based on a pyramid fusion MLP architecture. It achieves ultra-efficient and high-quality fusion of multi-focus images, making it suitable for real-time and resource-constrained applications.

**MLP-MFF** 是一种基于金字塔融合MLP结构的轻量级端到端多焦点图像融合方法，兼具高效性与高质量，适用于实时和资源受限场景。

---

## 🚀 Features
- **Lightweight**: Extremely small model size and low FLOPs
- **Pyramid Fusion**: Multi-scale feature extraction and fusion
- **End-to-End**: Directly outputs fused images without post-processing
- **Easy Training & Inference**: Simple scripts for training and prediction
- **High Performance**: Achieves SOTA results on multiple benchmarks

---

## 🏗️ Method Overview
MLP-MFF leverages a pyramid feature extraction backbone, multi-scale fusion modules, and a lightweight MLP-based decoder. The network is designed for efficiency and effectiveness, with attention and fusion blocks at each scale.

**Key modules:**
- Pyramid feature extraction
- Multi-scale fusion blocks
- Attention mechanisms
- Lightweight upsampling and decoding

---

## 📂 Directory Structure
```
├── Dataloader.py         # Data loading utilities
├── models/
│   └── network.py       # MLP-MFF network definition
├── train.py             # Training script
├── predict.py           # Inference script
├── utils.py             # Utility functions
├── tools/               # Additional tools
└── ...
```

---

## ⚙️ Requirements
- Python >= 3.7
- PyTorch >= 1.7
- torchvision
- numpy
- opencv-python
- Pillow
- tqdm
- thop (for FLOPs/params analysis)

Install dependencies:
```bash
pip install torch torchvision numpy opencv-python pillow tqdm thop
```

---

## 🚦 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/MLP-MFF.git
cd MLP-MFF/github
```

### 2. Prepare Datasets
Organize your multi-focus dataset as:
```
/path/to/dataset/
    train/
        sourceA/
        sourceB/
        groundtruth/
    test/
        sourceA/
        sourceB/
        groundtruth/
```

### 3. Train
```bash
python train.py --mff_datapath /path/to/dataset --epochs 20 --batch_size 32
```

### 4. Inference
```bash
python predict.py --model_path model.pth --input_dir /path/to/testset --output_dir ./results
```

---

## 📝 Citation
If you use this code or ideas in your research, please cite:
```bibtex
@article{YourPaper2024,
  title={MLP-MFF: Lightweight Pyramid Fusion MLP for Ultra-Efficient end-to-end Multi-focus Image Fusion},
  author={XinZhe Xie},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

---

## 🙏 Acknowledgements
- This project is developed by XinZhe Xie, Zhejiang University.
- Thanks to the open-source community and previous works on multi-focus image fusion.

---

## 📬 Contact
For questions or collaborations, please contact: [your-email@domain.com]

---

<p align="center">
  <b>MLP-MFF: Lightweight, Efficient, and Powerful Multi-focus Image Fusion</b>
</p> 
