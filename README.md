# MLP-MFF: Lightweight Pyramid Fusion MLP for Ultra-Efficient End-to-End Multi-focus Image Fusion

## 📖 Introduction

Limited depth of field in modern optical imaging systems often results in partially focused images. Multi-focus image fusion (MFF) addresses this by synthesizing an all-in-focus image from multiple source images captured at different focal planes. While deep learning-based MFF methods have shown promising results, existing approaches face significant challenges. Convolutional Neural Networks (CNNs) often struggle to capture long-range dependencies effectively, while Transformer and Mamba-based architectures, despite their strengths, suffer from high computational costs and rigid input size constraints, frequently necessitating patch-wise fusion during inference—a compromise that undermines the realization of a true global receptive field. To overcome these limitations, we propose MLP-MFF, a novel lightweight, end-to-end MFF network built upon the Pyramid Fusion Multi-Layer Perceptron (PFMLP) architecture. MLP-MFF is specifically designed to handle flexible input scales, efficiently learn multi-scale feature representations, and capture critical long-range dependencies. Furthermore, we introduce a Dual-Path Adaptive Multi-scale Feature-Fusion Module based on Hybrid Attention (DAMFFM-HA), which adaptively integrates hybrid attention mechanisms and allocates weights to optimally fuse multi-scale features, thereby significantly enhancing fusion performance. Extensive experiments on public multi-focus image datasets demonstrate that our proposed MLP-MFF achieves competitive, and often superior, fusion quality compared to current state-of-the-art MFF methods, all while maintaining a lightweight and efficient architecture.

## 📄 Paper Link

You can read the paper here:  
[MLP-MFF: Lightweight Pyramid Fusion MLP for Ultra-Efficient End-to-End Multi-focus Image Fusion](https://www.mdpi.com/1424-8220/25/16/5146)
## 🚀 Features
- **Lightweight**: Extremely small model size and low FLOPs
- **Pyramid Fusion**: Multi-scale feature extraction and fusion
- **End-to-End**: Directly outputs fused images without post-processing
- **Easy Training & Inference**: Simple scripts for training and prediction
- **High Performance**: Achieves SOTA results on multiple benchmarks

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
- Python >= 3.8
- PyTorch >= 2.6
- torchvision
- numpy
- opencv-python
- Pillow
- tqdm

Install dependencies:
```bash
pip install torch torchvision numpy opencv-python pillow tqdm
```

---

## 🚦 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Xinzhe99/MLP-MFF.git
cd MLP-MFF
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
About how to make datasets for training, you can refer the code in [LightMFF](https://github.com/Xinzhe99/LightMFF).
### 3. Train
```bash
python train.py --mff_datapath /path/to/dataset --epochs 20 --batch_size 32
```

### 4. Inference
```bash
python predict.py --model_path model.pth --input_dir /path/to/testset --output_dir ./results
```

### 5. Results download link.
```bash
https://pan.baidu.com/s/1yp9GYKGFMN3irKwnc1q2SQ?pwd=cite
```
---

## 📝 Citation
If you use this code or ideas in your research, please cite our paper.

```bibtex
@article{xie2025stackmff,
  title={StackMFF: end-to-end multi-focus image stack fusion network},
  author={Xie, Xinzhe and Qingyan, Jiang and Chen, Dong and Guo, Buyu and Li, Peiliang and Zhou, Sangjun},
  journal={Applied Intelligence},
  volume={55},
  number={6},
  pages={503},
  year={2025},
  publisher={Springer}
}

@article{xie2025multi,
  title={Multi-focus image fusion with visual state space model and dual adversarial learning},
  author={Xie, Xinzhe and Guo, Buyu and Li, Peiliang and He, Shuangyan and Zhou, Sangjun},
  journal={Computers and Electrical Engineering},
  volume={123},
  pages={110238},
  year={2025},
  publisher={Elsevier}
}

@article{xie2024swinmff,
  title={SwinMFF: toward high-fidelity end-to-end multi-focus image fusion via swin transformer-based network},
  author={Xie, Xinzhe and Guo, Buyu and Li, Peiliang and He, Shuangyan and Zhou, Sangjun},
  journal={The Visual Computer},
  pages={1--24},
  year={2024},
  publisher={Springer}
}

@inproceedings{xie2024underwater,
  title={Underwater Three-Dimensional Microscope for Marine Benthic Organism Monitoring},
  author={Xie, Xinzhe and Guo, Buyu and Li, Peiliang and Jiang, Qingyan},
  booktitle={OCEANS 2024-Singapore},
  pages={1--4},
  year={2024},
  organization={IEEE}
}
```
If you have any questions, please open an issue in this repository.
