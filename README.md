# MLP-MFF: Lightweight Pyramid Fusion MLP for Ultra-Efficient End-to-End Multi-focus Image Fusion

## 📖 Introduction

**MLP-MFF** is a lightweight, end-to-end multi-focus image fusion framework based on a pyramid fusion MLP architecture. It achieves ultra-efficient and high-quality fusion of multi-focus images, making it suitable for real-time and resource-constrained applications.

---

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


## 🙏 Acknowledgements
xxx

If you have any questions, please open an issue in this repository.
