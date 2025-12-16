# SE-ResNet18 for Brain Tumor MRI Classification

This repository provides the official implementation of **SE-ResNet18** for brain tumor classification using MRI images. The work focuses on **mitigating class-wise bias** in lightweight convolutional neural networks, with particular emphasis on improving discrimination between **glioma** and **meningioma**, two visually similar tumor types.

The code accompanies an academic study that combines **quantitative evaluation** (Accuracy, Macro F1, class-wise recall) and **qualitative interpretability analysis** (Grad-CAM) to demonstrate the effectiveness of channel-wise attention.

---

## 📌 Key Features

* Lightweight **ResNet18 baseline** trained from scratch
* **Squeeze-and-Excitation (SE)** blocks for channel-wise attention
* Detailed **class-wise analysis** (glioma & meningioma recall)
* **Confusion matrix**–based error analysis
* **Grad-CAM visualizations** for interpretability
* Clean and reproducible PyTorch implementation

---

## 🧠 Motivation

Most existing studies on brain tumor MRI classification emphasize overall accuracy using large pre-trained architectures. However, such approaches often overlook **class-wise bias**, which is clinically important. In particular, misclassification between glioma and meningioma can negatively impact treatment planning.

This project aims to:

* Analyze class-wise behavior of a lightweight CNN
* Identify systematic misclassification patterns
* Mitigate these biases using a simple yet effective attention mechanism

---

## 📂 Dataset

The experiments are conducted on a public **Brain Tumor MRI dataset** with four classes:

* Glioma
* Meningioma
* No Tumor
* Pituitary Tumor

> ⚠️ Due to licensing restrictions, the dataset is **not included** in this repository. Please download it from the original source and organize it as follows:

```text
data/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── valid/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

---

## 🏗️ Model Architecture

### Baseline: ResNet18

* Standard ResNet18 architecture
* Final fully connected layer adapted for 4-class classification
* Trained from scratch without pre-trained weights

### Proposed: SE-ResNet18

* SE blocks inserted into each residual block
* Channel-wise feature recalibration via:

  * Global Average Pooling (Squeeze)
  * Bottleneck fully connected layers (Excitation)
  * Sigmoid-based channel weighting

This design introduces minimal computational overhead while significantly improving class-wise balance.

---

## ⚙️ Training Details

* Framework: **PyTorch**
* Optimizer: Adam
* Loss Function: Cross-Entropy Loss
* Evaluation Metrics:

  * Accuracy
  * Macro F1-score
  * Class-wise Recall (Glioma, Meningioma)

Early stopping and best-checkpoint saving are applied based on validation performance.

---

## 📊 Results Summary

| Model       | Accuracy   | Macro F1   | Glioma Recall | Meningioma Recall |
| ----------- | ---------- | ---------- | ------------- | ----------------- |
| ResNet18    | 0.8940     | 0.8867     | 0.7233        | 0.8497            |
| SE-ResNet18 | **0.8955** | **0.8941** | **0.8400**    | **0.9118**        |

While overall accuracy remains comparable, SE-ResNet18 significantly improves **balanced performance**, particularly for challenging tumor classes.

---

## 🔍 Interpretability: Grad-CAM

Grad-CAM is used to visualize class-discriminative regions:

* Baseline ResNet18 shows **diffuse and boundary-biased activations**
* SE-ResNet18 demonstrates **more localized and consistent attention** on tumor-related regions

These qualitative observations align with the quantitative improvements in class-wise recall.

---

## 🚀 Usage

### Train the model

```bash
python train.py --model se_resnet18 --data_dir data/
```

### Evaluate on test set

```bash
python evaluate.py --checkpoint model/best_model.pt
```

### Generate Grad-CAM visualizations

```bash
python gradcam.py --checkpoint model/best_model.pt --class glioma
```

---

## 📁 Repository Structure

```text
├── data/              # Dataset directory (not included)
├── model/             # Saved checkpoints
├── models/            # Model definitions (ResNet18, SE blocks)
├── train.py           # Training script
├── evaluate.py        # Evaluation metrics
├── gradcam.py         # Grad-CAM visualization
├── utils/             # Helper functions
└── README.md
```

---

## 📖 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{se_resnet18_btmri,
  title={Mitigating Class-wise Bias in Brain Tumor MRI Classification Using SE-ResNet18},
  author={Your Name},
  journal={Under Review},
  year={2025}
}
```

---

## 🤝 Acknowledgements

* Original ResNet architecture: He et al.
* Squeeze-and-Excitation Networks: Hu et al.
* Brain Tumor MRI dataset contributors

---

## 📬 Contact

For questions or collaborations, feel free to open an issue or contact:

**Your Name**
Email: [your.email@domain.com](mailto:your.email@domain.com)
