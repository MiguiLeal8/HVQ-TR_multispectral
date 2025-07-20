# HVQ-TR for Anomaly Detection in Multispectral Images

## 📌 Model Summary

HVQ-TR (Hierarchical Vector Quantized Transformer) is a model designed for **unsupervised anomaly detection** in multiband or multispectral images. It relies on the reconstruction of normal regions and the reconstruction error in anomalous regions as a detection mechanism.

During inference, the model tries to reconstruct each region of the image using a set of learned prototypes (codebooks). Normal regions are reconstructed with low error, while anomalous ones—since they don't conform to learned patterns—generate high reconstruction errors, enabling their localization.

---

## 🧠 Model Component Overview

### General Architecture

HVQ-TR consists of the following main components:

- **Feature extractor (encoder):**
  - Based on `EfficientNet-B4`, adapted to multiband inputs (5 channels in this case).
  - Extracts multi-scale feature maps and freezes them to avoid the *identical shortcut* effect, which degrades detection.

- **Transformer Encoder:**
  - Composed of 4 `TransformerEncoder` layers, it processes spatial tokens to capture long-range dependencies.

- **Hierarchical Vector Quantization (VQ):**
  - Encoder outputs at each level are quantized using `Quantize` modules.
  - Each level includes 15 codebook versions to allow switching (although only class 0 is used in this project).
  - Prevents prototype collapse and increases robustness of the latent space.

- **Transformer Decoder:**
  - With 4 layers (`TransformerDecoder_hierachy`), it reconstructs the original features.
  - Uses cross-attention over quantized vectors and positional embeddings to preserve spatial structure.

- **Switching Mechanism:**
  - Although designed for multi-class switching, this configuration trains only with class 0, using one set of codebooks and output projectors.

- **Reconstruction and Prediction:**
  - Compares the reconstruction with original features using Euclidean distance (`L2`).
  - Interpolates the result to produce an anomaly map matching the original image size.

---

## 📁 Directory and File Structure

```text
TFGMiguelLeal/
├── Cod/
│   ├── data/                  # Not relevant after modifications
│   ├── experiments/           # Checkpoints, logs, and config.yaml
│   ├── hvq/                   # Model source code
│   ├── README.md              # This file
│   ├── requirements.txt       # Required dependencies
│   ├── pred_mean_<exp>.txt    # Predictions for an experiment
│   └── truth_<img>.txt        # Ground truth for an image
├── Img/                       # Images in .raw and .pgm format
├── Mod/                       # Stored pretrained model
└── Res/                       # Generated anomaly maps
```

---

## ⚙️ Usage Manual

### 1. Create virtual environment

```bash
cd HVQ-Multispectral/Cod/HVQ-Trans/HVQ-Trans-master
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🛠️ Project Configuration

### File: `experiments/oitaven/config.yaml`

- **trainer:**
  - `max_epoch`: number of training epochs
  - `clip_max_norm`: gradient clipping
  - `val_freq_epoch`: validation frequency

- **saver:**
  - `save_dir`: checkpoint directory
  - `auto_resume`: resume from last checkpoint

### File: `hvq/tools/train_val.py`

- `DATASET` and `GT`: paths to `.raw` and `.pgm` files
- `SAMPLES`: training sample percentage or count
- `batch_size`: batch size
- `results`: evaluate a pre-trained model if enabled
- `checkpoint`: load pre-trained models
- `threshold`: method for thresholding (Otsu, Li, Youden, etc.)

---

## 🚀 Run the Model

```bash
source venv/bin/activate
python3 -m hvq.tools.train_val --config experiments/oitaven/config.yaml
```

---

## 📌 References

- [Lu et al., 2023 - HVQ-TR: Hierarchical Vector Quantized Transformers for Unsupervised Anomaly Detection](https://proceedings.neurips.cc/paper_files/paper/2023/file/1abc87c67cc400a67b869358e627fe37-Paper-Conference.pdf)
