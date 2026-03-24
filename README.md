# ASA-G2AS-CBQ
Gradient-Guided Anisotropic Attention for Interpretable and Robust Cement Bond Quality Evaluation from VDL Data


Data in 
https://drive.google.com/drive/folders/1zTKWenxeEkhhggCCb_P1fm63JutwC4XV?usp=sharing

---

# VDL Guided Attention Training

This script, `run_guided_attention_training.py`, is a PyTorch-based deep learning framework designed for classifying **Variable Density Log (VDL)** slices. It supports multiple attention mechanisms, Guided Global Attention Smoothing (GGAS), and rigorous Two-level Leave-One-Out (LOO) cross-validation.

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have the following libraries installed:
* `torch`, `torchvision`, `numpy`, `pandas`, `Pillow`, `tqdm`, `scikit-learn`

### 2. Data Structure
By default, the script reads data from a directory named `vdl_slices_20px`. The structure should be organized by well and then by class:
```text
vdl_slices_20px/
├── BH-1/
│   ├── Good/ (PNG images)
│   ├── Midrate/
│   └── Poor/
├── BH-2_2-1/ ...
```


### 3. Core Configuration
You can toggle experimental settings using **Environment Variables** without modifying the code:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ATTENTION` | `ASA` | Options: `ASA` (Anisotropic Spatial Attention), `SE`, `ECA`, `CBAM`, `None` |
| `LAMBDA` | `0.0` | Guidance loss weight. Setting > 0 enables GGAS teacher-student training |
| `NORM_TYPE` | `BN` | Normalization: `BN` (Batch), `IN` (Instance), or `NONE` |
| `ASA_FUSION` | `gate` | ASA internal fusion: `sum`, `weighted`, or `gate` |
| `SEED` | `42` | Random seed for reproducibility |

### 4. Running Examples
**Standard run with default ASA:**
```bash
python run_guided_attention_training.py
```

**Enable Guided Learning (GGAS) with specific attention placement:**
```bash
export LAMBDA=0.05
export ATT_L1=0
export ATT_L2=1
export ATT_L3=1
python run_guided_attention_training.py
```


---

## 📊 Outputs
The script automatically generates several files to track performance and explainability:
* **Model Weights**: `.pt` files for each outer fold.
* **Training Logs**: `inner_curve_...csv` detailing the epoch selection process.
* **Evaluation Metrics**: `outer_loo_results_...csv` containing accuracy, F1-scores, and physics-based explainability scores (e.g., Pearson correlation with Grad-CAM).




