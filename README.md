# [AAAI 2024] Video Frame Prediction from a Single Image and Events

> This repository documents personal experiments based on the AAAI 2024 paper:  
> **"Video Frame Prediction from a Single Image and Events"**  
> The paper proposes a lightweight, high-speed model that predicts future video frames using a single image and asynchronous event data. It supports flexible prediction intervals and handles complex dynamic scenes effectively.

---

## 📌 Objective

This repo extends the original implementation with the following experimental goals:

- Run **0.25-second frame prediction** based on the framerate of two event datasets: `bs_ergb` and `hs_ergb`
- Preprocess raw `.npz` event files into `.npy` and `.hdf5` formats aligned with image frames
- Evaluate model behavior across low/high framerate scenarios using real-world event data

---

## 🗂️ Dataset Overview

| Dataset    | Paper |
|------------|-------------|
| `bs_ergb`  | Time Lens++: Event-based Frame Interpolation with Parametric Non-linear Flow and Multi-scale Fusion (CVPR 2022) |
| `hs_ergb`  | Time Lens: Event-based Video Frame Interpolation (CVPR 2021) |
| Common     | Provided as raw `.npz` files containing asynchronous events (unaligned with images) |

---

## 🧩 Preprocessing Pipeline

### 1. Convert `.npz` → `.npy`

- Raw event data is unaligned with image frames.  
  To address this, I adapted [`bs_ergb_to_npy.py`](https://github.com/ercanburak/EVREAL/tree/main/tools) from the [EVREAL] project.
- Custom scripts used:
  - `utils/bs_ergb_to_npy.py`
  - `utils/hs_ergb_to_npy.py`

```bash
python utils/bs_ergb_to_npy.py \
  --input_dir ./raw_npz_data/bs_ergb \
  --output_dir ./processed_npy/bs_ergb
```


### 2. Convert `.npy` → `.hdf5`

Converted `.npy` event sequences to `.hdf5` format using `convert_npz.ipynb`.  
This ensures compatibility with the model’s input loader.

---

## 🛠️ Environment Setup

```bash
conda create -y -n VFPSIE python=3.8
conda activate VFPSIE
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```
