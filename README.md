<div align="center">

# Omni-Freq-Deblur: Capturing Omni-Frequency for 2D and 3D Scene Deblurring

**[Quankai Zhao](https://github.com/zhaoquankai)**<sup>1</sup>, Bo Jiang<sup>1</sup>, Tianle Xie<sup>1</sup>, Wenpeng Qiu<sup>1</sup>, Haoxiang Wang<sup>1</sup>, Yuzhuo Wang<sup>1</sup>, Xin Dang<sup>1</sup>, Xiaoxuan Chen<sup>1</sup>, Yaowei Li<sup>1</sup>

<sup>1</sup>School of Electronic and Information Engineering, Northwest University

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://drive.google.com/drive/folders/1o1bZ26RbTPExVZ6YCyGTP3t_yt_blxtJ?usp=sharing) 
[![Dataset](https://img.shields.io/badge/Dataset-Download-green)](#-datasets)
[![Pretrained Model](https://img.shields.io/badge/Pretrained-Model-blue)](#-pre-trained-models)

</div>

---

## 🚀 Network Architecture

<p align="center">
  <img src="./figures/architecture.png" width="100%">
</p>

---

## 📦 Pre-trained Models

Download the checkpoints and place them into `./experiments/pretrained_models/`.

| Model | Task | Download Link |
| :--- | :---: | :--- |
| **Omni-Freq-Deblur (GoPro)** | 2D Deblurring | [[Google Drive]](https://drive.google.com/drive/folders/1o1bZ26RbTPExVZ6YCyGTP3t_yt_blxtJ?usp=sharing) |
| **Omni-Freq-Deblur (GSBlur)** | 3D Deblurring | [[Google Drive]](https://drive.google.com/drive/folders/1o1bZ26RbTPExVZ6YCyGTP3t_yt_blxtJ?usp=sharing) |

---

## 📂 Datasets

| Dataset | Type | Links |
| :--- | :---: | :--- |
| **GoPro** | 2D Dynamic | [[Official Page]](https://seungjunnah.github.io/Datasets/gopro.html) [[Google Drive]](https://drive.google.com/open?id=1S0vK8SpsEAnXvG780785Vp6qHjM7xY-R) |
| **HIDE** | 2D Human | [[Official Page]](https://github.com/joanshen0508/Hide-Dataset) [[Google Drive]](https://drive.google.com/drive/folders/15E8w-n3Z3p3O2zE_A3S6_i-T6u6r2V0-) |
| **Lai RealWorld** | 2D Real (CVPR 2025) | [[Official Page]](https://github.com/phoenix104104/cvpr16_deblur_study) |
| **DeRF (Deblur-NeRF)** | 3D Scene | [[GitHub]](https://github.com/limacv/Deblur-NeRF) [[Dataset Link]](https://drive.google.com/drive/folders/1LpM-3z7X8Q8v-XjYf0S7_k-Y_W-K_B) |
| **GSBlur** | 3D Gaussian | [[OpenReview]](https://openreview.net/forum?id=Awu8YlEofZ) |

### Dataset Organization
```text
Omni-Freq-Deblur/
├── datasets/
│   ├── GoPro/
│   ├── Lai2025/
│   └── GSBlur/
│       ├── test/
│       │   ├── input_noise/
│       │   └── target/
├── models/
│   └── Omni-Freq-Deblur.pdf