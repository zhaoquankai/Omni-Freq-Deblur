<div align="center">

# Omni-Freq-Deblur: Capturing Omni-Frequency for 2D and 3D Scene Deblurring

**[Quankai Zhao](https://github.com/zhaoquankai)**<sup>1</sup>, Bo Jiang<sup>1</sup>, Tianle Xie<sup>1</sup>, Wenpeng Qiu<sup>1</sup>, Haoxiang Wang<sup>1</sup>, Yuzhuo Wang<sup>1</sup>, Xin Dang<sup>1</sup>, Xiaoxuan Chen<sup>1</sup>, Yaowei Li<sup>1</sup>

<sup>1</sup>School of Electronic and Information Engineering, Northwest University

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](你的GoogleDrive文件链接) 
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

| Model | Dataset / Task | Download Link |
| :--- | :---: | :--- |
| **Omni-Freq-Deblur (GoPro)** | 2D Dynamic Deblurring | [[Download]](你的模型链接1) |
| **Omni-Freq-Deblur (GSBlur)** | 3D Gaussian Deblurring | [[Download]](你的模型链接2) |

---

## 📂 Datasets

| Dataset | Type | Official Download Link |
| :--- | :---: | :--- |
| **GoPro** | 2D Dynamic | [[Official Page]](https://seungjunnah.github.io/Datasets/gopro.html) |
| **HIDE** | 2D Human | [[GitHub]](https://github.com/joanshen0508/Hide-Dataset) |
| **Lai RealWorld** | 2D Real (CVPR 2025) | [[Official Page]](https://github.com/phoenix104104/cvpr16_deblur_study) |
| **DeRF (Deblur-NeRF)** | 3D Scene | [[GitHub]](https://github.com/limacv/Deblur-NeRF) |
| **GSBlur** | 3D Gaussian | [[OpenReview]](https://openreview.net/forum?id=Awu8YlEofZ) |

### Dataset Organization
```text
Omni-Freq-Deblur/
├── datasets/
│   ├── GoPro/
│   ├── Lai2025/
│   │   ├── test/
│   │   │   ├── input_noise/
│   │   │   └── target/
│   └── GSBlur/
├── models/
│   └── Omni-Freq-Deblur.pdf  <-- Your paper here
