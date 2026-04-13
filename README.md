<div align="center">

# Omni-Freq-Deblur: Capturing Omni-Frequency for 2D and 3D Scene Deblurring

**[Quankai Zhao](https://github.com/zhaoquankai)**<sup>1</sup>, Bo Jiang<sup>1</sup>, Tianle Xie<sup>1</sup>, Wenpeng Qiu<sup>1</sup>, Haoxiang Wang<sup>1</sup>, Yuzhuo Wang<sup>1</sup>, Xin Dang<sup>1</sup>, Xiaoxuan Chen<sup>1</sup>, Yaowei Li<sup>1</sup>

<sup>1</sup>School of Electronic and Information Engineering, Northwest University

[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://github.com/zhaoquankai/Omni-Freq-Deblur) 
[![Dataset](https://img.shields.io/badge/Dataset-Download-green)](#-datasets)
[![Pretrained Model](https://img.shields.io/badge/Pretrained-Model-blue)](https://github.com/zhaoquankai/Omni-Freq-Deblur)

</div>

---

## 🚀 Network Architecture

<p align="center">
  <img src="./figures/architecture.png" width="100%">
</p>

---

## 📂 Datasets

You can download the following datasets using the official links provided in the table:

| Dataset | Type | Download Link |
| :--- | :---: | :--- |
| **GoPro** | 2D Dynamic | [[Official Page]](https://seungjunnah.github.io/Datasets/gopro.html) [[Google Drive]](https://drive.google.com/open?id=1S0vK8SpsEAnXvG780785Vp6qHjM7xY-R) |
| **HIDE** | 2D Human | [[Official Page]](https://github.com/joanshen0508/Hide-Dataset) [[Google Drive]](https://drive.google.com/drive/folders/15E8w-n3Z3p3O2zE_A3S6_i-T6u6r2V0-) |
| **RealBlur** | 2D Real | [[Official Page]](https://github.com/rim-jhim/RealBlur) [[Google Drive]](https://drive.google.com/drive/folders/1R4_7y4X8XjS_zPZ_yqN7-Y_M6-Oq-8vL) |
| **DeRF (Deblur-NeRF)** | 3D Scene | [[Official Page]](https://github.com/limacv/Deblur-NeRF) [[Dataset Link]](https://drive.google.com/drive/folders/1LpM-3z7X8Q8v-XjYf0S7_k-Y_W-K_B) |
| **GSBlur** | 3D Gaussian | [[OpenReview]](https://openreview.net/forum?id=Awu8YlEofZ) [[Github]](https://github.com/zhaoquankai/Omni-Freq-Deblur) |

### Dataset Organization
Please organize your data as follows:
```text
Omni-Freq-Deblur/
├── datasets/
│   ├── GoPro/
│   │   ├── train/
│   │   └── test/
│   ├── HIDE/
│   └── GSBlur/
│       ├── test/
│       │   ├── input_noise/
│       │   └── target/
