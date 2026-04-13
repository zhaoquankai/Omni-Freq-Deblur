<div align="center">

# Omni-Freq-Deblur: Capturing Omni-Frequency for 2D and 3D Scene Deblurring

**[Quankai Zhao](https://github.com/zhaoquankai)**<sup>1</sup>, Bo Jiang<sup>1</sup>, Tianle Xie<sup>1</sup>, Wenpeng Qiu<sup>1</sup>, Haoxiang Wang<sup>1</sup>, Yuzhuo Wang<sup>1</sup>, Xin Dang<sup>1</sup>, Xiaoxuan Chen<sup>1</sup>, Yaowei Li<sup>1</sup>

<sup>1</sup>School of Electronic and Information Engineering, Northwest University

**[Paper (Under Review)]** | **[Supplementary Material]** | **[Pre-trained Models]** | **[Visual Results]**

</div>

> **Abstract:** Current deblurring methods, despite achieving remarkable performance on 2D scenes, often suffer from band-limited frequency recovery and an inability to handle 3D scenes. Unlike 2D blur, 3D blur is driven by six-degree-of-freedom (6-DoF) camera motion, which results in spatially-variant distortions where pixel displacement is intrinsically entangled with scene depth. To address this challenge, we propose Omni-Freq-Deblur, a unified framework that casts both 2D and 3D scene deblurring as an explicit spectral reconstruction problem. The framework decomposes the deblurring process into three complementary frequency bands: Zero-Frequency Calibration (ZFC), Low-Frequency Wavelet Self-Attention (L-WSA), and High-Frequency Wavelet Self-Attention (H-WSA).

---

## 🚀 Network Architecture

<p align="center">
  <img src="./figures/architecture.png" width="90%">
</p>

---

## 🛠️ Installation

```bash
git clone [https://github.com/zhaoquankai/Omni-Freq-Deblur.git](https://github.com/zhaoquankai/Omni-Freq-Deblur.git)
cd Omni-Freq-Deblur
conda create -n omnifreq python=3.10
conda activate omnifreq
pip install -r requirements.txt
python setup.py develop