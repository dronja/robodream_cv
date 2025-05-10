# 🧠 Face Hallucination

Low-resolution face images are a common issue in many fields, such as video surveillance systems, forensics, old photo restoration, face recognition systems, and more.

**Face Hallucination** is a type of *super-resolution*, specifically designed to restore details and increase the resolution of face images based on knowledge of facial structure.

---

## 🎯 Goal

Transform blurry or low-quality face images into high-resolution ones using knowledge about typical facial features — while **preserving the identity** of the person.

Typical input resolution: `32×24` or `16×12`  
Target resolution: `96×96` or higher  
> 🔥 **Main challenge:** identity and realism preservation

---

## 🛠️ Common Methods

- Traditional interpolation
- Bayesian-based approaches
- Convolutional Neural Networks (CNNs)
- Generative Adversarial Networks (GANs)

---

## 📐 Network Architectures

| Model | Description | Link |
|-------|-------------|------|
| **SRGAN** | Super-Resolution Generative Adversarial Network | [arXiv:1609.04802](https://arxiv.org/abs/1609.04802) |
| **Real-ESRGAN** | Enhanced SRGAN with real-world degradation handling | |
| **PULSE** | Self-Supervised Upsampling via Latent Space | [GitHub](https://github.com/alex-damian/pulse) |
| **Attention-FH** | Attention-Aware Face Hallucination | [arXiv:1708.03132](https://arxiv.org/abs/1708.03132) |
| **SRResNet** | Residual-based Super-Resolution network | [arXiv:1609.04802](https://arxiv.org/abs/1609.04802) |
| **C-Face Network** | Shape-controlled face hallucination | [arXiv:1503.03832](https://arxiv.org/abs/1503.03832) |

---

## 📂 Dataset

🗃️ Download: [GDrive](https://drive.google.com/file/d/1Qv2c8UN87Wq2qGlyQnPEMh9kj6n87oL8/view?usp=sharing)  
Collected from **FFHQ 2**, **CelebA**, and Internet sources  
- `209,813` face images  
- Size: `89×109` pixels

---

## 📊 Results

### 🌀 PULSE

Generated realistic but **identity-inconsistent** results:

![PULSE](images/pulse.jpeg)

---

### ❌ SRGAN

Performed **very poorly**, not suitable for this task:

![SRGAN](images/SRgan.png)

---

### ✅ SRResNet (simplified)

Showed **the best result**. Used:
- Residual connections
- PixelShuffle (DepthToSpace)

![SRResNet](images/FNET.png)

---

### ⚙️ C-Face (simplified)

Simple encoder-decoder, no residual connections.  
Performed **slightly worse** than SRResNet.

![CNET](images/CNET.png)

---

## 📌 Plans

🧬 **Combine SRResNet and C-Face** to leverage both residual learning and explicit shape control.


