# Ithildin-Crypt

**Zero‑Cryptographic Decryption using Deep Visual Secret Sharing (DVSS) and Generative Adversarial Networks**

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/nittsuj/Ithildin-Crypt)
*(Click the badge above to try the live web application!)*

## Overview
Ithildin-Crypt combines Deep Visual Secret Sharing (DVSS) with Generative Adversarial Networks (GANs). It allows sensitive visual data to be hidden within ordinary-looking image shares and reconstructed without the need for traditional mathematical cryptographic keys.

## Key Features
* **Zero-Cryptographic Decryption:** Replaces standard cryptographic key exchanges with visual neural reconstruction.
* **Dual-Head GAN Architecture:** Powered by a custom `UNetDualHead` engineered to handle complex image-to-image translation.
* **Stabilized Generative Training:** Utilizes Bilinear Interpolation upsampling and TTUR (Two Time-Scale Update Rules) to prevent severe mode collapse.

## Repository Contents
* `app.py` - The main Gradio web application and U-Net architecture.
* `requirements.txt` - Python dependencies (using the CPU-only PyTorch build for optimized cloud deployment).
* *Note: The pre-trained model weights (`.pth`) are hosted directly on the Hugging Face Space due to GitHub's file size limits.*
