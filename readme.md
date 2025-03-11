# Video Super-Resolution Using SRGAN

This repository contains an implementation of Super-Resolution for videos using the Super-Resolution Generative Adversarial Network (SRGAN). The project enhances low-resolution videos by reconstructing high-resolution frames while preserving details and textures.

## 🚀 Features
- Uses SRGAN to upscale video frames.
- Enhances visual quality while maintaining realistic textures.
- Supports multiple input resolutions.
- Implements both Generator and Discriminator models for adversarial training.

## 📂 Installation
Clone the repository:
```bash
git clone https://github.com/Aayush7900/Video-Super-Resolution-Using-SRGAN.git
cd Video-Super-Resolution-Using-SRGAN
```
Create and Activate a Conda Environment:
```bash
conda create -n myenv python=3.11
conda activate myenv
```
Install Dependencies:
```bash
conda install -c nvidia cuda-toolkit=11.8
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 -c pytorch
pip install opencv-python tqdm flask
```
To run app:
```bash
python app.py
```
