# Children Age Estimation Model Inference

This repository provides a **Python script** to test multiple **CNN and Transformer-based models** for **children age estimation** using either **iris-only** or **full-eye** images.

## 🚀 Features
- Supports **10 different CNN & Transformer models** trained for children age estimation:
  - `ResNet-50`
  - `EfficientNet-B3`
  - `ConvNeXt-Tiny`
  - `DenseNet-121`
  - `MobilenetV3`
  - `Swin-Tiny`
  - `ViT-B16`
  - `ViT-L32`
  - `MaxViT-T`
  - `MobileViT-S`
- Each model is trained on both:
  - **Iris-only images** (`iris.pth`)
  - **Full-eye images** (`eye.pth`)
---

## 📥 Download the Pretrained Models
Since `.pth` model files are large, they are **not included in this repository**.  
You must **download them manually** from Google Drive and place them inside the `models/` folder.

1. **Download models from Google Drive**  
   📌 **Google Drive Link**: [https://drive.google.com/drive/folders/1aHE6rauxsAiT4UOfYTqTV4DcrkdkvcQ2?usp=drive_link]

2. **Extract & Place the Models**  
   - Place each `eye.pth` and `iris.pth` file inside the corresponding model folder.  
   - Example:  
     ```
     models/ResNet-50/iris.pth
     models/ResNet-50/eye.pth
     ```

## 🔧 Installation
**Clone Repository, Install Dependencies and Run the Inference**
```bash
git clone https://github.com/yourusername/age-estimation-inference.git
cd age-estimation-inference

pip install torch torchvision timm pillow numpy

python inference.py


