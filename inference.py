import torch
import torchvision.transforms as transforms
import timm
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_mapping = {
    "ResNet-50": "resnet50",
    "EfficientNet-B3": "efficientnet_b3",
    "ConvNeXt-Tiny": "convnext_tiny",
    "DenseNet-121": "densenet121",
    "MobilenetV3": "mobilenetv3_large_100",
    "Swin-Tiny": "swin_tiny_patch4_window7_224",
    "ViT-B16": "vit_base_patch16_224",
    "ViT-L32": "vit_large_patch32_224",
    "MaxViT-T": "maxvit_tiny_rw_224",
    "MobileViT-S": "mobilevit_s"
}

print("\nAvailable Models:", ", ".join(model_mapping.keys()))
model_name = input("Enter the model you want to test (e.g., ResNet-50): ").strip()

if model_name not in model_mapping:
    raise ValueError(f"Invalid model! Choose from: {', '.join(model_mapping.keys())}")

data_type = input("Enter the type of data (iris/eye): ").strip().lower()
if data_type not in ["iris", "eye"]:
    raise ValueError("Invalid data type! Choose either 'iris' or 'eye'.")

data_path = f"./data/{data_type}"
model_path = f"./models/{model_name}/{data_type}.pth"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data folder not found: {data_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

print(f"\nLoading model: {model_name} ({data_type})...")
model = timm.create_model(model_mapping[model_name], pretrained=False, num_classes=3)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

valid_extensions = ('.jpg', '.JPG', '.bmp')

image_files = [f for f in os.listdir(data_path) if f.endswith(valid_extensions)]

if not image_files:
    raise FileNotFoundError("No valid images found in the folder.")

print(f"\nProcessing {len(image_files)} images from {data_type} dataset...")

for image_file in image_files:
    image_path = os.path.join(data_path, image_file)

    try:
        image = Image.open(image_path).convert("L")  
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            predicted_class = torch.argmax(output, dim=1).item()

        class_labels = ['0', '1', '2']  

        print(f"{image_file} → Predicted Class: {class_labels[predicted_class]}")

    except Exception as e:
        print(f"Error processing {image_file}: {e}")
