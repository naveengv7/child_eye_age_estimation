import tensorflow as tf
import numpy as np
import os
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

TFLITE_MODEL_PATH = "mobilenet_iris.tflite"
TEST_DIR = "./test"  # Folder structure: /class_name/image.png


IMAGE_SIZE = (224, 224)
CLASS_NAMES = sorted(os.listdir(TEST_DIR))  # ['0', '1', '2']


interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  
    img = img.resize(IMAGE_SIZE)
    img = np.stack([np.array(img)] * 3, axis=-1)  # (H, W) -> (H, W, 3)
    img = (img / 255.0 - 0.5) / 0.5 
    img = img.astype(np.float32)
    img = np.transpose(img, (2, 0, 1))  # -> (3, 224, 224)
    return np.expand_dims(img, axis=0)  # -> (1, 3, 224, 224)



all_preds = []
all_labels = []
total_inference_time = 0
total_images = 0

for label_idx, class_name in enumerate(CLASS_NAMES):
    class_dir = os.path.join(TEST_DIR, class_name)
    for fname in os.listdir(class_dir):
        img_path = os.path.join(class_dir, fname)
        input_data = preprocess_image(img_path)

        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        end = time.time()

        output = interpreter.get_tensor(output_details[0]['index'])  # Shape: (1, 3)
        pred = np.argmax(output[0])

        all_preds.append(pred)
        all_labels.append(label_idx)

        total_inference_time += (end - start)
        total_images += 1


accuracy = accuracy_score(all_labels, all_preds) * 100
precision = precision_score(all_labels, all_preds, average='weighted') * 100
recall = recall_score(all_labels, all_preds, average='weighted') * 100
f1 = f1_score(all_labels, all_preds, average='weighted') * 100
avg_time = (total_inference_time / total_images) * 1000  # ms

print(f" Test Accuracy: {accuracy:.2f}%")
print(f" Precision: {precision:.2f}%")
print(f" Recall: {recall:.2f}%")
print(f" F1-Score: {f1:.2f}%")
print(f"Avg Inference Time: {avg_time:.2f} ms/image")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
