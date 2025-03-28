import os
import json
import cv2
import numpy as np
import tensorflow as tf
import time
from utils import TensorRTModel
from random import shuffle
import sys
shape = 128

def load_model(model_name):
    # path = f"dataset/{model_name.split('_')[0]}/models/{model_name}"
    path = f"dataset/{model_name.split('_')[0]}/models/{model_name}.trt"
    if not os.path.exists(path):
        print(f"Model {model_name} does not exist at {path}!")
        exit(1)
    tensor_engine = TensorRTModel.from_file(path)
    print(f"{model_name}: Model loaded successfully!")
    return tensor_engine

def preprocess_image(file_path, shape):
    img = cv2.imread(file_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (shape, shape), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0) 
    return img

def run_model_test(model_name):
    base_path = f"dataset/{model_name.split('_')[0]}"
    class_dirs = [d for d in os.listdir(base_path) if d.isdigit() and os.path.isdir(os.path.join(base_path, d))]
    class_dirs.sort(key=lambda x: int(x))

    image_paths = []
    labels = []
    for c in class_dirs:
        c_int = int(c)
        class_true_dir = os.path.join(base_path, c, "true")
        if not os.path.exists(class_true_dir):
            continue
        all_images = [img for img in os.listdir(class_true_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        shuffle(all_images)
        all_images = all_images[:50000]
        
        for img_name in all_images:
            image_paths.append(os.path.join(class_true_dir, img_name))
            labels.append(c_int)

    model = load_model(model_name)
    print(f"Testing model: {model_name}, Total images: {len(image_paths)}")

    n_classes = len(class_dirs)
    class_to_index = {int(c): i for i, c in enumerate(class_dirs)}
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)

    total = 0
    correct = 0
    start = time.time()

    for img_path, actual_class in zip(image_paths, labels):
        img = preprocess_image(img_path, shape)
        if img is None:
            continue
        result = model.predict(img)  # Expecting [1, n_classes]
        time.time()
        predicted_class = np.argmax(result)
        end_time = time.time()
        actual_idx = class_to_index[actual_class]
        predicted_idx = class_to_index[predicted_class]

        confusion_matrix[actual_idx, predicted_idx] += 1
        if predicted_class == actual_class:
            correct += 1
        total += 1

    print(f"Mean time taken: {(end_time - start) / total:.5f} seconds")

    end = time.time()
    fps = total / (end - start) if (end - start) > 0 else 0
    overall_accuracy = correct / total if total > 0 else 0.0

    # Compute per-class metrics
    # TP = CM[c,c]
    # FN = sum(CM[c,:]) - CM[c,c]
    # FP = sum(CM[:,c]) - CM[c,c]
    # TN = total - (TP+FP+FN)

    metrics_per_class = {}
    for i, c in enumerate(class_dirs):
        TP = confusion_matrix[i, i]
        FN = np.sum(confusion_matrix[i, :]) - TP
        FP = np.sum(confusion_matrix[:, i]) - TP
        TN = total - (TP + FP + FN)

        accuracy = (TP + TN) / (TP + TN + FP + FN) if TP+TN+FP+FN > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * TP / (2*TP + FP + FN) if (2*TP + FP + FN) > 0 else 0.0

        metrics_per_class[c] = {
            "TP": int(TP),
            "FP": int(FP),
            "TN": int(TN),
            "FN": int(FN),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }

    print(f"Time taken: {end - start:.2f} seconds")
    print(f"FPS: {fps:.2f}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")

    # Save results
    results_path = f"dataset/{model_name.split('_')[0]}/results"
    os.makedirs(results_path, exist_ok=True)

    results_data = {
        "Total": total,
        "Correct": correct,
        "FPS": fps,
        "Overall_Accuracy": overall_accuracy,
        "Confusion_Matrix": confusion_matrix.tolist(),
        "Metrics_Per_Class": metrics_per_class
    }

    with open(os.path.join(results_path, f"{model_name}.json"), "w") as f:
        json.dump(results_data, f, indent=4)

if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 2:
        print("Usage: python test_multi_model.py <model_name>")
    target_model = argv[1]
    run_model_test(target_model)
