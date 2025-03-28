import threading
import random
import json
import os
import tensorflow as tf
import time
import cv2
import numpy as np
from utils import CustomPrint, TensorRTModel

shape = 128

def load_model(model_name):
    key = model_name
    path = f"dataset/{key.split('_')[0]}/models/{key}"
    if not os.path.exists(path):
        print(f"Model {key} does not exist!\n"
              f"Please check if the model {key} exists in the right path:\n"
              f"{path}")
        exit(1)
    tensor_engine = TensorRTModel.from_file(path)
    print(f"{key}: done!")
    return tensor_engine

def load_non_trt_model(model_name):
    path = f"dataset/{model_name.split('_')[0]}/models/{model_name}"
    if not os.path.exists(path):
        print(f"Model {model_name} does not exist at {path}!")
        exit(1)
    model = tf.keras.models.load_model(path)
    print(f"{model_name}: Model loaded successfully!")
    return model

def preprocess_image(file_path, shape):
    img = cv2.imread(file_path)
    return img

def model_acctest(model_name, true_files, false_files):
    if 'vit' in model_name:
        model = load_non_trt_model(model_name)
    else:
        model = load_model(model_name)
    print(f"Testing model: {model_name}, True images: {len(true_files)}, False images: {len(false_files)}")
    
    tp, fn, fp, tn = 0, 0, 0, 0
    start = time.time()
    
    for file_path in true_files:
        img = preprocess_image(file_path, shape)
        img = tf.image.resize(img, (shape, shape))
        img = img[:, :, 0]
        img = tf.cast(img, dtype=tf.float32) / 255.0
        img = np.expand_dims(img, axis=2)
        img = np.expand_dims(img, axis=0)
        result = model.predict(img)
        if result >= 0.9:
            tp += 1
        else:
            fn += 1

    for file_path in false_files:
        img = preprocess_image(file_path, shape)
        img = tf.image.resize(img, (shape, shape))
        img = img[:, :, 0]
        img = tf.cast(img, dtype=tf.float32) / 255.0
        img = np.expand_dims(img, axis=2)
        img = np.expand_dims(img, axis=0)
        result = model.predict(img)
        if result >= 0.9:
            fp += 1
        else:
            tn += 1

    end = time.time()
    
    fps = (len(true_files) + len(false_files)) / (end - start)
    print(f"Time taken: {end - start}")
    print(f"FPS: {fps}")
    
    # Save results
    os.makedirs(f"dataset/{model_name.split('_')[0]}/results", exist_ok=True)
    with open(f"dataset/{model_name.split('_')[0]}/results/{model_name}.json", "w") as f:
        json.dump({
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "FPS": fps,
            "Accuracy": (tp + tn) / (tp + tn + fp + fn),
            "Precision": tp / (tp + fp) if tp + fp > 0 else 0.0,
            "Recall": tp / (tp + fn) if tp + fn > 0 else 0.0,
            "F1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
        }, f, indent=4)

def run_model_test(test_model):
    false_files = []
    games = [x for x in os.listdir("dataset/") if test_model.split('_')[0] not in x and os.path.isdir(os.path.join("dataset/", x))]
    for app in games:
        app_path = os.path.join("dataset", app)
        inds = [int(ind) for ind in os.listdir(app_path) if ind.isdigit()]
        if len(inds) == 0:
            continue
        max_ind_path = os.path.join(app_path, str(max(inds)), "false")
        false_files.extend([os.path.join(max_ind_path, file) for file in os.listdir(max_ind_path)])

    true_files = [os.path.join("dataset", test_model.split('_')[0], test_model.split('_')[1], "true", img)
                    for img in os.listdir(os.path.join("dataset", test_model.split('_')[0], test_model.split('_')[1], "true"))]

    random.shuffle(false_files)
    random.shuffle(true_files)

    false_sample = false_files[:50000]
    true_sample = true_files[:50000]

    model_acctest(test_model, true_sample, false_sample)

if __name__ == "__main__":
    models = [
        'clashofclans_1', 'clashofclans_2', 'clashofclans_3', 'clashofclans_4',
        'hillclimb_0', 'hillclimb_1', 'hillclimb_2', 'hillclimb_3', 'hillclimb_4',
        'minecraft_0', 'minecraft_1', 'minecraft_2', 'minecraft_3', 'minecraft_4', 'minecraft_7'
    ]

    threads = []
    for test_model in models:
        thread = threading.Thread(target=run_model_test, args=(test_model,))
        threads.append(thread)

    for i in range(0, len(threads), 3):
        for thread in threads[i:i+3]:
            thread.start()
        for thread in threads[i:i+3]:
            thread.join()
