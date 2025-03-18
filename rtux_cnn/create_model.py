import argparse
import cv2
import tensorflow as tf
import json
import sys
import os
import time
import numpy as np
from datetime import datetime

import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference
import tensorflow as tf
from tf2onnx import tfonnx, optimizer, tf_loader

sys.path.insert(1, "onnx_to_tf/")
from build_engine import EngineBuilder

def spatial_attention(inputs):
    x = tf.keras.layers.Conv2D(1, kernel_size=1, padding='same')(inputs)
    attention_weights = tf.keras.activations.sigmoid(x)
    return inputs * attention_weights

def auxiliary_classifier(x):
    aux = tf.keras.layers.AveragePooling2D((5, 5), strides=3)(x)
    aux = tf.keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu')(aux)
    aux = tf.keras.layers.Flatten()(aux)
    aux = tf.keras.layers.Dense(1024, activation='relu')(aux)
    aux = tf.keras.layers.Dropout(0.7)(aux)
    aux = tf.keras.layers.Dense(1, activation='sigmoid')(aux)
    return aux

def googlenet_bn(input_shape):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = tf.keras.layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = inception_bn(x, 64, 96, 128, 16, 32, 32)
    aux1 = auxiliary_classifier(x)

    x = inception_bn(x, 128, 128, 192, 32, 96, 64)
    x = spatial_attention(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = inception_bn(x, 192, 96, 208, 16, 48, 64)
    aux2 = auxiliary_classifier(x)

    x = inception_bn(x, 160, 112, 224, 24, 64, 64)
    x = inception_bn(x, 128, 128, 256, 24, 64, 64)
    x = inception_bn(x, 112, 144, 288, 32, 64, 64)
    x = spatial_attention(x)
    
    x = inception_bn(x, 256, 160, 320, 32, 128, 128)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = inception_bn(x, 256, 160, 320, 32, 128, 128)
    x = inception_bn(x, 384, 192, 384, 48, 128, 128)
    x = spatial_attention(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    # x = tf.keras.layers.Dense(16384, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    # x = tf.keras.layers.Dense(8192, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dense(512, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=[x, aux1, aux2])
    return model

def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool):
    conv_1x1 =tf.keras.layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)
    conv_3x3 =tf.keras.layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_3x3 =tf.keras.layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(conv_3x3)
    conv_5x5 =tf.keras.layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_5x5 =tf.keras.layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(conv_5x5)
    pool_proj = tf.keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj =tf.keras.layers.Conv2D(filters_pool, (1, 1), padding='same', activation='relu')(pool_proj)
    output = tf.keras.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=-1)
    return output

def inception_bn(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool):
    conv_1x1 = tf.keras.layers.Conv2D(filters_1x1, (1, 1), padding='same', activation=None)(x)
    conv_1x1 = tf.keras.layers.BatchNormalization()(conv_1x1)
    conv_1x1 = tf.keras.layers.Activation('relu')(conv_1x1)
    
    conv_3x3 = tf.keras.layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation=None)(x)
    conv_3x3 = tf.keras.layers.BatchNormalization()(conv_3x3)
    conv_3x3 = tf.keras.layers.Activation('relu')(conv_3x3)
    conv_3x3 = tf.keras.layers.Conv2D(filters_3x3, (3, 3), padding='same', activation=None)(conv_3x3)
    conv_3x3 = tf.keras.layers.BatchNormalization()(conv_3x3)
    conv_3x3 = tf.keras.layers.Activation('relu')(conv_3x3)
    
    conv_5x5 = tf.keras.layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation=None)(x)
    conv_5x5 = tf.keras.layers.BatchNormalization()(conv_5x5)
    conv_5x5 = tf.keras.layers.Activation('relu')(conv_5x5)
    conv_5x5 = tf.keras.layers.Conv2D(filters_5x5, (5, 5), padding='same', activation=None)(conv_5x5)
    conv_5x5 = tf.keras.layers.BatchNormalization()(conv_5x5)
    conv_5x5 = tf.keras.layers.Activation('relu')(conv_5x5)
    
    pool_proj = tf.keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = tf.keras.layers.Conv2D(filters_pool, (1, 1), padding='same', activation=None)(pool_proj)
    pool_proj = tf.keras.layers.BatchNormalization()(pool_proj)
    pool_proj = tf.keras.layers.Activation('relu')(pool_proj)
    
    output = tf.keras.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=-1)
    return output

def horizontal_stitch(img1, img2):
    height = tf.shape(img1)[0]
    height_float = tf.cast(height, tf.float32)
    split_ratio = tf.random.uniform([], minval=0.25, maxval=0.75, dtype=tf.float32)
    split_point = tf.cast(height_float * split_ratio, tf.int32)
    top_half_img1 = img1[:split_point, :, :]
    bottom_half_img2 = img2[split_point:, :, :]
    final_img = tf.concat([top_half_img1, bottom_half_img2], axis=0)
    final_img = tf.cast(final_img, tf.float32)
    return final_img

def vertical_stitch(img1, img2):
    width = tf.shape(img1)[1]
    width_float = tf.cast(width, tf.float32)
    split_ratio = tf.random.uniform([], minval=0.25, maxval=0.75, dtype=tf.float32)
    split_point = tf.cast(width_float * split_ratio, tf.int32)
    left_half_img1 = img1[:, :split_point, :]
    right_half_img2 = img2[:, split_point:, :]
    stitched_img = tf.concat([left_half_img1, right_half_img2], axis=1)
    stitched_img = tf.cast(stitched_img, tf.float32)
    return stitched_img

def add_random_noise(img, noise_level=None):
    if noise_level is None:
        noise_level = tf.random.uniform([], minval=0.01, maxval=0.04, dtype=tf.float32)
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=noise_level, dtype=tf.float32)
    noisy_img = img + noise
    noisy_img = tf.clip_by_value(noisy_img, 0.0, 1.0)
    return noisy_img

def center_stitch(bg, cen):
    min_size = tf.cast(tf.minimum(tf.shape(bg)[0], tf.shape(bg)[1]) / 10, tf.int32)
    max_size = tf.cast(tf.minimum(tf.shape(bg)[0], tf.shape(bg)[1]) / 2, tf.int32)
    size = tf.random.uniform([], minval=min_size, maxval=max_size, dtype=tf.int32)
    overlay_img = tf.image.resize(cen, [size, size])
    overlay_img = tf.image.convert_image_dtype(overlay_img, bg.dtype)
    x = (tf.shape(bg)[1] - size) // 2
    y = (tf.shape(bg)[0] - size) // 2
    overlay_with_padding = tf.pad(overlay_img, [[y, tf.shape(bg)[0] - size - y], [x, tf.shape(bg)[1] - size - x], [0, 0]])
    result = bg * (1 - tf.cast(tf.greater(overlay_with_padding, 0), bg.dtype)) + overlay_with_padding
    result = tf.cast(result, tf.float32)
    return result

def vert_mirror(img):
    return tf.image.flip_left_right(img)

def hor_mirror(img):
    return tf.image.flip_up_down(img)

def zoom(img):
    zoom_factor = tf.random.uniform([], minval=0.5, maxval=0.9, dtype=tf.float32)
    zoomed_img = tf.image.central_crop(img, zoom_factor)
    zoomed_img = tf.image.resize(zoomed_img, [IMG_SIZE, IMG_SIZE])
    return zoomed_img

def load_pipelined(true_dir, false_dirs, height, width, batch_size):
    def preprocess(file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, [height, width])
        img = tf.cast(img, dtype=tf.float32) / 255.
        return img, label

    def load_stitch_false(file1, file2):
        img1, _ = preprocess(file1, 0)
        img2, _ = preprocess(file2, 0)

        random_index = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
        augmented_image = tf.case([
            (tf.equal(random_index, 0), lambda: horizontal_stitch(img1, img2)),
            (tf.equal(random_index, 1), lambda: vertical_stitch(img1, img2)),
            (tf.equal(random_index, 2), lambda: center_stitch(img1, img2))
        ], exclusive=True)

        augmented_image = tf.image.resize(augmented_image, [height, width])
        augmented_image = tf.ensure_shape(augmented_image, [height, width, 1])
        augmented_image = tf.cast(augmented_image, tf.float32)

        return augmented_image, 0

    def load_augment_true(file_path):
        img, _ = preprocess(file_path, 1)

        random_index = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
        augmented_image = tf.case([
            (tf.equal(random_index, 0), lambda: vert_mirror(img)),
            (tf.equal(random_index, 1), lambda: hor_mirror(img)),
            (tf.equal(random_index, 2), lambda: add_random_noise(img))
        ], exclusive=True)
        augmented_image = tf.ensure_shape(augmented_image, [height, width, 1])
        augmented_image = tf.cast(augmented_image, tf.float32)
        return augmented_image, 1

    def load_augment_false(file_path):
        img, _ = preprocess(file_path, 0)

        random_index = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
        augmented_image = tf.case([
            (tf.equal(random_index, 0), lambda: vert_mirror(img)),
            (tf.equal(random_index, 1), lambda: hor_mirror(img)),
            (tf.equal(random_index, 2), lambda: add_random_noise(img))
        ], exclusive=True)
        augmented_image = zoom(augmented_image)
        augmented_image = tf.ensure_shape(augmented_image, [height, width, 1])
        augmented_image = tf.cast(augmented_image, tf.float32)
        return augmented_image, 0

    # Create datasets for true and false images
    true_paths = tf.data.Dataset.list_files(f"{true_dir}/*", shuffle=True)
    false_paths = tf.data.Dataset.list_files([f"{subdir}/*" for subdir in false_dirs], shuffle=True)

    true_size = sum(1 for _ in true_paths)
    false_size = sum(1 for _ in false_paths)

    print(f"Original True size: {true_size}, False size: {false_size}")

    # Create balanced datasets
    false_paths_1 = false_paths.take(false_size // 2)
    false_paths_2 = false_paths.skip(false_size // 2)
    false_dataset = tf.data.Dataset.zip((false_paths_1, false_paths_2)).map(
        load_stitch_false, num_parallel_calls=tf.data.AUTOTUNE)

    # original_false_dataset = false_paths.map(
    #     lambda x: preprocess(x, 0), num_parallel_calls=tf.data.AUTOTUNE)
    # augmented_false_dataset = false_paths.map(
    #     load_augment_false, num_parallel_calls=tf.data.AUTOTUNE)

    # false_dataset = false_dataset.concatenate(original_false_dataset)
    # false_dataset = false_dataset.concatenate(augmented_false_dataset)

    # false_size = false_size // 2 + false_size * 2

    # true_dataset = true_paths.map(
        # lambda x: preprocess(x, 1), num_parallel_calls=tf.data.AUTOTUNE)

    augmented_false_dataset = false_paths.map(
        load_augment_false, num_parallel_calls=tf.data.AUTOTUNE)

    # false_dataset = false_dataset.concatenate(original_false_dataset)
    false_dataset = false_dataset.concatenate(augmented_false_dataset)

    false_size = false_size // 2 + false_size

    true_dataset = true_paths.map(
        load_augment_true, num_parallel_calls=tf.data.AUTOTUNE)

    # if true_size < false_size // 2:
    if true_size < false_size:
        # additional_true = (false_size // 2 ) - true_size
        additional_true = false_size - true_size
        print(f"Additional true images needed: {additional_true}")
        repeated_true_paths = true_paths.repeat()
        additional_true_dataset = repeated_true_paths.take(additional_true).map(
            load_augment_true, num_parallel_calls=tf.data.AUTOTUNE)
        
        true_dataset = true_dataset.concatenate(additional_true_dataset)
        true_size += additional_true

    print(f"Final False size: {false_size}, Final True size: {true_size}")

    dataset = tf.data.Dataset.sample_from_datasets(
        [false_dataset, true_dataset], weights=[0.5, 0.5], seed=42)

    dataset_size = false_size + true_size

    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    train_dataset_raw = dataset.take(train_size)
    val_dataset_raw = dataset.skip(train_size)

    train_dataset = train_dataset_raw.batch(batch_size).prefetch(tf.data.AUTOTUNE).repeat()
    val_dataset = val_dataset_raw.batch(batch_size).prefetch(tf.data.AUTOTUNE).repeat()

    print(f"Total images in dataset: {dataset_size}")
    print(f"Total images in training dataset: {train_size}")
    print(f"Total images in validation dataset: {val_size}")

    return train_dataset, val_dataset, train_size, val_size

def eval_test(model, path, action):
    test_dir = f"{path}/test/{action}"
    true_dir = f"{test_dir}/true"
    false_dir = f"{test_dir}/false"
    tp, fp = 0, 0
    for img in os.listdir(true_dir):
        image = cv2.imread(os.path.join(true_dir, img))
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = image[:, :, 0]
        image = tf.cast(image, dtype=tf.float32) / 255.
        image = np.expand_dims(image, axis=2)
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)
        print(f"TRUE: {pred}")
        if isinstance(pred, np.ndarray):
            pred = pred[0]
        elif isinstance(pred, list):
            pred = pred[0][0]
        if pred >= 0.95:
            tp += 1
    for img in os.listdir(false_dir):
        image = cv2.imread(os.path.join(false_dir, img))
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = image[:, :, 0]
        image = tf.cast(image, dtype=tf.float32) / 255.
        image = np.expand_dims(image, axis=2)
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)
        print(f"False: {pred}")
        if isinstance(pred, np.ndarray):
            pred = pred[0]
        elif isinstance(pred, list):
            pred = pred[0][0]
        if pred >= 0.95:
            fp += 1
    return tp, fp, len(os.listdir(true_dir)), len(os.listdir(false_dir))

def convert_to_onnx(model_dir, batch_size, input_size):
    saved_model_path = os.path.realpath(model_dir)
    assert os.path.isdir(saved_model_path)
    graph_def, inputs, outputs = tf_loader.from_saved_model(saved_model_path, None, None, "serve", ["serving_default"])
    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graph_def, name="")
    with tf_loader.tf_session(graph=tf_graph):
        onnx_graph = tfonnx.process_tf_graph(tf_graph, input_names=inputs, output_names=outputs, opset=11)
    onnx_model = optimizer.optimize_graph(onnx_graph).make_model("Converted from {}".format(saved_model_path))
    graph = gs.import_onnx(onnx_model)
    graph.inputs[0].shape[0] = batch_size
    graph.outputs[0].shape[0] = batch_size
    if input_size and input_size > 0:
        if graph.inputs[0].shape[3] == 3:
            graph.inputs[0].shape[1] = input_size
            graph.inputs[0].shape[2] = input_size
        elif graph.inputs[0].shape[1] == 3:
            graph.inputs[0].shape[2] = input_size
            graph.inputs[0].shape[3] = input_size
    for i in range(4):
        if type(graph.inputs[0].shape[i]) != int or graph.inputs[0].shape[i] <= 0:
            sys.exit(1)
    for node in [n for n in graph.nodes if n.op == "Clip"]:
        for input in node.inputs[1:]:
            input.values = np.float32(input.values)
    graph.cleanup().toposort()
    model = shape_inference.infer_shapes(gs.export_onnx(graph))
    graph = gs.import_onnx(model)
    graph.cleanup().toposort()
    model = gs.export_onnx(graph)
    onnx_path = os.path.realpath(f"{model_dir}.onnx")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    onnx.save(model, onnx_path)
    return onnx_path

class WarmUpAndCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, total_steps):
        super(WarmUpAndCosineDecay, self).__init__()
        self.initial_lr = tf.cast(initial_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)
        self.cosine_steps = tf.cast(total_steps - warmup_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.initial_lr * (step / self.warmup_steps)
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * (step - self.warmup_steps) / self.cosine_steps))
        cosine_lr = self.initial_lr * cosine_decay
        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: cosine_lr)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_lr.numpy(),
            "warmup_steps": self.warmup_steps.numpy(),
            "total_steps": self.total_steps.numpy()
        }

class ConvergeCheckCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_precision = logs.get('val_precision')
        if val_precision is not None and val_precision < 0.01:
            print(f"Stopping training: val_precision {val_precision} < 0.01")
            # self.model.stop_training = True
        elif val_precision is None:
            print("Warning: val_precision not found in logs")

class PrintMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"\nEpoch {epoch + 1} metrics:")
        for metric, value in logs.items():
            print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--game', type=str, required=True, help='Which game to test')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('-a', '--action', type=int, required=True, help='Which action ind to create')
    parser.add_argument('-B', '--batch', type=int, default=128, help='Size of the batch')
    parser.add_argument('-m', '--memory', type=int, default=None, help='How much memory to allocate to the GPU')
    parser.add_argument('-C', '--calibrate', type=str, required=True, help='Calibration file from br_calib.py')
    parser.add_argument('-t', '--tensorrt', action='store_true', help='Convert model to TensorRT')
    args = vars(parser.parse_args())

    start_time = time.time()
    IMG_SIZE = 128
    KERNEL_SIZE = 3

    if args['memory'] is not None:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=args['memory'])])
                logical_gpus = tf.config.list_logical_devices('GPU')
            except RuntimeError as e:
                print("ERROR:", e)

    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='prc', curve='PR'),
    ]

    tf.random.set_seed(123)
    tf.keras.utils.enable_interactive_logging()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    path = f"dataset/{args['game']}/{args['action']}"
    test_dir = f"dataset/{args['game']}/test"
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} does not exist")
        sys.exit(1)
    print(f"Loading from {path}")

    with open(args['calibrate'], 'r') as f:
        parsed_data = json.load(f)
        br = parsed_data[0]["BRIGHTNESS"]
        cr = parsed_data[0]["CONTRAST"]
        exp = parsed_data[0]["EXPOSURE"]
        new_w, new_h = parsed_data[1]["width"], parsed_data[1]["height"]
    if new_h < new_w:
        samp_img = (new_h, new_h, 3)
    else:
        samp_img = (new_w, new_w, 3)
    samp_img = (IMG_SIZE, IMG_SIZE)

    current_games = ['asphalt', "GenshinImpact", "supercell", "csr"]

    # with open(f"dataset/dataset_false.json") as f:
        # all_false = json.load(f)
    false_lst = [f"{path}/false"]
    # for d in all_false.keys():
        # false_lst.append(d) if f"{args['game']}" not in d else None
    print(false_lst)
    # exit(0)
    model_well = False
    hist_dict = {}
    logs = []   
    train_ds, val_ds, train_size, val_size =  load_pipelined(
        f"{path}/true", false_lst, samp_img[0], samp_img[1], args['batch'])

    steps_per_epoch = train_size // args['batch']
    validation_steps = val_size // args['batch']  

    print(f"Training size: {train_size}, Validation size: {val_size}")
    print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")

    good_train_cnt, all_train_cnt = 0, 0
    best_model = None

    STAMP = f"{args['game']}_{args['action']}"
    print(f"Training {STAMP}")

    bst_model_path = STAMP + '.h5'

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=2, mode='min', 
            verbose=1, min_delta=0.01),
        tf.keras.callbacks.ModelCheckpoint(
            bst_model_path, monitor="val_loss", mode="min", 
            save_best_only=True, save_weights_only=True, 
            verbose=1),
        ConvergeCheckCallback(),
        PrintMetricsCallback()
    ]

    model = googlenet_bn((IMG_SIZE, IMG_SIZE, 1))

    initial_lr = 0.00001
    warmup_steps = 5000
    total_steps = 100000
    lr_schedule = WarmUpAndCosineDecay(initial_lr, warmup_steps, total_steps)

    start_time = time.time()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                loss="binary_crossentropy",
                metrics=METRICS)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args['epochs'],
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps  
    )

    end_time = time.time()
    print(f"Training time: {end_time - start_time}s")
    
    os.makedirs(f"dataset/{args['game']}/models", exist_ok=True)
    with open(f"dataset/{args['game']}/models/{STAMP}_time.txt", "w") as file:
        file.write(f"Training time: {end_time - start_time}s\ntrain size: {train_size}, val size: {val_size}")
    model.load_weights(bst_model_path)

    model.save(f"dataset/{args['game']}/models/{STAMP}")
    model.save(f"dataset/{args['game']}/models/{STAMP}.keras")

    tp, fp, true_cnt, false_cnt = eval_test(model, f"dataset/{args['game']}", args['action'])
    logs.append([tp, fp])

    f1 = 2 * tp / (2 * tp + fp + (true_cnt - tp))
    print(f"TP: {tp}, FP: {fp}, FN: {true_cnt - tp}, TN: {false_cnt - fp}, F1: {f1}")

    if args['tensorrt']:
        onnx_path = convert_to_onnx(f"dataset/{args['game']}/models/{STAMP}", 1, None)

        builder = EngineBuilder(False)
        builder.create_network(onnx_path)
        builder.create_engine(
            f"dataset/{args['game']}/models/{STAMP}.trt",
            'fp32',
            calib_input=None,
            calib_cache="./calibration_cache",
            calib_batch_size=1,
            calib_preprocessor="V2",
            calib_timing_cache="./timing_cache",
        )

    model.save_weights(f"dataset/{args['game']}/models/{STAMP}.weights.h5")
    with open(f"dataset/{args['game']}/models/{STAMP}.txt", "w") as file:
        file.write(f"Model: {STAMP}\n"
                   f"Kernel Size: ({KERNEL_SIZE}, {KERNEL_SIZE})\n"
                   f"Batch Size: {args['batch']}\n"
                   f"Epochs: {args['epochs']}\n"
                   f"Image Size: ({samp_img[0]}, {samp_img[1]}\n"
                   f"TP, FP: {tp}, {fp}\n"
                   f"Hyperparameter tuning successful : {model_well}\n"
                   f"Model creation time: {datetime.now()}\n"
                   f"Training time: ({end_time - start_time}s)\n")

    data = {
        "Model": STAMP,
        "Kernel Size": (KERNEL_SIZE, KERNEL_SIZE),
        "Batch Size": args['batch'],
        "Epochs": args['epochs'],
        "Image Size": (samp_img[0], samp_img[1]),
        "TP": tp,
        "FP": fp,
        "Hyperparameter tuning successful": model_well,
        "Model creation time": f"{datetime.now()}",
    }

    with open(f"dataset/{args['game']}/models/{STAMP}.json", "w") as file:
        json.dump(data, file, indent=4)

    with open(f"dataset/{args['game']}/models/{STAMP}_log.txt", "w") as file:
        for i in logs:
            file.write(f"tp: {i[0]}, fp: {i[1]}\n")

    sys.exit(0)
