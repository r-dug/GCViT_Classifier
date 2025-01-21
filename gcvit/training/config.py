import os

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Debugging flags
DEBUG_MODEL = False
DEBUG_DATA = False

# my model
VERSION = "0_0_3"
BASE_MODEL = "gcvitxxtiny"
MODEL_NAME = f"ascocarp_{VERSION}_{BASE_MODEL}"

# Important dirs and files
DATASET_DIR = "/home/richard/Documents/ML_AI/Computer Vision/Classification/DATA/ascocarp_445" #replace with cloud dataset if on GCP
TF_DIR = "./../assets/models/tf"
TFLITE_DIR = "./../assets/models/tflite/"
RESULTS_DIR = "./../assets/training_performance"

CHECKPOINT_PATH = f"./../assets/checkpoints/{MODEL_NAME}.weights.h5"
TF_MODEL_PATH = os.path.join(TF_DIR, f"{MODEL_NAME}.keras")
MODEL_JSON_PATH = os.path.join(TF_DIR, f"{MODEL_NAME}.json")
TFLITE_MODEL_PATH = os.path.join(TFLITE_DIR, f"{MODEL_NAME}.tflite")

# Model input dims
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224,224,3)

# Training hyper parameters
BATCH_SIZE = 4
EPOCHS = 10

# AUTOTUNING constant
AUTO = tf.data.AUTOTUNE

MODEL_CONFIGS = {
    "gcvitxxtiny": {
        "window_size": (7, 7, 14, 7),
        "embed_dim": 64,
        "depths": (2, 2, 6, 2),
        "num_heads": (2, 4, 8, 16),
        "mlp_ratio": 3.0,
        "path_drop": 0.2,
    },
    "gcvitxtiny": {
        "window_size": (7, 7, 14, 7),
        "embed_dim": 64,
        "depths": (3, 4, 6, 5),
        "num_heads": (2, 4, 8, 16),
        "mlp_ratio": 3.0,
        "path_drop": 0.2,
    },
    "gcvittiny": {
        "window_size": (7, 7, 14, 7),
        "embed_dim": 64,
        "depths": (3, 4, 19, 5),
        "num_heads": (2, 4, 8, 16),
        "mlp_ratio": 3.0,
        "path_drop": 0.2,
    },
    "gcvitsmall": {
        "window_size": (7, 7, 14, 7),
        "embed_dim": 96,
        "depths": (3, 4, 19, 5),
        "num_heads": (3, 6, 12, 24),
        "mlp_ratio": 2.0,
        "path_drop": 0.3,
        "layer_scale": 1e-5,
    },
    "gcvitbase": {
        "window_size": (7, 7, 14, 7),
        "embed_dim": 128,
        "depths": (3, 4, 19, 5),
        "num_heads": (4, 8, 16, 32),
        "mlp_ratio": 2.0,
        "path_drop": 0.5,
        "layer_scale": 1e-5,
    },
    "gcvitlarge": {
        "wIndow_size": (7, 7, 14, 7),
        "embed_dim": 192,
        "depths": (3, 4, 19, 5),
        "num_heads": (6, 12, 24, 48),
        "mlp_ratio": 2.0,
        "path_drop": 0.5,
        "layer_scale": 1e-5,
    },
}

# callback configurations
plateau = ReduceLROnPlateau(    monitor="val_loss", 
                                mode="min", 
                                patience=3,
                                min_lr=1e-7, 
                                factor=0.5, 
                                min_delta=0.01,
                                verbose=1)

checkpointer = ModelCheckpoint( filepath=CHECKPOINT_PATH, 
                                verbose=1, 
                                save_best_only=True,
                                monitor="val_accuracy",
                                mode="max",
                                save_weights_only=True)

convergence = EarlyStopping(    monitor="val_accuracy",
                                min_delta=0.001,
                                patience=5,
                                verbose=1,
                                mode="max",
                                baseline=None,
                                restore_best_weights=True,
                                start_from_epoch=5)
