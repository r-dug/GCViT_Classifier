# config_toy_cifar10.py
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Debugging flags
DEBUG_MODEL = False
DEBUG_DATA = False

# ---- Toy run identity ----
VERSION = "toy_0_0_1"
BASE_MODEL = "gcvitxxtiny"   # smallest/fastest variant
MODEL_NAME = f"toy_cifar10_{VERSION}_{BASE_MODEL}"

# ---- Output dirs (safe defaults) ----
# Keep outputs local to repo to avoid "../" assumptions.
ASSETS_DIR = "./assets"
TF_DIR = os.path.join(ASSETS_DIR, "models", "tf")
TFLITE_DIR = os.path.join(ASSETS_DIR, "models", "tflite")
RESULTS_DIR = os.path.join(ASSETS_DIR, "training_performance")
CKPT_DIR = os.path.join(ASSETS_DIR, "checkpoints")

os.makedirs(TF_DIR, exist_ok=True)
os.makedirs(TFLITE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(CKPT_DIR, f"{MODEL_NAME}.weights.h5")
TF_MODEL_PATH = os.path.join(TF_DIR, f"{MODEL_NAME}.keras")
MODEL_JSON_PATH = os.path.join(TF_DIR, f"{MODEL_NAME}.json")
TFLITE_MODEL_PATH = os.path.join(TFLITE_DIR, f"{MODEL_NAME}.tflite")

# ---- Dataset selection ----
# Your pipeline can branch on this. No local files required.
DATASET_NAME = "cifar10"   # use tf.keras.datasets.cifar10
DATASET_DIR = None         # unused for CIFAR-10

# ---- Model input dims ----
# CIFAR-10 is 32x32. If your model expects 224x224, you can resize in the data pipeline.
IMAGE_SIZE = (32, 32)
INPUT_SHAPE = (32, 32, 3)

# ---- Training hyperparameters (fast) ----
BATCH_SIZE = 64
EPOCHS = 2

# Reproducibility
SEED = 42
tf.keras.utils.set_random_seed(SEED)

# CPU-friendly
AUTO = tf.data.AUTOTUNE

MODEL_CONFIGS = {
    "gcvitxxtiny": {
        "window_size": (7, 7, 14, 7),
        "embed_dim": 64,
        "depths": (2, 2, 6, 2),
        "num_heads": (2, 4, 8, 16),
        "mlp_ratio": 3.0,
        "path_drop": 0.1,
    },
    # (Optional) keep others if you want, but toy should default to xxtiny.
}

# ---- Callbacks (lightweight) ----
plateau = ReduceLROnPlateau(
    monitor="val_loss",
    mode="min",
    patience=1,
    min_lr=1e-7,
    factor=0.5,
    min_delta=0.01,
    verbose=1,
)

checkpointer = ModelCheckpoint(
    filepath=CHECKPOINT_PATH,
    verbose=1,
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    save_weights_only=True,
)

convergence = EarlyStopping(
    monitor="val_accuracy",
    min_delta=0.001,
    patience=2,
    verbose=1,
    mode="max",
    restore_best_weights=True,
)
