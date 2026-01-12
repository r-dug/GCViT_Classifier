import tensorflow as tf

import keras
from keras import ops

import numpy as np

from gcvit.model import GCViT
from gcvit.training.util import Util
from gcvit.training import Data
from gcvit.training.config import *
import os

def load_initial_weights(model: keras.Model) -> None:
    """
    Load weights into `model`.

    Priority:
      1) Local fine-tune checkpoint at CHECKPOINT_PATH (if present)
      2) Downloaded ImageNet pretrained checkpoint from ckpt_link
    """
    # 1) Local checkpoint (your fine-tuning)
    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        print(f"[weights] Loading local checkpoint: {CHECKPOINT_PATH}")
        model.load_weights(CHECKPOINT_PATH, skip_mismatch=True)
        return

    # 2) Download pretrained weights/model
    print(f"[weights] Local checkpoint not found. Downloading pretrained: {ckpt_link}")
    ckpt_path = keras.utils.get_file(
        fname=ckpt_link.split("/")[-1],
        origin=ckpt_link,
        cache_subdir="gcvit_pretrained",
    )
    print(f"[weights] Downloaded to: {ckpt_path}")

    # Try loading as a full saved model first (.keras usually is)
    try:
        pretrained = keras.models.load_model(ckpt_path, compile=False)
        print("[weights] Loaded pretrained as a full Keras model; transferring weights...")
        model.set_weights(pretrained.get_weights())
        return
    except Exception as e:
        print(f"[weights] Could not load as full model ({type(e).__name__}: {e}). "
              f"Trying model.load_weights(..., skip_mismatch=True)")

    # Fallback: try treating it as weights
    model.load_weights(ckpt_path, skip_mismatch=True)

# Model Configs
MODEL_CONFIG = MODEL_CONFIGS[BASE_MODEL]

# link to keras file with pretrained weights from image_net
ckpt_link = (
    f"https://github.com/awsaf49/gcvit-tf/releases/download/v1.1.6/{BASE_MODEL}.keras"
)

def train(model: keras.Model, train_layers:int, phase:int, train_data:tf.data.Dataset, val_data:tf.data.Dataset):
    """
    Helper function for model training to clean up code.
    """

    # freeze non training layers
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-train_layers:]:
        layer.trainable = True
    model.summary((224, 224, 3))
    
    # compile and fit model
    model.compile(
        optimizer= keras.optimizers.Adam(learning_rate=1e-4), 
        loss=tf.keras.losses.CategoricalCrossentropy(), 
        metrics=['accuracy']
    )
    history = model.fit(
        train_data, 
        validation_data=val_data, 
        epochs=EPOCHS, 
        callbacks=[plateau, checkpointer, convergence],
        verbose=1
    )
    # save model and training performance
    try:
        model.save(TF_MODEL_PATH)
        Util.plot_performance(
            phase=f"phase_{phase}_", 
            training_results=history
            )
    except Exception as e:
        print(e)

def main():
    train_data, val_data = Data.build_train_and_val()
    
    class_names = train_data.class_names
    n_classes = len(class_names)

    if DEBUG_DATA == True:
        Util.show_image_samples(train_data, class_names)
        Util.show_image_samples(val_data, class_names)
    
    Util.write_label_file(class_names)
    
    train_data = Data.preprocess_train(train_data)
    val_data = Data.preprocess_val(val_data)
    
    # Build Model
    model = GCViT(**MODEL_CONFIG, num_classes=n_classes)
    input = ops.array(np.random.uniform(size=(1, 224, 224, 3))) # Defines expected input 
    out = model(input)
    load_initial_weights(model)
    # training in phases, iteratively deepening the layers we train
    # train(model=model, train_layers=1, phase=1, train_data=train_data, val_data=val_data)
    train(model=model, train_layers=3, phase=2, train_data=train_data, val_data=val_data)
    train(model=model, train_layers=4, phase=3, train_data=train_data, val_data=val_data)
    # train(model=model, train_layers=5, phase=4, train_data=train_data, val_data=val_data)

if __name__ == "__main__":
    
    main()
