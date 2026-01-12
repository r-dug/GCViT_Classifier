import tensorflow as tf

import keras
from keras import ops

import numpy as np

from gcvit.model import GCViT
from gcvit.training.util import Util
from gcvit.training import Data
from gcvit.training.config import *

# set up gpu so it won't run out of memory?
# util.gpu_setup()

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
    # Load Weights
    ckpt_path = keras.utils.get_file(ckpt_link.split("/")[-1], ckpt_link)
    model.load_weights(CHECKPOINT_PATH, skip_mismatch=True)

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


if __name__ == "__main__":
    
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

    # training in phases, iteratively deepening the layers we train
    # train(model=model, train_layers=1, phase=1, train_data=train_data, val_data=val_data)
    train(model=model, train_layers=3, phase=2, train_data=train_data, val_data=val_data)
    train(model=model, train_layers=4, phase=3, train_data=train_data, val_data=val_data)
    # train(model=model, train_layers=5, phase=4, train_data=train_data, val_data=val_data)
