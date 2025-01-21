import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


import keras
from keras import layers

from .config import *
class Data():
    @staticmethod
    def data_augmenter() -> keras.Sequential:
        '''
        Returns a keras Sequential model for augmenting image data. Map this sequential model to training data.
        Returns:
            tf.keras.Sequential
        '''
        augmenter = keras.Sequential([ 
            layers.Rescaling(scale=1./255),
            layers.RandomFlip("horizontal and vertical"),
            layers.RandomRotation(     factor = 0.1, 
                                    fill_mode='nearest'),
            layers.RandomZoom(0.05),
            layers.RandomContrast(0.1, name="rand_contrast"),
            layers.RandomBrightness(0.1)
        ])
        
        return augmenter
    
    
    @staticmethod
    def preprocess_train(dataset: tf.data.Dataset)->tf.data.Dataset:
        """
        Custom data preprocessing function, mapping data augmentation to training image dataset.
        """
        augmenter = Data.data_augmenter()
        return dataset.map(lambda x, y: (augmenter(x, training=True), y)).prefetch(AUTO)
    @staticmethod
    def preprocess_val(dataset: tf.data.Dataset)->tf.data.Dataset:
        """
        Custom data preprocessing function, mapping data augmentation to validation (or test) image dataset.
        """
        def normalize(image, label):
            image = image / 255   # normalization
            return image, label
        return dataset.map(normalize, AUTO).prefetch(AUTO)
    @staticmethod    
    def build_train_and_val()->tf.data.Dataset:
        """
        Returns training and validation datasets based on 
        """
        # create dataset with builtin tf tools... 
        # This function is noted as deprecated, but has been working fine for a long time.
        train_data, val_data = image_dataset_from_directory(  
            DATASET_DIR,
            labels = 'inferred',
            label_mode = 'categorical',
            image_size= IMAGE_SIZE,
            validation_split = 0.2,
            subset = 'both',
            crop_to_aspect_ratio = True,
            batch_size= BATCH_SIZE,
            shuffle=True,
            seed=244
        )

        return train_data, val_data
