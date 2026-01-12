import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


import keras
from keras import layers

from gcvit.training.config import *
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
        if DATASET_NAME == "cifar10":
            CIFAR10_CLASS_NAMES = [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ]
            (x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()

            x_train = x_train.astype("float32") / 255.0
            x_val = x_val.astype("float32") / 255.0
        
            train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(BATCH_SIZE).prefetch(AUTO)
            val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE).prefetch(AUTO)

            train_data = train_data.map(lambda x, y: (x, tf.one_hot(tf.squeeze(y, -1), depth=n_classes)))
            val_data   = val_data.map(lambda x, y: (x, tf.one_hot(tf.squeeze(y, -1), depth=n_classes)))


            train_data.class_names = CIFAR10_CLASS_NAMES
            val_data.class_names = CIFAR10_CLASS_NAMES
            
        else:
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
