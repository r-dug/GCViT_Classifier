import tensorflow as tf

import matplotlib.pyplot as plt

from gcvit.training.config import *

class Util():
    @staticmethod
    def configure_for_performance(ds:tf.data.Dataset, AUTOTUNE)->None:
        '''
        a helper function to configure a dataset for better performance...
        current steps:
        - .cache() - stores a cache of the dataset. by default this method stores cache to memory, but our dataset os far to large so we store to disk.
        - .map() - uses our preprocessing layers, applied to the data on every epoch...
        - .shuffle() - shuffles the data, so training is less deterministic, and thus improved 
        - .prefetch() - CPU is utilized while GPU trains, to decrease the bottleneck of IO.

        These ops act on the dataset and aim to improve training efficiency.
        '''
        ds = (ds.prefetch(AUTOTUNE))
            # .shuffle(buffer_size=1000, reshuffle_each_iteration=True)
            # .map(augmenter, num_parallel_calls = AUTOTUNE)
            
        return ds
    @staticmethod
    def gpu_setup()->None:
        # device check. train on gpu.
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPUs: {gpus} ")

        # GPU config
        try:
            tf.config.run_functions_eagerly(True)
            with tf.init_scope():
                print(tf.executing_eagerly())
            print(tf.executing_eagerly())
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            tf.data.experimental.enable_debug_mode()
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        # Virtual devices must be set before GPUs have been initialized
        except RuntimeError as e:
            print(e)
    @staticmethod
    def write_label_file(class_labels:list)->None:
        try:
            with open('labels.txt', 'w') as f:
                for name in class_labels:
                    f.write(f"{name}\n")
        except Exception as e:
            print(e)
    @staticmethod
    def show_image_samples(ds:tf.data.Dataset, class_labels:list[str])->None:
        plt.figure(figsize=(10, 10))
        for images, labels in ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                # normalized = images[i]/255 if tf.reduce_max(images) > 1.0 else images[i] 
                plt.imshow(images[i])
                plt.title(class_labels[tf.argmax(labels[i]).numpy()])
                plt.axis("off")
        plt.show()
    @staticmethod
    def plot_performance(phase:str, training_results:tf.keras.callbacks.History)->None:
        '''A simple logging function for the performance'''
        acc = [0.] + training_results.history['accuracy']
        val_acc = [0.] + training_results.history['val_accuracy']
        loss = training_results.history['loss']
        val_loss = training_results.history['val_loss']
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')
        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,max(max(val_loss), max(loss))])
        plt.title(f'Training and Validation Performance')
        plt.xlabel('epoch')
        plt.suptitle(phase, fontsize=16)
        plt.savefig(os.path.join(f"{RESULTS_DIR}",f"{MODEL_NAME}_{phase}.png"))
