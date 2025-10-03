import tensorflow as tf


def check_gpu():
    """
    Check if a GPU is available and print the result.
    """
    gpus = tf.config.list_physical_devices("GPU")
    return len(gpus) > 0
