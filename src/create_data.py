import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

a = 10
image, label = tfds.as_numpy(tfds.load(
    'mnist',
    split='test',
    batch_size=-1,
    as_supervised=True,
))





