import tensorflow as tf
import keras
import numpy as np


X= tf.range(10)
print(X)

#dataset = tf.data.Dataset.from_tensor_slices(X)
dataset= tf.data.Dataset.range(10)

for item in dataset:
  print(item)

