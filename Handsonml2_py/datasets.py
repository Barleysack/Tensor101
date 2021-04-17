import tensorflow as tf
import keras
import numpy as np



#dataset = tf.data.Dataset.from_tensor_slices(X)
dataset= tf.data.Dataset.range(10)



for item in dataset:
  print(item)

dataset = dataset.repeat(3).batch(7,drop_remainder=True)
for item in dataset:
  print(item)
dataset = dataset.unbatch()
dataset = dataset.filter(lambda x : x<10)

for item in dataset.take(3):
  print(item)

dataset = tf.data.Dataset.range(10).repeat(3)
dataset = dataset.shuffle(buffer_size=5, seed =11).batch(7,drop_remainder=True)
for item in dataset:
  print(item)