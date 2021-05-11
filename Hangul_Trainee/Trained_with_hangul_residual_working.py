import glob
import os
import pathlib
import numpy as np
import tensorflow as tf
import keras
from tensorflow import keras
from tensorflow.keras import layers
import pandas
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorboard


base_dir= 'C:/Users/admin/Desktop/workspace/tensor101/data/'
data_dir = tf.keras.utils.get_file(origin=base_dir, 
                                   fname='hangul', 
                                   )
    




batch_size = 16
img_height = 28
img_width = 28


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,labels='inferred', label_mode='int',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,color_mode='grayscale')




val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,labels='inferred' ,label_mode='int',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,color_mode='grayscale')

class_names = train_ds.class_names
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



class ResidualBlock(keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(n_neurons, activation="elu",
                                          kernel_initializer="he_normal")
                       for _ in range(n_layers)]

    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return inputs + Z



class ResidualRegressor(keras.models.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.flat1= keras.layers.Flatten()
        self.hidden1 = keras.layers.Dense(30, activation="elu",
                                          kernel_initializer="he_normal")
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out = keras.layers.Dense(output_dim)

    def call(self, inputs):
        Z = self.hidden1(inputs)
        for _ in range(1 + 3):
            Z = self.block1(Z)
        Z = self.block2(Z)
        return self.out(Z)


block1 = ResidualBlock(2, 30)
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal"),
    block1, block1, block1, block1,
    ResidualBlock(2, 30),
    keras.layers.Dense(1)
])



num_classes = 30



model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint("hangulmodel_residual.h5", save_best_only=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
callback=[tensorboard_callback,checkpoint_cb]

model.fit(
  train_ds,
  batch_size=batch_size,
  validation_data=val_ds,
  epochs=30,callbacks=callback
)




