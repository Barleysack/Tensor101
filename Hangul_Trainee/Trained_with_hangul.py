import glob
import os
import pathlib
import numpy as np
import PIL
import PIL.Image
import tensorflow as tf
import keras
from tensorflow import keras
from tensorflow.keras import layers
import pandas
from tqdm import tqdm
import matplotlib.pyplot as plt



base_dir= 'C:/Users/admin/Desktop/workspace/tensor101/data/'
data_dir = tf.keras.utils.get_file(origin=base_dir, 
                                   fname='hangul', 
                                   )
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    




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

##케라스에 데이터셋이 클래스가 나뉘어 들어가있지 않았다. 
##직접 클래스가 나뉜 것을 넣어주었다. 
class_names = train_ds.class_names
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



num_classes = 30

model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.5),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(
  optimizer='nadam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint("hangulmodel.h5", save_best_only=True)


model.fit(
  train_ds,
  batch_size=batch_size,
  validation_data=val_ds,
  epochs=30,callbacks=checkpoint_cb
)




