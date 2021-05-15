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



base_dir= 'C:/Users/admin/Desktop/workspace/tensor101/data/'
data_dir = tf.keras.utils.get_file(origin=base_dir, 
                                   fname='hangul', 
                                   )
 
"""
X = []
y = []

for f in sorted(os.listdir(base_dir)):
    if os.path.isdir(base_dir+f):
        print(f"{f} is a target class")
        for i in sorted(os.listdir(base_dir+f)):
            print(f"{i} is an input image path")
            X.append(base_dir+f+'/'+i)
            y.append(f)
print(X)
print(y) 이렇게하면 저 편안하고 예쁜 메소드 못쓰고 고생해야할텐데 ㅠ 억덕혜 해야할까"""



batch_size = 16
img_height = 28
img_width = 28


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  base_dir,labels='inferred', label_mode='int',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,color_mode='grayscale')




val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  base_dir,labels='inferred' ,label_mode='int',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,color_mode='grayscale')

class_names = train_ds.class_names
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
print(train_ds[1].np)

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
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint("hangulmodel.h5", save_best_only=True)


model.fit(
  train_ds,
  batch_size=batch_size,
  validation_data=val_ds,
  epochs=30,callbacks=checkpoint_cb
)




