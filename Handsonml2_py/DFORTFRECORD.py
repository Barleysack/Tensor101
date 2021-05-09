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


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    
    
    print(e)
base_dir= 'C:\\Users\\Finally\\Desktop\\Works\\Workspace\\Misc\\data\\hangul\\'
data_dir = tf.keras.utils.get_file(origin=base_dir, 
                                   fname='hangul', 
                                   )
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    




batch_size = 16
img_height = 28
img_width = 28


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir, label_mode='categorical',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir, label_mode='categorical',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

##케라스에 데이터셋이 클래스가 나뉘어 들어가있지 않았다. 
##직접 클래스가 나뉜 것을 넣어주었다. 
class_names = train_ds.class_names
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def image_example(imageString, label, name):
    imageShape = tf.image.decode_jpeg(imageString).shape

   feature = {
        'image_raw': _bytes_feature(imageString), #이미지는 0~255의 3차원값들
        'landmark_id': _int64_feature(label), #
        'id':_bytes_feature(name) #이미지 이름
    }