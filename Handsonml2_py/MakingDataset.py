import glob
import os
import pathlib
import numpy as np
import PIL
import PIL.Image
import tensorflow as tf

base_dir= 'C:/Users/Finally/Desktop/Works/Workspace/Misc/data/hangul/'
data_dir = tf.keras.utils.get_file(origin=base_dir, 
                                   fname='hangul', 
                                   )
'''fnames = glob.glob(base_dir+'*.jpg')

  

    
try:
  for i in range(len(fnames)):
    x=fnames[i]
    y=x[58:60]
    os.makedirs('./data/hangul/')
    print("folder created")

  
except:
  for i in range(len(fnames)):
    x=fnames[i+1]
    y=x[58:60]
    os.makedirs('./data/hangul')
    print("folder created")'''

    
batch_size = 32
img_height = 28
img_width = 28


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=1,
  image_size=(img_height, img_width),
  batch_size=batch_size)



