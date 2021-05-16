import glob
import os
import pathlib
import numpy as np
import tensorflow as tf
import keras
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt



base_dir= 'C:/Users/admin/Desktop/workspace/tensor101/data/'

 

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
print(y)

