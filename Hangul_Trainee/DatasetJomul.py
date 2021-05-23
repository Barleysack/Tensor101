import glob
import os
import pathlib
import numpy as np
import tensorflow as tf
import keras
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt



base_dir= 'C:/Users/finally/Desktop/works/workspace/data/'

 

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

dataset=tf.data.Dataset.from_tensor_slices(X)
#생각해보니 굳이 이렇게까지 할 필요 없을 것 같다.
#딥러닝 컴퓨터 비전 책을 활용해서 다시 한번 도전해보자.
#YOLO 관해서만 알아봐도 괜찮을 듯.