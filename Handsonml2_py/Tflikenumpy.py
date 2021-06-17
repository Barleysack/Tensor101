import tensorflow as tf
import sklearn
from tensorflow import keras
import numpy as np
import os
import tensorboard
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

PROJECT_ROOT_DIR = "os.getcwd()"
CHAPTER_ID = "deep"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)



t=tf.constant([[1., 2., 3.], [4., 5., 6.]])
tf.constant(42)


print(t.shape)
print(t[1:,1:])

print(tf.square(t))

print(t @ tf.transpose(t))


b=tf.constant("hello world")
print(b)
u=tf.constant([ord(c) for c in "cccc"])
print(u)


v= tf.Variable([[1,2,3],[4,5,6]])
print(v)

s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]],
                    values=[1., 2., 3.],
                    dense_shape=[3, 4])

print(s)

tf.sparse.to_dense(s)
s2 = s * 2.0