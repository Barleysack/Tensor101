import tensorflow_datasets as tfds
import tensorflow as tf 
from tensorflow import keras
from sklearn.datasets import load_sample_image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

china = load_sample_image("china.jpg")/255 #각 픽셀의 강도는 1~255로 표현되어있음. 이를 0~1로 바꿈 . 
flower = load_sample_image("flower.jpg")/255#앞에서 불러왔던 샘플이미지 적재.

images = np.array([china, flower])

images_resized = tf.image.resize(images, [224, 224])

tf.random.set_seed(42)
#은하수를 여행하는 히치하이커를 위한 안내서의 광팬이신듯 하다. 
X = images_resized.numpy()

conv_transpose = keras.layers.Conv2DTranspose(filters=5, kernel_size=3, strides=2, padding="VALID")
#전치 합성곱 층 사용 
#빈 행과 열을 삽입하고 일반적인 합성곱을 사용하는 방식. 
#스트라이드가 클수록 출력이 커집니다. 
output = conv_transpose(X)
k=output.shape

print(k)



def normalize(X):
    return (X - tf.reduce_min(X)) / (tf.reduce_max(X) - tf.reduce_min(X))

fig = plt.figure(figsize=(12, 8))
gs = mpl.gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 2])

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("Input", fontsize=14)
ax1.imshow(X[0])  # plot the 1st image
ax1.axis("off")
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("Output", fontsize=14)
ax2.imshow(normalize(output[0, ..., :3]), interpolation="bicubic")  # plot the output for the 1st image
ax2.axis("off")
plt.show()