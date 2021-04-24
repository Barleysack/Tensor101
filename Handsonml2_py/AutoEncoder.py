#비지도 학습
#단순히 입력을 출력으로 복사하는 방법을 배우나, 네트워크에 제약을 가해 학습시킨다 
import sklearn
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os


import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

def plot_image(image):
  plt.imshow(image, cmap="binary")
  plt.axis("off")


#3D->2D, 주성분 분석. PCA. 
def generate_3d_data(m, w1=0.1, w2=0.3, noise=0.1):
    angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
    data = np.empty((m, 3))
    data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
    data[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
    data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * np.random.randn(m)
    return data

X_train = generate_3d_data(60)
X_train = X_train - X_train.mean(axis=0, keepdims=0)


np.random.seed(42)
tf.random.set_seed(42)
#케라스 모델은 다른 모델의 층으로 사용할 수 있다. 오토인코더의 출력 개수가 입력의 개수와 동일하다. *3개
encoder = keras.models.Sequential([keras.layers.Dense(2, input_shape=[3])]) #인코더라고 말했지만 평범한 덴스층이네.
decoder = keras.models.Sequential([keras.layers.Dense(3, input_shape=[2])])
autoencoder = keras.models.Sequential([encoder, decoder])

autoencoder.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1.5))

history= autoencoder.fit(X_train,X_train, epochs=20)

#훈련의 부산물 코딩(그 코딩 아님)
codings = encoder.predict(X_train)

fig = plt.figure(figsize=(4,3))
plt.plot(codings[:,0], codings[:, 1], "b.")
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.grid(True)
plt.show()

