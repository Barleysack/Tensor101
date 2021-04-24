import sklearn
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

from sklearn.manifold import TSNE

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

def plot_image(image):
  plt.imshow(image, cmap="binary")
  plt.axis("off")


(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data() #MNIST 데이터 사용
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))



#두개의 서브모델 만듬
stacked_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),#28X28의 흑백 이미지를 받는다.  784크기 벡터로 만들기 위해 펼치고, 크기가 줄어드는 Dense층 두개에 통과시킨다. 
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu"),
])
#출력은 크기가 30인 벡터를 출력한다. 
stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[30]), #여기서 크기가 30인 코딩을 받는다. 다음 크기가 점점 커지는 Dense층 두개에 통과시킨다.
    keras.layers.Dense(28 * 28, activation="sigmoid"),#784개 유닛으로 확장
    keras.layers.Reshape([28, 28])#최종 출력 크기는 28*28 로 만들어 이 출력이 인코더의 입력과 동일하도록 만든다. 

])


stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])
stacked_ae.compile(loss="binary_crossentropy", #픽셀이 검정일 확률을 구한다. 
                   optimizer=keras.optimizers.SGD(lr=1.5), metrics=[rounded_accuracy])
history = stacked_ae.fit(X_train, X_train, epochs=20,#입력과 타깃으로 사용해 훈련한다.
                         validation_data=(X_valid, X_valid))#검증도 마찬가지.

X_valid_compressed = stacked_encoder.predict(X_valid)
tsne = TSNE()
X_valid_2D = tsne.fit_transform(X_valid_compressed)
X_valid_2D = (X_valid_2D - X_valid_2D.min()) / (X_valid_2D.max() - X_valid_2D.min())




plt.figure(figsize=(10, 8))
cmap = plt.cm.tab10
plt.scatter(X_valid_2D[:, 0], X_valid_2D[:, 1], c=y_valid, s=10, cmap=cmap)
image_positions = np.array([[1., 1.]])
for index, position in enumerate(X_valid_2D):
    dist = np.sum((position - image_positions) ** 2, axis=1)
    if np.min(dist) > 0.02: # if far enough from other images
        image_positions = np.r_[image_positions, [position]]
        imagebox = mpl.offsetbox.AnnotationBbox(
            mpl.offsetbox.OffsetImage(X_valid[index], cmap="binary"),
            position, bboxprops={"edgecolor": cmap(y_valid[index]), "lw": 2})
        plt.gca().add_artist(imagebox)
plt.axis("off")

plt.show()
#어떤 모델에 해당되는지 같이 나타내준 주석박스 추가 버전.
