import sklearn
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt 




(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data() #MNIST 데이터 사용
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]


def plot_multiple_images(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")



def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


def show_reconstructions(model, images=X_valid, n_images=5):
    reconstructions = model.predict(images[:n_images])
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_image(images[image_index])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_image(reconstructions[image_index])
    return
#간만에 다시 읽으니 잘 읽히네...
# 판별자의 풀링층을 스트라이드 합성곱으로 바꾸고 생성자의 풀링층은 트랜스포스...전치합성곱으로 바꿈.
# 생성자와 판별자에 배치 정규화 활용. 생성자의 출력층과 판별자의 입력층은 제외.
#층을 깊게 쌓기 위해 완전연결 은닉층은 제거.
#생성자는 tanh 함수 사용해야할 출력층 빼고 relu사용. 
#판별자는 LeakyRelu 사용.




codings_size = 100

generator = keras.models.Sequential([
    keras.layers.Dense(7 * 7 * 128, input_shape=[codings_size]),#크기 100 코딩을 받아
    keras.layers.Reshape([7, 7, 128]),#7*7*128(6272차원)으로 투영하고 그 크기의 텐서로 변환. 
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="SAME",#업샘플링 관련 좀 찾아봐주실..? 이거 크기가 어떻게 정해지더라...?
                                 activation="selu"),#7*7에서 14*14로 업샘플링 되고 깊이는 128에서 64로 줄어듬. 
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding="SAME",#정규화를 거쳐 스트라이드 2 전치합성곱층에 주입. 다시 14*14에서 28*28로 업샘플링.
                                 activation="tanh"),#출력층에 이를 활용, 출력범위를 -1에서 1까지로 변환. 
])
discriminator = keras.models.Sequential([
    keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="SAME",
                        activation=keras.layers.LeakyReLU(0.2),
                        input_shape=[28, 28, 1]),
    keras.layers.Dropout(0.4),
    keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="SAME",
                        activation=keras.layers.LeakyReLU(0.2)),
    keras.layers.Dropout(0.4),
    keras.layers.Flatten(),
    keras.layers.Dense(1, activation="sigmoid")
])
gan = keras.models.Sequential([generator, discriminator])

discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
discriminator.trainable = False
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")


X_train_dcgan = X_train.reshape(-1, 28, 28, 1) * 2. - 1. # reshape and rescale
#출력범위로 인해 스케일을 조정하고, 채널차원을 추가해야한다. reshape 메소드 학습 요망.
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(X_train_dcgan)
dataset = dataset.shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)


train_gan(gan, dataset, batch_size, codings_size)

#ProGAN 확인.
#모드 붕괴의 개념.: 생성자 출력의 다양성이 줄어든다. 생성자가 특정 클래스 이외 다른 클래스에 대해 잊어먹게 된다. 판별자도 다른걸 잊게 된다...


