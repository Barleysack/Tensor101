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
import tensorboard


base_dir= 'C:/Users/admin/Desktop/workspace/tensor101/data/'
data_dir = tf.keras.utils.get_file(origin=base_dir, 
                                   fname='hangul', 
                                   )
    




batch_size = 16
img_height = 28
img_width = 28


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,labels='inferred', label_mode='int',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,color_mode='grayscale')


dtype=np.float32,

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,labels='inferred' ,label_mode='int',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,color_mode='grayscale')

class_names = train_ds.class_names
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



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






def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs))              # not shown in the book
        for X_batch in dataset:
            # phase 1 - training the discriminator
            noise = tf.random.normal(shape=[batch_size, codings_size])#랜덤 노이즈
            generated_images = generator(noise)#Generator's noise added-images
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)#Add noise-image to real samples
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size) #y1은 가짜일경우 0, 진짜일 경우 1. 
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)#fit 대신 써먹는 메서드.
            # phase 2 - training the generator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)#죄다 가짜로 맹근 것, 죄다 진짜라고 할테야.
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)#하지만 이진 크로스엔트로피...!
        plot_multiple_images(generated_images, 8)                     # not shown
        plt.show()       


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
    keras.layers.Dense(1, activation="relu")
])
gan = keras.models.Sequential([generator, discriminator])

discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
discriminator.trainable = False
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")
 
 
"""X_train_dcgan = train_ds.reshape(-1, 28, 28, 1) * 2. - 1. # reshape and rescale
#출력범위로 인해 스케일을 조정하고, 채널차원을 추가해야한다. reshape 메소드 학습 요망.
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(X_train_dcgan)
dataset = dataset.shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)"""

train_t_ds = train_ds.astype(np.float32)
train_gan(gan, train_t_ds, batch_size, codings_size)   




