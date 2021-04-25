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


#생성자는 랜덤한 분포 (ex.가우시안 분포)를 입력으로 받고 이미지와 같은 데이터를 출력한다.
#랜덤한 입력은 생성할 이미지의 잠재 표현으로 생각할 수 있다. 생성자는 변이형 디코더와 비스무리하게 만들어지지만....다르다.




codings_size = 30

generator = keras.models.Sequential([#생성자, 디코더처럼 만드네? 훈련 방식은 완전히 다르다고 함. 생성자는 진짜같은 이미지를 만드는 것을 목표로 함.
    keras.layers.Dense(100, activation="selu", input_shape=[codings_size]),
    keras.layers.Dense(150, activation="selu"),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
discriminator = keras.models.Sequential([#판별자 . 생각보다 간단하네? 왜지? 이건 계속 구분해내는게 포인트. 아마 후에 이진분류로 들어가겠지.
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(150, activation="selu"),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(1, activation="sigmoid")
])
gan = keras.models.Sequential([generator, discriminator])


#Phase 1: 판별자를 훈련한다. 실제 이미지 배치를 샘플링해두고 생성자에서 생성한 동일한 수의 가짜 이미지를 합침.
#가짜의 레이블은 0으로 세팅하고 진짜는 1로 배치한다.
#한 스텝동안 이렇게 이진크로스엔트로피를 사용해 훈련한다. 역전파는 판별자의 가중치만을 최적화한다.
#Phase 2:  생성자를 훈련한다. 생성자를 사용해 다른 가짜 이미지 배치를 만들고, 이번에는 배치에 진짜이미지 없이 레이블을 모두 1로 세팅. 
#판별자의 가중치를 동결한다. 따라서 역전파는 생성자의 가중치에만 영향을 미침
#생성자는 이미지를 받고 만드는게 아니다. 판별자에서 넘어오는 그래디언트만 받음. 그것만으로 진짜 이미지 정보가 간접적으로 들어옴


discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
discriminator.trainable = False#두번째 단계에서 판별자를 훈련하지 않도록 지정한다.
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

#생성자는 gan모델을 통해서만 훈련된다. 따로 컴파일하지 않는다. gan 모델도 크로스엔트로피 사용. 
#trainable 속성은 컴파일때만 영향을 미침 . 이후 discriminator.fit()할때는 알아서 훈련됨. train_on_batch때도 마찬가지.
#gan 호출할때는 훈련되지 않음. 
#훈련이 일반적인 반복이 아니라 FIT() 못쓴다. 사용자 정의 훈련 반복문을 만든다.


#1. 이미지를 순회하는 데이터셋 만들기. 
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)


#2

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



train_gan(gan, dataset, batch_size, codings_size, n_epochs=1)

