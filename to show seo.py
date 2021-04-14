import keras.datasets.mnist
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.
##데이터셋 받아오고/ 훈련셋 나누고/ 리사이징

K = keras.backend

class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []
    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)

#최적의 Learning rate를 찾기 위해 학습률과 손실을 기록하고 새로넣어주는 클래스

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
#난수 시-드랑 전에 돌아가던 레이어 치우고-
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

#케라스api 사용한 모델 설계
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-3),
              metrics=["accuracy"])
expon_lr = ExponentialLearningRate(factor=1.005)

#모델컴파일/ 손실함수는 교차...배타...뭐시깽이... 일반적 확률적 경사하강법을 최적화 방법으로 사용, 러닝레이트는 0.001로 시작
#위쪽 클래스 호출 후 적용된 rate만큼 Learning rate 변화

history = model.fit(X_train, y_train, epochs=5,
                    validation_data=(X_valid, y_valid),
                    callbacks=[expon_lr])
#콜백으로 위쪽 Learning rate별 손실 기록해주는 클래스 호출


plt.plot(expon_lr.rates, expon_lr.losses)
plt.gca().set_xscale('log')
plt.hlines(min(expon_lr.losses), min(expon_lr.rates), max(expon_lr.rates))
plt.axis([min(expon_lr.rates), max(expon_lr.rates), 0, expon_lr.losses[0]])
plt.grid()
plt.xlabel("Learning rate")
plt.ylabel("Loss")
plt.show()
#저 위의 클래스 결과를 바탕으로 그래프 출력
#출력되는 결과를 보시면 Learning rate에 따라 loss를 보여주는데, loss가 폭발적으로 증가하는 시점은 learning rate가 최대 0.6~0.7 사이입니다.
#가장 낮은 loss를 보여줄 것으로 예상 되는 것은 0.6~0.7의 learning rate라는건데, 막상 모델을 그렇게 설정할때보다 해당 최대 learning rate의 절반으로 설정했을때
#더 낮은 epoch에서 목표 정확도에 도달하는 것을 볼 수 있습니다.
#요컨대, 왜 이상적으로 보이는 learning rate의 절반으로 파라미터 설정을 해야 모델이 더 똑똑해지는건가요?

