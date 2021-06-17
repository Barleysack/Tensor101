import tensorflow as tf
import keras
import CLN 
import numpy as np
import tensorboard
import os
from datetime import datetime


(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255.
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test.astype(np.float32) / 255.

model = keras.models.Sequential([
    CLN.LN(),
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(50, activation="selu"),
    keras.layers.BatchNormalization(),
    
       keras.layers.Dense(10, activation="softmax")    
])

#옵티마이저를 변경하며 텐서보드로 확인했다.계산속도의 차이일 뿐이지 않은가 싶다.
#완전 연결층의 뉴런 수를 줄여보고, 배치 정규화도 시행해보았으나...(활성화함수를 selu로, 완전연결층 이후 배치정규화를 시행하면 좋다길래 시행해보았다)
#꾸준히 오버피팅되는 모습을 보여줌. 

#규제를 추가하고, 계산속도 향상을 위해 네스테로프 가속경사로 최적화 해보았다.
#규제를 추가하자 마침내 오버피팅에서 벗어났다.
#로그 넘버_0416-211308 확인 
#내가 만든(!) 층 정규화 레이어를 추가하자 오버피팅에서 더욱 벗어날 수 있었다. 
#드롭아웃 추가 전 크게 효과가 없던 것 같은데, 잘 모르겠다.
#텐서보드로 확인하자 미세히 성능이 향상되었다.더 낮은 에폭에서 피팅되는듯한 모습.
#모델이 충분히 강력해진듯 하다.


model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=0.005,momentum=0.9, nesterov=True),
              metrics=["accuracy"])

early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)



logdir="logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")



tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
callbacks = [early_stopping_cb,tensorboard_callback ]



model.fit(X_train, y_train, epochs=30,
          validation_data=(X_valid, y_valid),
          callbacks=callbacks)


model.summary()