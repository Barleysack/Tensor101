import tensorflow as tf
from tensorflow import keras
import numpy as np
import os




(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train_full = X_train_full[..., np.newaxis].astype(np.float32) / 255.
X_test = X_test[..., np.newaxis].astype(np.float32) / 255.
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_new = X_test[:3]




model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28, 1]),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-2),
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

np.round(model.predict(X_new), 2)

model_version = "0001"
model_name = "my_mnist_model"
model_path = os.path.join(model_name, model_version)
model_path



#모델과 함께 이름과 버전을 포함한 경로를 전달하면 이 함수는 이 경로에 모델의 계산 그래프와 가중치를 저장한다.
#model.save(model_path)이것도 가능하다. 
#전처리까지 모두 포함하는게 보통 좋은 생각이다. 

tf.saved_model.save(model, model_path)

saved.model=tf.saved_model.load(model_path)
y_pred=saved_mode(tf.constant(X_new,dtype=tf.float32))

#CLI 명령어들 또한 존재한다. 
!saved_model_cli show --dir {model_path} --all #모델에 관련한 정보를 보이라
#해당 모델을 ~의 인풋을 사용해 굴려봐라!
!saved_model_cli run --dir {model_path} --tag_set serve \ 
                     --signature_def serving_default    \
                     --inputs {input_name}=my_mnist_tests.npy

 #요런식으로 해당 모델을 불러낼 수 있다.

#도커를 사용할 수 있다. 도커에 관련해서 좀 배워야하지 않나? 
#일종의 가상머신과 비슷하지만 훨씬 빠르고 가볍다. 
#배포 관련 내용은 추가적으로 TinyML에서 확인 바람.




