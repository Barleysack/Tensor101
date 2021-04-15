import Dcustom
import sys
import sklearn
import tensorflow as tf
import numpy as np 
from tensorflow import keras
import os

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]







model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
for n_hidden in (300, 100, 50, 50, 50):
    model.add(keras.layers.Dense(n_hidden, activation="selu"))
model.add(Dcustom.Densewithme(20, activation = "selu"))
model.add(keras.layers.Dense(8, activation="softmax"))


model.compile(loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.SGD(lr=1e-3),
                metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid))
model.save("my_model_A_customlayer.h5")
model.summary()