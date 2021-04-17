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

def split_dataset(X, y):
    y_5_or_6 = (y == 5) | (y == 6) # sandals or shirts
    y_A = y[~y_5_or_6]
    y_A[y_A > 6] -= 2 # class indices 7, 8, 9 should be moved to 5, 6, 7
    y_B = (y[y_5_or_6] == 6).astype(np.float32) # binary classification task: is it a shirt (class 6)?
    return ((X[~y_5_or_6], y_A),
            (X[y_5_or_6], y_B))

(X_train_A, y_train_A), (X_train_B, y_train_B) = split_dataset(X_train, y_train)
(X_valid_A, y_valid_A), (X_valid_B, y_valid_B) = split_dataset(X_valid, y_valid)
(X_test_A, y_test_A), (X_test_B, y_test_B) = split_dataset(X_test, y_test)
X_train_B = X_train_B[:200]
y_train_B = y_train_B[:200]

model_A = keras.models.load_model("my_model_A.h5")
#LOAD THE MODEL
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
#GET FIRST -TRAINED LAYER FROM MODEL
model_B_on_A.add(keras.layers.Dense(1 ,name = "dee" ,activation="sigmoid"))

model_A_clone = keras.models.clone_model(model_A)
#USE CLONED MODEL NOT TO INFLUENCE MODEL A
model_A_clone.set_weights(model_A.get_weights())
#COPY WEIGHTS TO CLONE (TO MAKE IT REASON)

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False
#HOLD THE LAYER FROM A TO HOLD THE THINGS HE LEARNED
model_B_on_A.compile(loss="binary_crossentropy",
                     optimizer=keras.optimizers.SGD(lr=1e-3),
                     metrics=["accuracy"])


history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4,
                           validation_data=(X_valid_B, y_valid_B))
#TRAIN MODEL ,FROZEN LAYER

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True
#THAW THE LAYER / AFTER THIS, WE BETTER LOWER THE LR , PREVENTING RE-USED WEIGHTS FROM GETTING F-ED UP
model_B_on_A.compile(loss="binary_crossentropy",
                     optimizer=keras.optimizers.SGD(lr=1e-3, momentum=0.9,nesterov=True),
                     metrics=["accuracy"])
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16,
                           validation_data=(X_valid_B, y_valid_B))
#opitimizers: nesterov/momentum/adagrad(adaptive)
model_B_on_A.evaluate(X_test_B, y_test_B)