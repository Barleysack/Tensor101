import os 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


class RU(keras.layers.Layer): #잔차 블록을 구현해보자
  def __init__(self, filters, strides=1, activation= "relu", **kwargs):
    super().__init__(**kwargs)
    self.activation = keras.activations.get(activation)
    self.main_layers= [
      keras.layers.Conv2D(filters,3,strides=strides, padding="same", use_bias=False),
      keras.layers.BatchNormalization()]
    self.skip_layers = []

    if strides > 1 :
      self.skip_layers = [
        keras.layers.Conv2D(filters,1,strides=strides, padding="same", use_bias=False),
        keras.layers.BatchNormalization()]   

  def call(self,inputs):
    Z = inputs
    for layer in self.main_layers:
      Z =layer(Z)
    skip_Z = inputs
    for layer in self.skip_layers:
      skip_Z = layer(skip_Z)
    return self.activation(Z+skip_Z)   



model = keras.models.Sequential(name="RUmodel") #함수형으로 하면 for문 넣기 편안하다. 생각없이 일반형 써먹다가 두번 일했네.

model.add(keras.layers.Conv2D(64,7,strides=2,input_shape=[224,224,3], #필터수, 수용층 (커널) 크기, 입력 모양
    padding="same", use_bias=False))

model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3,strides=2,padding="same"))
prev_filters= 64
for filters in [64]*3 + [128]*4 +[256]*6+[512]*3:
     strides= 1 if filters == prev_filters else 2
     model.add(RU(filters,strides=strides))
     prev_filters=filters
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10,activation="softmax"))

model.compile()
model.summary()


    