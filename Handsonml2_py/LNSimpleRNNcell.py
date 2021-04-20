import sklearn
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LayerNormalization
import CLN






class LNsimple(keras.layers.Layer): 
  def __init__(self, units, activation="tanh",**kwargs): #유닛 개수, 활성화 함수 매개변수로.
    super().__init__(**kwargs)
    self.state_size = units
    self.output_size= units#를 설정한 다음
    self.simple_rnn_cell = keras.layers.SimpleRNNCell(units, activation = None)#활성화함수 없이 셀 설정(선형연산-층정규화-활성화함수 순서를 위해)
    self.layer_norm= CLN.LN() #내가 해뒀던거 굴러가나...?
    self.activation = keras.activations.get(activation)
  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
      if inputs is not None:
        batch_size = tf.shape(inputs)[0]
        dtype = inputs.dtype
        return [tf.zeros([batch_size, self.state_size], dtype=dtype)]
  def call(self, inputs, states):# 현재입력과 이전 은닉 상태의 선형 조합 계산
        outputs, new_states = self.simple_rnn_cell(inputs, states)#심플 RNN에서 출력은 은닉상태와 동일
        norm_outputs = self.activation(self.layer_norm(outputs))#newstates[0]은 outputs와 동일, 층정규화 후 활성화함수 적용. 
        return norm_outputs, [norm_outputs] #하나는 새로운 은닉상태가 되고, 하나는 출력이 됨

        

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.RNN(LNsimple(20), return_sequences=True,
                     input_shape=[None, 1]),
    keras.layers.RNN(LNsimple(20), return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])

model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train, epochs=20,
                    validation_data=(X_valid, Y_valid))