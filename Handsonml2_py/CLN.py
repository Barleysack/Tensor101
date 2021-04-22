import tensorflow as tf
import keras


class LN(keras.layers.Layer):
  def __init__(self, eps=0.001,**kwargs):
    super().__init__(**kwargs)
    self.eps=eps 
  def build(self, batch_input_shape):
        self.alpha = self.add_weight(
            name="alpha", shape=batch_input_shape[-1:],
            initializer="ones")
        self.beta = self.add_weight(
            name="beta", shape=batch_input_shape[-1:],
            initializer="zeros")
        super().build(batch_input_shape)

  def call(self, X): 
    mean,variance=tf.nn.moments(X, axes=-1, keepdims=True)
    return self.alpha*(X-mean)/(tf.sqrt(variance+self.eps))+self.beta

  def compute_output_shape(self, batch_input_shape):
    return batch_input_shape

  def get_config(self):
    base_config = super().get_config()
    return {**base_config, "eps": self.eps}


