import tensorflow as tf
import keras




class Densewithme(keras.layers.Layer):
  def __init__(self,units,activation=None, **kwargs):
    super().__init__(**kwargs)
    self.units = units
    self.activation = keras.activations.get(activation)
  def build(self, batch_input_shape):
        self.kernel = self.add_weight(
            name="kernel", shape=[batch_input_shape[-1],self.units],
            initializer="glorot_normal")
        self.bias = self.add_weight(
          name ="bias", shape=[self.units],initializer="zeros")
        super().build(batch_input_shape)

  def call(self,X):
     return self.activation(X @ self.kernel + self.bias)

  def compute_output_shape(self, batch_input_shape):
     return tf.TensorShape(batch_input_shape.as_list()[:-1]+[self.units])


  def get_config(self):
        base_config =   super().get_config()
        return {**base_config, "units": self.units, 
             "activation":keras.activations.serialize(self.activation)}
    
class MyMultilayer(keras.layers.Layer):
  def call(self, X):
    X1, X2 = X
    return( [X1+X2, X1 * X2, X1/X2])

  def compute_output_shape(self,batch_input_shape):
    b1,b2 = batch_input_shape
    return[b1,b1,b1]


class ResidualBlock(keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(n_neurons, activation="elu",
                                          kernel_initializer="he_normal")
                       for _ in range(n_layers)]

    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return inputs + Z


class WADmodel(keras.models.Model): #10장 서브클래스 모델 재사용, 근데 이건 인풋이 2개; 아까 만든 레이어와 연결해보는...
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.hidden3 = Densewithme(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)

    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        hidden3 = self.hidden3(input_A)
        concat = keras.layers.concatenate([hidden3, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output