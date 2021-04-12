<<<<<<< HEAD
import numpy as np
import tensorflow as tf
import keras


np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
=======
import numpy as np
import tensorflow as tf
import keras


np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
>>>>>>> 3b5ad9f094c4e08a23bbd39c791d00a4797c73a6
])