import tensorflow as tf
import keras


def cube(x):
  return x**3



cube(tf.constant(2.0))

tf_cube = tf.function(cube)

print(tf_cube(tf.constant(2.0)))

@tf.function
def tf_quadraple(x):
  return x * 4


a=tf_quadraple.python_function(9)
print(a)

b=tf_quadraple(9)
print(b)
#TF FUNCTIONS ARE FASTER THAN NORMAL PYTHON FUNCTIONS
#USE DECORATORS TO AUTO-GRA-PH,@tf.function


#concrete_function = tf_cube.get_concrete_function(tf.constant(2.0))
#concrete_function.graph
#concrete_function(tf.constant(2.0))

#concrete_function is tf_cube.get_concrete_function(tf.constant(2.0))

@tf.function
def add_10(x):
    for i in range(10):
        x += 1
    return x


add_10(tf.constant(5.0))
a=add_10.get_concrete_function(tf.constant(5)).graph.get_operations()
print(a)