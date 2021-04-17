import tensorflow as tf 

w1, w2 = tf.Variable(5.), tf.Variable(3.)
def f(w1,w2):
  return 3*w1**2+2*w1*w2

with tf.GradientTape() as tape:
    z = f(w1, w2)

tape.gradient(z, [w1, w2])

x = tf.Variable(100.)
with tf.GradientTape() as tape:
    z = my_softplus(x)

