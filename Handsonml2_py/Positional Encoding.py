import sklearn
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib as mpl
#위치인코딩.
#단순히 i번째 위치인코딩이 문장에 있는 i번째 단어의 단어 임베딩에 전해짐. 
#모델을 사용해 위치 인코딩을 학습할 수 있으나 
#일반적으로 여러 주기의 사인/코사인 함수로 정의한 고정된 위치 인코딩을 선호함. 
#하기된 클래스는 위치인코딩 층을 만드는 클래스로서, 
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


class PositionalEncoding(keras.layers.Layer):
  def __init__(self, max_steps, max_dims, dtype=tf.float32, **kwargs):
    super().__init__(dtype=dtype,**kwargs)
    if max_dims % 2 ==1: max_dims+= 1 #max_dims는 짝수
    p, i = np.meshgrid(np.arange(max_steps),np.arange(max_dims//2))
    pos_emb = np.empty((1,max_steps, max_dims))
    pos_emb[0, :, ::2] = np.sin(p / 10000**(2 * i / max_dims)).T
    #P(p,2i)=sin(p/10000^2i/d) 일때 좌변은 문장에서 p번째 위치에 있는 단어를 위한 인코딩의 i번째 원소.
    pos_emb[0, :, 1::2] = np.cos(p / 10000**(2 * i / max_dims)).T
#코사인의 위치 인코딩... 그 다음 단어를 위한 인코딩의 i+1번쨰의 원소겠지?
    self.positional_embedding = tf.constant(pos_emb.astype(self.dtype))
    def call(self, inputs):
        shape = tf.shape(inputs)
        return inputs + self.positional_embedding[:, :shape[-2], :shape[-1]]
  

max_steps = 201
max_dims = 512
pos_emb = PositionalEncoding(max_steps, max_dims)
PE = pos_emb(np.zeros((1, max_steps, max_dims), np.float32))[0].numpy()


#시각화를 위한 긴...작업.
i1, i2, crop_i = 100, 101, 150
p1, p2, p3 = 22, 60, 35
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(9, 5))
ax1.plot([p1, p1], [-1, 1], "k--", label="$p = {}$".format(p1))
ax1.plot([p2, p2], [-1, 1], "k--", label="$p = {}$".format(p2), alpha=0.5)
ax1.plot(p3, PE[p3, i1], "bx", label="$p = {}$".format(p3))
ax1.plot(PE[:,i1], "b-", label="$i = {}$".format(i1))
ax1.plot(PE[:,i2], "r-", label="$i = {}$".format(i2))
ax1.plot([p1, p2], [PE[p1, i1], PE[p2, i1]], "bo")
ax1.plot([p1, p2], [PE[p1, i2], PE[p2, i2]], "ro")
ax1.legend(loc="center right", fontsize=14, framealpha=0.95)
ax1.set_ylabel("$P_{(p,i)}$", rotation=0, fontsize=16)
ax1.grid(True, alpha=0.3)
ax1.hlines(0, 0, max_steps - 1, color="k", linewidth=1, alpha=0.3)
ax1.axis([0, max_steps - 1, -1, 1])
ax2.imshow(PE.T[:crop_i], cmap="gray", interpolation="bilinear", aspect="auto")
ax2.hlines(i1, 0, max_steps - 1, color="b")
cheat = 2 # need to raise the red line a bit, or else it hides the blue one
ax2.hlines(i2+cheat, 0, max_steps - 1, color="r")
ax2.plot([p1, p1], [0, crop_i], "k--")
ax2.plot([p2, p2], [0, crop_i], "k--", alpha=0.5)
ax2.plot([p1, p2], [i2+cheat, i2+cheat], "ro")
ax2.plot([p1, p2], [i1, i1], "bo")
ax2.axis([0, max_steps - 1, 0, crop_i])
ax2.set_xlabel("$p$", fontsize=16)
ax2.set_ylabel("$i$", rotation=0, fontsize=16)

plt.show()