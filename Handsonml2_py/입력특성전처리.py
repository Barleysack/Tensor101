# All attributes need to be changed into numbers...
#categorical/text attributes need to be converted
#Need to get layer for normalization...
#how to encode categorical to one-hot vector

#어휘 사전이 크면 임베딩을 사용하여 인코딩한다.
#카테고리가 10 이하면 원-핫 인코딩
#카테고리가 50 이상이면 임베딩

#임베딩은 범주를 표현하는 훈련 가능한 밀집 벡터이다. 
#랜덤 초기화 된 후 이후 학습에 따라 훈련된다. 



#JUST USE keras.layers.Textvectorization...
#Or use keras.layers.normalization....

#keras.layers.experimental.preprocessing.TextVectorization
#keras.layers.normalization

#Embedding?- 

import keras
import tensorflow as tf
vocab= ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
num_oov_buckets=2
embedding_dim = 2
embed_init = tf.random.uniform([len(vocab)+num_oov_buckets,embedding_dim])
embedding_matrix = tf.Variable(embed_init)


print(embedding_matrix)

#교재와 노트북이 큰 차이를 보여 교재를 다시 봐야 할 듯.
#TFDS 데이터셋을 사용하면 편안합니다.

import tensorflow_datasets as tfds

dataset = tfds.load(name='mnist')
mnist_train, mnist_test = dataset["train"], dataset["test"]

mnist_train = mnist_train.shuffle(10000).batch(32).prefetch(1)
#이런식으로.....
# 관련 메서드를 알아볼 필요가 있다.
#map 메서드 확인 필요.
'