#######NUMPY VERSION ISSUE!!! TF 2.4.1 does not work perfectly with Numpy 1.20.1
#######Downgraded to 1.19.2
#######MEM ISSUE AGAIN?#######
#######이쪽은 내가 어떻게 건들기가 쉽지 않다. 메모리 단에서 에러가 나는데 내가 뭘 하겠는가?
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
import tensorboard
import os
import numpy as np



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    
    
    print(e)





(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data()
X_train[0][:10]



word_index = keras.datasets.imdb.get_word_index()
id_to_word = {id_ + 3: word for word, id_ in word_index.items()}
for id_, token in enumerate(("<pad>", "<sos>", "<unk>")):
    id_to_word[id_] = token
" ".join([id_to_word[id_] for id_ in X_train[0][:10]])


datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
datasets.keys()
train_size = info.splits["train"].num_examples
test_size = info.splits["test"].num_examples
print(train_size)
#datasets.keys() 메서드를 확인하면 데이터셋으로 적재된 것이 어떤 키워드로 나뉘어있는지 알 수 있다.

for X_batch, y_batch in datasets["train"].batch(2).take(1):
    for review, label in zip(X_batch.numpy(), y_batch.numpy()):
        print("Review:", review.decode("utf-8")[:200], "...")
        print("Label:", label, "= Positive" if label else "= Negative")
        print()


def preprocess(X_batch, y_batch):
    X_batch = tf.strings.substr(X_batch, 0, 300) #배치마다 앞 300자만 평가하도록 자른다.
    X_batch = tf.strings.regex_replace(X_batch, rb"<br\s*/?>", b" ")#/br(줄바꿈태그)를 공백으로 바꾼다. 문자와 작은 따옴표가 아닌 모든 문자를 공백으로 바꾼다.
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")#문자와 작은 따옴표가 아니라면 공백으로 바꾼다.
    X_batch = tf.strings.split(X_batch) #리뷰를 공백으로 나눈다.
    return X_batch.to_tensor(default_value=b"<pad>"), y_batch #상단 라인에서 ragged tensor가 반환되니, 밀집 텐서로 바꾸며 동일한 크기가 되도록패딩토큰 pad로 모든 리뷰를 패딩한다. 


preprocess(X_batch, y_batch) 


from collections import Counter

vocabulary = Counter()
for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
    for review in X_batch:
        vocabulary.update(list(review.numpy()))


#모든 어휘를 모델이 읽을 필요는 없다.
vocab_size = 10000
truncated_vocabulary = [
    word for word, count in vocabulary.most_common()[:vocab_size]]

word_to_id = {word: index for index, word in enumerate(truncated_vocabulary)}
for word in b"This movie was faaaaaantastic".split():
    print(word_to_id.get(word) or vocab_size)

words = tf.constant(truncated_vocabulary)
word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
num_oov_buckets = 1000
table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)


table.lookup(tf.constant([b"Hello movie is awefulsome".split()]))
#사용 빈도 별로 인덱싱 결과, 위 문장은 5176    12     7 10602를 반환하였다.
#10000을 넘는 다는 것은 현 데이터에 없는 단어라는 뜻.

#유용한 함수라고 한다. tft.compute_and_apply_vocabulary() 롹인하라는데..?
#리뷰를 배치로 묶고 preprocess 함수를 사용하여 단어의 시퀀스로 변환한다. 그리고 다음
#앞서 만든 테이블을 사용하는 함수를 만들어 단어를 인코딩한다. 그리고 프리페치도 잊지 않는다.
def encode_words(X_batch, y_batch):
    return table.lookup(X_batch), y_batch

train_set = datasets["train"].repeat().batch(32).map(preprocess)
train_set = train_set.map(encode_words).prefetch(1)




#이로서 훈련세트까지 만드는데 성공하였다.
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
embed_size = 128 
model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size,# 첫 층은 임베딩층. 단어 ID를 임베딩으로 변환한다. 단어 ID당 (vocab_Size+num_oov_buckets) 하나 행과 임베딩 차원당( 128, 여기선) 하나의 열을 가진다. 
    #모델의 입력은 (배치 크기, 타임스텝 수)인 2D텐서지만 임베딩 층의 출력은 [배치크기, 타임스텝수, 임베딩 크기]로 3D 텐서가 된다.
                           mask_zero=True, # 패딩 토큰을 무시하도록 모델에 알려줘야하는데, 이건 임베딩 층에서 해당 매개변수 처리로 충분해짐. 이어지는 층에선 (ID=0)이면 무시해버림.
                           input_shape=[None]),
    keras.layers.GRU(128, return_sequences=True),
    keras.layers.GRU(128),#마지막 타임스텝의 결과값만 가져온다.
    keras.layers.Dense(1, activation="sigmoid") #출력층은 시그모이드 활성화함수를 사용하는 하나의 뉴런이다. 
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]) #좋다, 나쁘다이기에 이진분류에 대한 추정 확률(정확도)를 출력한다.
history = model.fit(train_set, steps_per_epoch=train_size // 32, epochs=5, callbacks = [tensorboard_callback])

"""K = keras.backend
embed_size = 128
inputs = keras.layers.Input(shape=[None])
mask = keras.layers.Lambda(lambda inputs: K.not_equal(inputs, 0))(inputs)
z = keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size)(inputs)
z = keras.layers.GRU(128, return_sequences=True)(z, mask=mask)
z = keras.layers.GRU(128)(z, mask=mask)
outputs = keras.layers.Dense(1, activation="sigmoid")(z)
model = keras.models.Model(inputs=[inputs], outputs=[outputs])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(train_set, steps_per_epoch=train_size // 32, epochs=5)"""#수동 마스킹, 자세히 보면 각각 모든 층에 마스크매개변수를 꼽아놨다.


