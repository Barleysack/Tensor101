import tensorflow_datasets as tfds
import tensorflow as tf 
from tensorflow import keras
from sklearn.datasets import load_sample_image
import numpy as np
from functools import partial
#데이터셋이 생각보다 용량이 커서 코랩으로 구동 예정.

dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
dataset_size = info.splits["train"].num_examples #데이터셋의 데이터 갯수
class_names = info.features["label"].names #클래스 레이블(꽃 이름)
n_classes = info.features["label"].num_classes #클래스 갯수

test_set_raw, valid_set_raw, train_set_raw = tfds.load(
    "tf_flowers",
    split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
    as_supervised=True) #스플릿 API가 책 출판 당시보다 발전. 저렇게만 쓰면 RAW SET은 완성.




def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label #리사이징, XCEPTION 모델의 전처리 함수 사용.


batch_size = 32
train_set = train_set_raw.shuffle(1000).repeat()
train_set = train_set.map(preprocess).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)

base_model = keras.applications.xception.Xception(weights="imagenet",
                                                  include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)
model = keras.models.Model(inputs=base_model.input, outputs=output)

#큰 데이터셋 용량으로 코랩으로 이동.