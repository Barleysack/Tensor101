#메모리, 파워, 컴퓨팅 파워 등등의 문제는 이후 사용할 임베디드 시스템에서 상당히 자주 문제가 된다.
#똑같은 resnet을 쓰되, 메모리 사용량을 줄여보자.
#연산량을 줄이면 될까?

import tensorflow as tf 
from tensorflow import keras
from sklearn.datasets import load_sample_image
import numpy as np
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    # 프로그램 시작 시 메모리 할당량 증가 허용하기. 좋은 해결책으로 보이지는 않는다.
    
    print(e)

china = load_sample_image("china.jpg")/255 #각 픽셀의 강도는 1~255로 표현되어있음. 이를 0~1로 바꿈 . 
flower = load_sample_image("flower.jpg")/255#앞에서 불러왔던 샘플이미지 적재.

images = np.array([china, flower])


#더 얕은 층을 가진 모델을 찾아본다.

model = keras.applications.VGG16(weights='imagenet')
images_resized = tf.image.resize(images, [ 224,224])
#inputs = keras.applications.resnet50.preprocess_input(images_resized*255)
inputs=tf.keras.applications.vgg16.preprocess_input(images_resized*255, data_format=None)
Y_proba = model.predict(inputs)
#좀 얕아보여서 돌렸는데, 역시 동일한 문제가 나타났다. 
#@serereuk 선생님께서 텐서RT를 알아보라고 하신다.
#현 교재를 끝낸 후 빠르게 넘어가보자.
top_K=tf.keras.applications.vgg16.decode_predictions(
    Y_proba, top=5
)
for image_index in range(len(images)):
  print("이미지 #{}".format(image_index))
  for class_id, name, y_proba in top_K[image_index]:
    print("{}-{:12s} {:.2f}%".format(class_id,name, y_proba*100)) #파이썬 내 문자열 표시 형식 학습 요망. 언뜻 봐서는 포맷 먹이는 것 같은데 모르겠네요;
    print()