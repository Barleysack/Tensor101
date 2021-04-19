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


china = load_sample_image("china.jpg")/ 255 #각 픽셀의 강도는 1~255로 표현되어있음. 이를 0~1로 바꿈 . 
flower = load_sample_image("flower.jpg")/ 255#앞에서 불러왔던 샘플이미지 적재.

images = np.array([china, flower])
model = keras.applications.resnet50.ResNet50(weights="imagenet") #생각보다 학습된 가중치...모델의 크기가 상당하다. 이후 TF LITE에서는 무시할 수 없는 문제가 될 것.

images_resized = tf.image.resize(images, [ 224,224])
inputs = keras.applications.resnet50.preprocess_input(images_resized*255) #픽셀값이 0에서 255 사이로 생각하기에, 앞에서 줄여준걸 다시 곱해줌.


Y_proba = model.predict(inputs) #예측 메소드 ,#처음으로 만난 메모리 문제.  상단부 메모리 할당 관련 메소드를 찾아와야했다.
#5기가 할당으로도 안된다니 이후 어려우면 구글 코랩을 사용하자...
top_K = keras.applications.resnet50.decode_predictions(Y_proba, top = 3 ) 
for image_index in range(len(images)):
  print("이미지 #{}".format(image_index))
  for class_id, name, y_proba in top_K[image_index]:
    print("{}-{:12s} {:.2f}%".format(class_id,name, y_proba*100)) #파이썬 내 문자열 표시 형식 학습 요망. 언뜻 봐서는 포맷 먹이는 것 같은데 모르겠네요;
    print()