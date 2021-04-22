
from tensorflow import keras

model = keras.models.Sequential([
    keras.layers.GRU(10, return_sequences=True, input_shape=[None, 10]),
    keras.layers.Bidirectional(keras.layers.GRU(10, return_sequences=True))
])

model.summary()
#Bidirectional 층은 GRU층을 복사한다. (반대방향으로)
#그 다음 두 층을 실행하여 그 출력을 연결한다.
#GRU층이 10개의 유닛을 가지면 Bidirectional 층은 20개의 값을 출력한다. 
