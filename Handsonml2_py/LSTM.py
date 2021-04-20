#Long-Short-Term-Memory
import keras
import tensorflow
from tensorflow import keras


#LSTM에 대한 해설이 노트북에 없어 직접 적는다.
#Author tends lose its passion toward book chapter by chapter
#Seriously, you rly had to do like this?
#인간적으로 이 어려운 주제로 용두사미라니

"""
1.LSTM 셀 한개는 이전의 단기상태/현재 입력 벡터를 4개의 분리된 완전연결층에 주입한다.

2.g(t): 주 완전 연결 층은 현재 입력과 이전의 단기상태를 분석하는 일반적인 역할을 담당한다. 

기본 셀에서 일반적으로 존재하는 층이나 장기상태에 가장 중요한 부분이 저장된다. 나머지는 버린다.

3. 나머지 세 완전연결층은 게이트 제어기로서, 로지스틱 활성화 함수를 통하기에 출력이 0부터 1사이이다.
출력은 원소별 곱셈 연산으로 주입되어, 0은 게이트 off, 1은 게이트 on으로 삼는다.

4. f(t) : 삭제 게이트는 장기 상태의 어느 부분이 삭제되어야하는지 제어한다.
5. i(t)  : 입력 게이트는 어느 부분이 장기 상태에 추가되어야하는지 제어한다. 
6. o(t) : 출력 게이트는 장기 상태의 어느 부분을 읽어 이 타임 스텝의 출력으로 내보내야하는지 정한다.


7. 요약해 LSTM은 중요한 입력을 인식하고, 장기 상태에 저장하고, 필요한 기간동안 보존하고, 필요할때 추출을 위해 학습한다. 
시계열, 긴 텍스트, 오디오녹음 등에서 장기 패턴을 잡아내는데 뛰어나다.

8. 케라스의 고수준 API로 구현되어있다. 감사하게도. 소스코드는 말도 안되는 길이이다.
9. 케라스 레이어 클래스는 다음을 중시한다. build(), which is called to define weights.
call() , this is the main part of a layer class. It’s called to perform the computation.
compute_output_shape(), this is pretty self-explanatory what it does.


"""


model = keras.models.Sequential([
    keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])

model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train, epochs=20,
                    validation_data=(X_valid, Y_valid))