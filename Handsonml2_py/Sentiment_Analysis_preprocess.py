import tensorflow as tf
from tensorflow import keras
import sklearn


(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data()
#x_train은 리뷰들의 리스트. 각 리뷰는 넘파이 정수 배열로 표현. 각 정수는 단어를 나타냄. 구두점 제거 후 
#소문자로 변환된 뒤 공백으로 나누어 빈도에 따라 인덱스가 붙음. 정수가 낮을 수록 자주 등장하는 단어라고 볼 수 있다. 
#정수 0,1,2는 각각 패딩토큰, sos토큰(Start_of_Sequence), Unknown 단어.
#디코딩 방법은 다음과 같다. 


word_index = keras.datasets.imdb.get_word_index()
id_to_word = {id_ + 3: word for word, id_ in word_index.items()}
for id_, token in enumerate(("<pad>", "<sos>", "<unk>")):
    id_to_word[id_] = token
H=" ".join([id_to_word[id_] for id_ in X_train[0][:10]])
print(H)

