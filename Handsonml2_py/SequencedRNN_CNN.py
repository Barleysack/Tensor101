import sklearn
import tensorflow as tf
from tensorflow import keras
import numpy as np 
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
#중간 50짜리 세트 만들때 뭐 문제가 있는듯?
#코랩으로 재시도 요망

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rnn"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)





#시계열 데이터셋을 예측해보자.
def generate_time_series(batch_size, n_steps): #데이터셋 생성
  #batch_size로 요청한 만큼 n_steps길이의 시계열을 만듬. 각 시계열에는 타임스텝마다 하나의 값만 존재.
  #시계열 데이터가 단변량인 상태. 
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # + noise
    return series[..., np.newaxis].astype(np.float32)#차원을 더해줌. 어떻게 바뀌는건지 이해가 가지 않는다. 
#반환하는 것은 넘파이 배열, (Batch_size, n_steps, 1)#



n_steps = 50
series = generate_time_series(10000, n_steps + 1) #시계열 데이터의 다음을 예측한다.
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]
#데이터셋 나눔


def plot_series(series, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$"):
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bx", markersize=10)
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "ro")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps + 1, -1, 1])

fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))
for col in range(3):
    plt.sca(axes[col])
    plot_series(X_valid[col, :, 0], y_valid[col, 0],
                y_label=("$x(t)$" if col==0 else None))
#save_fig("time_series_plot")
#같은 화면으로 봐야지.plt.show()"""

y_pred = X_valid[:, -1]
np.mean(keras.losses.mean_squared_error(y_valid, y_pred))

plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
plt.show()

print(y_pred.shape)

#시계열 마지막 값을 그대로 예측- Naive forecasting
model = keras.models.Sequential([
  keras.layers.Flatten(input_shape=[50,1]),
  keras.layers.Dense(1)
],name="SimpleDensyDenseModelforLR")

model.compile(loss="mse", optimizer="adam")
"""history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))

model.evaluate(X_valid, y_valid)
model.summary()"""


"""def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)

plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()"""
#위의 모델 플로팅...

y_pred = model.predict(X_valid)
plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
plt.show()


#위의 모델로 그려낸 prediction 플로팅... 근데 진짜 MATPLOT언제 배우지?

#간단한 RNN 고수준 API. 같은 에폭인데 생각보다 시간이 많이 걸리네? 

""""np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.SimpleRNN(1, input_shape=[None, 1],
])

optimizer = keras.optimizers.Adam(lr=0.005)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))


model.evaluate(X_valid, y_valid)
model.summary()"""


#Forecasting Several Steps Ahead

keras.backend.clear_session()
np.random.seed(43) # not 42, as it would give the first series in the train set

series = generate_time_series(1, n_steps + 10) #10스텝 뒤를 예측한다.
X_new, Y_new = series[:, :n_steps], series[:, n_steps:]
X = X_new
for step_ahead in range(10):
    y_pred_one = model.predict(X[:, step_ahead:])[:, np.newaxis, :]
    X = np.concatenate([X, y_pred_one], axis=1)

Y_pred = X[:, n_steps:]
print(Y_pred.shape)


def plot_multiple_forecasts(X, Y, Y_pred):
    n_steps = X.shape[1]
    ahead = Y.shape[1]
    plot_series(X[0, :, 0])
    plt.plot(np.arange(n_steps, n_steps + ahead), Y[0, :, 0], "ro-", label="Actual")
    plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred[0, :, 0], "bx-", label="Forecast", markersize=10)
    plt.axis([0, n_steps + ahead, -1, 1])
    plt.legend(fontsize=14)



np.random.seed(42)
tf.random.set_seed(42)
np.random.seed(42)
#Need Debugging here 
n_steps = 50
series = generate_time_series(10000, n_steps + 10)
X_train, Y_train = series[:7000, :n_steps], series[:7000, -10:, 0]
X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:, 0]
X_test, Y_test = series[9000:, :n_steps], series[9000:, -10:, 0]

X = X_valid
for step_ahead in range(10):
    y_pred_one = model.predict(X[:, step_ahead:])[:, np.newaxis, :]
    X = np.concatenate([X, y_pred_one], axis=1)
#중간 매개변수 step_ahead가 아예 들어가있지 않았다. 
#for문이 돌지 못했다는 뜻.

Y_pred = X[:, n_steps:, 0]


model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(10)
])
#해당 모델을 시퀀스-투-시퀀스로 바꾼다면 모든 층에 return_sequence를 트루로, 이후 모든 타임 스텝에서 출력을 dense에 지정해야 한다. 
#해당 API는 keras.layers.TimeDistributed(keras.layers.Dense(n))
#이것은 타임스텝마다 덴스가 독립적으로 적용하고 모델이 벡터가 아닌 시퀀스를 출력한다는 것을 드러낸다. 
#시퀀스의 개념을 정립할 필요성이 있어 보인다. 
#현재 컴파일 하면 출력 텐서가 호환되지 않는다고 하는데, 이를 해당 api로 바로잡는다
#출력이 [배치크기*타임스텝수,출력차원]====> [배치크기, 타임스텝수, 출력차원 ]
model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, Y_train, epochs=20,
                    validation_data=(X_valid, Y_valid))

np.random.seed(43)

series = generate_time_series(1, 50 + 10)
X_new, Y_new = series[:, :50, :], series[:, -10:, :]
Y_pred = model.predict(X_new)[..., np.newaxis]
plot_multiple_forecasts(X_new, Y_new, Y_pred)
plt.show()
#이후 10스텝을 예상할 모델. RNN이 계산 시간이 더 걸리는 이유가 뭘까?


np.random.seed(43)

series = generate_time_series(1, 50 + 10)
X_new, Y_new = series[:, :50, :], series[:, -10:, :]
Y_pred = model.predict(X_new)[..., np.newaxis]


plot_multiple_forecasts(X_new, Y_new, Y_pred)
plt.show()