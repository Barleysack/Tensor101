import tensorflow_datasets as tfds
import tensorflow as tf 
from tensorflow import keras
from sklearn.datasets import load_sample_image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#귀찮음에 모듈 덩어리를 통으로 복사해왔다. 




def maximum_precisions(precisions):
    return np.flip(np.maximum.accumulate(np.flip(precisions)))



recalls = np.linspace(0, 1, 11)
#재현율당 최대 정밀도를 계산,
precisions = [0.91, 0.94, 0.96, 0.94, 0.95, 0.92, 0.80, 0.60, 0.45, 0.20, 0.10]
#최대 정밀도를 평균하는 것을 평균정밀도(AP) 두개 이상 AP를 계산하고 다시 평균낸것이
#mAP!
#IOU임계점 정의용으로 사용.
#Intersection Over Union
#예측한 바운딩 박스와 타깃 바운딩 박스 사이 중첩 영역을 전체 영역으로 나눈 값. 
#보통 iou 0.5를 기준으로 옳고 그름을 판단한다고;
#구현된 메서드는 tf.keras.metrics.MeanIoU
max_precisions = maximum_precisions(precisions)
mAP = max_precisions.mean()
plt.plot(recalls, precisions, "ro--", label="Precision")
plt.plot(recalls, max_precisions, "bo-", label="Max Precision")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.plot([0, 1], [mAP, mAP], "g:", linewidth=3, label="mAP")
plt.grid(True)
plt.axis([0, 1, 0, 1])
plt.legend(loc="lower center", fontsize=14)
plt.show()