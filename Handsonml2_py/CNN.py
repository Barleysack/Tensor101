import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "cnn"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)



mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")

def plot_color_image(image):
    plt.imshow(image, interpolation="nearest")
    plt.axis("off")
china = load_sample_image("china.jpg")/ 255 #각 픽셀의 강도는 1~255로 표현되어있음. 이를 0~1로 바꿈 . 
flower = load_sample_image("flower.jpg")/ 255

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


images = np.array([china, flower])

batch_size, height, width, channels = images.shape


filters = np.zeros(shape=(7, 7, channels, 3), dtype=np.float32)
#0으로 된 행렬 7*7 행렬 두개 추가, 아래에서 필터로서 적용.
filters[:, 3, :, 0] = 1  # vertical line
filters[3, :, :, 1] = 1  # horizontal line



outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")

 #plot 1st image's 2nd feature map
#plt.imshow(outputs[1, :, :, 1], cmap="gray") 
#plt.axis("off") 
#plt.show()


#for image_index in (0, 1):
    #for feature_map_index in (0, 1):
        #plt.subplot(2, 2, image_index * 2 + feature_map_index + 1)
        #plot_image(outputs[image_index, :, :, feature_map_index])

#plt.show()



def crop(images):
    return images[150:220, 130:250]
plot_image(crop(images[0, :, :, 0]))

#plt.show()

#for feature_map_index, filename in enumerate(["china_vertical", "china_horizontal"]):
    #plot_image(crop(outputs[0, :, :, feature_map_index]))

    #plt.show()

#conv = keras.layers.Conv2D(filters=32, kernel_size=3,strides=1,padding="same",activation="relu")
#plot_image(filters[:, :, 0, 0])
#plt.show()
#plot_image(filters[:, :, 0, 1])
#plt.show()



max_pool = keras.layers.MaxPool2D(pool_size=2)
cropped_images = np.array([crop(image) for image in images], dtype=np.float32)
output = max_pool(cropped_images)


fig = plt.figure(figsize=(12, 8))
gs = mpl.gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[2, 1])

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("Input", fontsize=14)
ax1.imshow(cropped_images[0])  # plot the 1st image
ax1.axis("off")
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("Output", fontsize=14)
ax2.imshow(output[0])  # plot the output for the 1st image
ax2.axis("off")
save_fig("china_max_pooling")
plt.show()



class DepthMaxPool(keras.layers.Layer):
    def __init__(self, pool_size, strides=None, padding="VALID", **kwargs):
        super().__init__(**kwargs)
        if strides is None:
            strides = pool_size
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
    def call(self, inputs):
        return tf.nn.max_pool(inputs,
                              ksize=(1, 1, 1, self.pool_size),
                              strides=(1, 1, 1, self.pool_size),
                              padding=self.padding)

depth_pool = DepthMaxPool(3)#깊이 방향 최대 풀링은 불변성 학습에 적합하다. 아래는 gpu로 돌렸을때의 에러. 불변성 학습이 적합- 이미지가 두께/밝기, 왜곡, 색상에서 변동해도 불변성을 학습할 수 있다. 
with tf.device("/cpu:0"): #tensorflow.python.framework.errors_impl.UnimplementedError: Depthwise max pooling is currently only implemented for CPU devices. [Op:MaxPool]
    depth_output = depth_pool(cropped_images)
k=depth_output.shape
print(k)


avg_pool = keras.layers.AvgPool2D(pool_size=2)
#Not maximum values, only averages. Performance is worse than maxpooling. Yet its losses are less than maxpooling. 
#max pooling cost less.


global_avg_pool = keras.layers.GlobalAvgPool2D()
#특성층 전역에 대한 평균 값 계산. 정보의 파괴를 가져온다. 
#유용한 점이 있다고 하는데 모르니 @serereuk께 여쭈어보자. 

global_avg_pool(cropped_images)
