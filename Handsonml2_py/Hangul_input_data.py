import os
from glob import glob
from PIL import Image
import numpy as np


data_list = glob('data\\hangul_characters_v1\\*\\*.jpg')
pathlen=len(data_list)
print(pathlen)
