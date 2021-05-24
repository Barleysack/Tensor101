import sys, os
from typing import Collection
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import pandas

#유니코드 1100-11FF가 한글 자모.(256)
#호환용 한글자모는 3130-318F(96)
#한글 자모 확장 A960-A97F 32
#한글 자모 확장 D7B0-D7FF 80

font_path="C:/Users/Finally/Desktop/Works/Workspace/TTF_to_Png/Fonts(ttf)/"
fonts = os.listdir(font_path)

#유니코드 조합
uc = "0 1 2 3 4 5 6 7 8 9 A B C D E F"
start = "1100"
end = "11FF"

uc = uc.split(" ")
#각 digit을 포문을 돌려보자
collections = [a+b+c+d 
                    for a in uc 
                    for b in uc 
                    for c in uc 
                    for d in uc]


collections = np.array(collections)

s = np.where(start == collections)[0][0]
e = np.where(end == collections)[0][0]

collections = collections[s:e+1]

print(collections)
unicodeChars = chr(int(collections[0], 16))
for uni in tqdm(collections):
    
    unicodeChars = chr(int(uni, 16))
    
    path = "./Hangul_dataset/" + unicodeChars
    
    os.makedirs(path, exist_ok = True)
        
    for ttf in fonts:
        
        font = ImageFont.truetype(font = font_path + ttf, size = 100)
        
        x, y = font.getsize(unicodeChars)
        
        theImage = Image.new('RGB', (x + 3, y + 3), color='white')
        
        theDrawPad = ImageDraw.Draw(theImage)
        
        theDrawPad.text((0.0, 0.0), unicodeChars[0], font=font, fill='black' )
        
        msg = path + "/" + ttf[:-4] + "_" + unicodeChars
        
        theImage.save('{}.png'.format(msg))

