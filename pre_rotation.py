import glob
import os

import cv2
import numpy as np
out_dir = "output_18"
#path = glob.glob("/Users/louis/Desktop/dataset/*.jpg")
#img = cv2.imread("/Users/louis/Desktop/dataset/2015-11-22_1144.jpg")
#path = "/Users/louis/Desktop/dataset/2015-11-22_1144.jpg"
os.makedirs(out_dir,exist_ok=True)
for path in glob.glob("/Users/louis/Desktop/dataset/*.jpg"):
    img = cv2.imread(str(path))
    #高さを定義
    height = img.shape[0]                         
    #幅を定義
    width = img.shape[1]  
    #回転の中心を指定                          
    center = (int(width/2), int(height/2))

    #回転角を指定
    angle = 18.0
    #スケールを指定
    scale = 1.0
    #getRotationMatrix2D関数を使用
    trans = cv2.getRotationMatrix2D(center, angle , scale)
    #アフィン変換
    image2 = cv2.warpAffine(img, trans, (width,height))
    
    basename = os.path.basename(path)
    out_path = os.path.join(out_dir, basename)
    # 保存
    cv2.imwrite(out_path, image2)
