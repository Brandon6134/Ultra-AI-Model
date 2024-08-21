import numpy as np
from PIL import Image, ImageEnhance
import cv2
import os

paths=[r"C:\Users\BKONG\Downloads\existTesting",
    r"C:\Users\BKONG\Downloads\existTesting2"]
for path in paths:
    alterPath = os.path.join(path,'Altered Imgs')
    try:
        os.mkdir(alterPath)
        print('made new directory') 
    except:
        print('directory already exists')
    width,height=128,256

    #pvShift is positive vertical shift upwards, and nvShift is negative vertical shift downwards (values are in pixels)
    pvShift,nvShift=5,20

    #factor is contrast value
    factor = np.array([2,1.5,0.5])

    #crop are the height and width dimensions to take of the img
    crop=np.array([[0,256,0,128],[pvShift,height,0,128],
                [0,height-nvShift,0,128]])
    string=["_2.0c","_v_+10_1.5c","_v_-10_0.5c"]

    #loop through each img in the directory folder and alter them
    for images in os.listdir(path):
        for k in range(0,len(crop)):
            if (images.endswith(".png")):
                im = cv2.imread(path+'/'+images,-1)

                #apply contrast change
                for i in range (im.shape[0]): #traverses through height of the image
                    for j in range (im.shape[1]): #traverses through width of the image
                        #if statements check if the contrast applied would make the value equal below 0 or over 65535 
                        #if they are, make them equal to 0 or 65535; if not then apply contrast equation
                        if (im[i][j]-32768) * factor[k] + 32768 < 0:
                            im[i][j] = 0
                        elif (im[i][j]-32768) * factor[k] + 32768 > 65535:
                            im[i][j] = 65535
                        else:
                            im[i][j] = (im[i][j]-32768) * factor[k] + 32768
                
                #im = cv2.resize(im, resize[k])
                im = im[crop[k][0]:crop[k][1],crop[k][2]:crop[k][3]]
                im = cv2.resize(im,(128,256))
                os.chdir(alterPath)
                cv2.imwrite(images[0:-4]+string[k]+'.png', im) 