import numpy as np
from PIL import Image, ImageEnhance
import cv2
import os

#paths contains the folder paths of the projects desired to make altered images of
paths=[r"C:\Users\BKONG\Downloads\picTesting",
       r"C:\Users\BKONG\Downloads\picTesting2"]

#places the altered images in a new subfolder called 'altered imgs'
for path in paths:
    alterPath = os.path.join(path,'Altered Imgs')
    try:
        os.mkdir(alterPath)
        print('made new directory') 
    except:
        print('directory already exists')

    width,height=512,300
    #base=[int((height-256)/2),int((height-256)/2 +256),int((width-128)/2),int((width-128)/2 +128)]

    #maximum horizontal pixel shift
    maxhShift=30

    #first value for maxvShift is max vertical shift pixels upwards, second is max vertical shift pixels downwards
    maxvShift=[5,5]

    #Formula for total number of altered images created: (num_hshifts+1)*2 + (num_vShifts-2)*2
    #e.g. (10+1)*2 + (3-1)*2 = 24 created (numbers used for France and Voelstalpine)
    num_hShifts=10
    num_vShifts=3


    #crop starting from very top of img, downwards
    base=[maxvShift[0],256+maxvShift[0],int((width-128)/2),int((width-128)/2 +128)]

    #below lines make arrays of x and y coordinates to crop the image
    hShift=np.linspace(-maxhShift,maxhShift,num=num_hShifts+1)
    hShift=np.append(hShift,np.ones(num_vShifts-2)*maxhShift)
    hShift=np.append(hShift,np.linspace(maxhShift,-maxhShift,num=num_hShifts+1))
    hShift=np.append(hShift,np.ones(num_vShifts-2)*-maxhShift)
    hShift=hShift.astype(int)

    vShift=np.ones(num_hShifts+1)*-maxvShift[0]
    vShift=np.append(vShift,np.linspace(-maxvShift[0],maxvShift[1],num=num_vShifts)[1:-1])
    vShift=np.append(vShift,np.ones(num_hShifts+1)*maxvShift[1])
    vShift=np.append(vShift,np.linspace(maxvShift[1],maxvShift[0],num=num_vShifts)[1:-1])
    vShift=vShift.astype(int)

    factor=np.linspace(0.5,2,num=num_hShifts+1)
    factor=np.append(factor,np.linspace(0.5,2,num=num_vShifts)[1:-1])
    factor=np.append(factor,np.linspace(0.5,2,num=num_hShifts+1))
    factor=np.append(factor,np.linspace(0.5,2,num=num_vShifts)[1:-1])
    for i in range(0,len(factor)):
        factor[i]=round(factor[i],2)
        
    #loop through each img in the directory folder and alter them
    for images in os.listdir(path):
        for k in range(0,len(hShift)):
            if (images.endswith(".png")):
                im = cv2.imread(path+'/'+images,-1)
                im = im[base[0]+vShift[k]:base[1]+vShift[k],
                        base[2]+hShift[k]:base[3]+hShift[k]]

                for i in range (im.shape[0]): #traverses through height of the image
                    for j in range (im.shape[1]): #traverses through width of the image
                        if (im[i][j]-32768) * factor[k] + 32768 < 0:
                            im[i][j] = 0
                        elif (im[i][j]-32768) * factor[k] + 32768 > 65535:
                            im[i][j] = 65535
                        else:
                            im[i][j] = (im[i][j]-32768) * factor[k] + 32768
                
                os.chdir(alterPath)
                string='_'+str(k)+'_h,v_Shift_'+str(hShift[k])+','+str(-vShift[k])+'_cont_'+str(factor[k])
                cv2.imwrite(images[0:-4]+string+'.png', im)


    for images in os.listdir(path):
        if (images.endswith(".png")):
            im = cv2.imread(path+'/'+images,-1)
            im = im[base[0]:base[1],
                    base[2]:base[3]]
            os.chdir(alterPath)
            string='_'+'base'
            cv2.imwrite(images[0:-4]+string+'.png', im)

