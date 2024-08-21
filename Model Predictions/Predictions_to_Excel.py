import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import xlsxwriter

#Predicts the images in the folder, and writes the results to an excel file.
#imgFolderPath is the folder directory, name is the desired name of the excel file,
#last_10 is a boolean indicating if only the last 10% of the folder's images should be 
#predicted. True value means yes, False value means no.
#DefectRange is optional: if not given, predicts all of the images found in the folder
#(or all of the last 10% of images if last_10 is true). If range is given, 
#only predicts the images within that range. Range starts from 0, not 1.
def PredictionWriter(imgFolderPath,name,last_10,DefectRange=None):
    print(int(len(os.listdir(imgFolderPath))*0.9)-1)
    print(int(len(os.listdir(imgFolderPath))-1))
    workbook = xlsxwriter.Workbook(name+'Predictions.xlsx')
    worksheet = workbook.add_worksheet()
    path=r"C:\Users\BKONG\OneDrive - Xylem Inc\Documents\ultra-ai\attempt 13 weights\weights.32-0.22.hdf5"
    #path=r"C:\Users\BKONG\OneDrive - Xylem Inc\Documents\ultra-ai\model_building\ultra-ai-image-model-2021-12-15.hdf5"
    model = load_model(path)
    if DefectRange is None:
        DefectRange=[0,len(os.listdir(imgFolderPath))-1]
    if last_10:
        for index,images in enumerate(os.listdir(imgFolderPath)[int(len(os.listdir(imgFolderPath))*0.9)-1:-1]):
            if (images.endswith(".png")):
                x = (np.array(Image.open(imgFolderPath+'\\'+images)))
                worksheet.write(index,0, images)
                worksheet.write(index,1, model.predict(np.array([x,]))[0][0])
    else:
        for index,images in enumerate(os.listdir(imgFolderPath)[DefectRange[0]:DefectRange[1]+1]):
            if (images.endswith(".png")):
                x = (np.array(Image.open(imgFolderPath+'\\'+images)))
                worksheet.write(index,0, images)
                worksheet.write(index,1, model.predict(np.array([x,]))[0][0])
    workbook.close()

path2=r"C:\Users\BKONG\Downloads\existTesting2\Altered Imgs"
PredictionWriter(path2,"aug13 testing4",True)