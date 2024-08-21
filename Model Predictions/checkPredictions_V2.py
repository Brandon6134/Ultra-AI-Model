import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

# Predictions takes a folder path of pngs, and outputs an array of the predictions of both the old and new models.
# Uses 1 or 2 img folder paths. If two are given, assumes imgFolderPath2 is path of new model's img folder.
def Predictions(imgFolderPath,imgFolderPath2=None):
    path_new=r"C:\Users\BKONG\OneDrive - Xylem Inc\Documents\ultra-ai\attempt 13 weights\weights.32-0.22.hdf5"
    path_old=r"C:\Users\BKONG\OneDrive - Xylem Inc\Documents\ultra-ai\model_building\ultra-ai-image-model-2021-12-15.hdf5"
    model_new = load_model(path_new)
    model_old= load_model(path_old)
    predict_new = []
    predict_old = []
    if imgFolderPath2==None:
        for images in os.listdir(imgFolderPath)[int(len(os.listdir(imgFolderPath))*0.9)-1:-1]:
            if (images.endswith(".png")):
                x=np.array(Image.open(imgFolderPath+'\\'+images))
                predict_old.append(model_old.predict(np.array([x,]))[0][0])
                predict_new.append(model_new.predict(np.array([x,]))[0][0])
    else:
        print(int(len(os.listdir(imgFolderPath))*0.9)-1)
        print(int(len(os.listdir(imgFolderPath2))*0.9)-1)
        for images in os.listdir(imgFolderPath)[int(len(os.listdir(imgFolderPath))*0.9)-1:-1]:
            if (images.endswith(".png")):
                x=np.array(Image.open(imgFolderPath+'\\'+images))
                predict_old.append(model_old.predict(np.array([x,]))[0][0])
        for images in os.listdir(imgFolderPath2)[int(len(os.listdir(imgFolderPath2))*0.9)-1:-1]:
            if (images.endswith(".png")):
                x=np.array(Image.open(imgFolderPath2+'\\'+images))
                predict_new.append(model_new.predict(np.array([x,]))[0][0])
    return predict_old,predict_new

# Thresholds takes an array and evalutes the percentage of the array above different threshold values
# (0.5 through 0.95, increasing with increments of 0.05). If type is equal to 0, does comparision for defect.
# For any other value, does comparison for nominal. Function returns an array of these percentage values.
def Thresholds(array,type):
    thresh = np.arange(0.5,1,0.05)
    percents=np.zeros(len(thresh))
    for index,k in enumerate(thresh):
        if type==0:
            percents[index]=sum(array>k)/len(array)
        else:
            percents[index]=sum(array<k)/len(array)
    return percents

# PlotDefect takes two arrays, first being the old model's defect accuracy percentage array and the second being the new model's.
# It plots the these value in a line graph against a set x axis and saves their graphs as pngs. The function takes a title for naming
# each graph and their png, if type is 0 then plots names include 'defect' otherwise plot names include 'nominal', decimals indicates the 
# number of decimals wanted in the data labels percentages.
def Plot(y_axis_old,y_axis_new,title,type,decimals):
    x_axis = np.arange(0.5,1,0.05)
    print(x_axis)
    print(y_axis_old)
    print(y_axis_new)
    plt.plot(x_axis,y_axis_old,marker='o')
    plt.plot(x_axis,y_axis_new,marker='o')
    plt.xlabel('Prediction Threshold Values')
    plt.xticks(x_axis)
    plt.legend(["Current Model","New Model"])
    for i in range(len(x_axis)):
        plt.annotate(str(round(y_axis_old[i],decimals))+'%',(x_axis[i],y_axis_old[i]))
        plt.annotate(str(round(y_axis_new[i],decimals))+'%',(x_axis[i],y_axis_new[i]))
    if type==0:
        plt.ylabel('Percentage of Defects Accuractely Predicted')
        plt.title('AI Model Defect Prediction Accuracy Stats for '+title)
        plt.savefig('plot_imgs/'+title+' Defects.png')
    else:
        plt.ylabel('Percentage of Nominals Accuractely Predicted')
        plt.title('AI Model Nominal Prediction Accuracy Stats for '+title)
        plt.savefig('plot_imgs/'+title+' Nominals.png')
    plt.clf()

#PlotNominal takes two arrays, first being the old model's nominal accuracy percentage and second being the new models.
# It plots these as a bar graph against each other. Function also takes a title for naming each graph and their png.
# def PlotNominalBarGraph(y_axis_old,y_axis_new,title):
#     fig,ax=plt.subplots()
#     ax.bar(['Current Model','New Model'],[y_axis_old,y_axis_new])
#     ax.set_ylabel('Percentage of Nominals Predicted Under 0.5')
#     print(y_axis_old)
#     print(y_axis_new)
#     plt.title('AI Model Nominals Prediction Accuracy Stats for '+title)

#     rects = ax.patches
#     labels=[str(round(y_axis_old,2))+'%',str(round(y_axis_new,2))+'%']
#     for rect,label in zip(rects,labels):
#         height=rect.get_height()
#         ax.text(
#             rect.get_x() + rect.get_width() / 2, height + 0.5, label, ha="center", va="bottom"
#         )
#     plt.savefig('plot_imgs/'+title+' Nominals.png')
#     plt.clf()

path1=r"C:\Ultra AI Data And Training Set\Ultra AI Training Set\UltraAI Phase 2 Data Set\ultra-training-images-6-inspections"
path2=r"C:\Ultra AI Data And Training Set\Ultra AI Training Set\UltraAI Phase 2 Data Set\ultra-training-images-6-inspections - 2.0"

DEFECT_DIRS_1  =  []
DEFECT_DIRS_2  =  []
NOMINAL_DIRS  =  []

old_def_percents=[]
new_def_percents=[]
old_nom_percents=[]
new_nom_percents=[]

#Call for Predictions, Thresholds, then Plot functions
for i in range(len(NOMINAL_DIRS)):
    old,new=(Predictions(DEFECT_DIRS_2[i]))
    old_def_percents=Thresholds(old,0)
    new_def_percents=Thresholds(new,0)
    Plot(old_def_percents*100,new_def_percents*100,DEFECT_DIRS_2[i].split('\\')[-2],0,0)
    
    old,new=(Predictions(NOMINAL_DIRS[i]))
    new_def_percents=Thresholds(new,1)
    old_def_percents=Thresholds(old,1)
    Plot(old_def_percents*100,new_def_percents*100,NOMINAL_DIRS[i].split('\\')[-2],1,1)


