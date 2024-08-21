import os
#import scipy.io.wavfile
#import scipy.signal
import numpy as np
import tensorflow as tf
#from matplotlib import pyplot

from keras.utils import Sequence
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Input, LayerNormalization
from keras.optimizers import SGD, Adam
from keras.metrics import AUC

from keras.callbacks import ModelCheckpoint, EarlyStopping, ProgbarLogger

from keras.applications import ResNet50, VGG16, MobileNet, EfficientNetV2B0
from sklearn.model_selection import train_test_split

from PIL import Image
import xlsxwriter

import requests
import json
import numpy as np
import base64
import sys
import matplotlib.pyplot as plt
import time
import docker

# List of directories to pull images. Positive class is DEFECT_DIRS, negative NOMINAL_DIRS.
# IMPORTANT!!! Defects must be in a directory "defects".
#path=r"C:\Ultra AI Data And Training Set\Ultra AI Training Set\UltraAI Phase 2 Data Set\ultra-training-images-6-inspections"
path=r"C:\Ultra AI Data And Training Set\Ultra AI Training Set\UltraAI Phase 2 Data Set\ultra-training-images-6-inspections - 2.0"

#folder paths containing defect and nominal imgs would be inserted below
DEFECT_DIRS  =  []
NOMINAL_DIRS  =  []

print("Loading file list...")

defect_files_train = []
defect_files_test = []

nominal_files_train = []
nominal_files_test = []

train_files = []
test_files = []

#Below are for evaluating the model per project folder

defect_sep = []
nominal_sep = []

#Below will contain an array of arrays, each array holding
#their respective project's image paths
sep_files = []
conc_sep_files = []

for DEFECT_DIR in DEFECT_DIRS:
    print(DEFECT_DIR)

    defect_files = [os.path.join(DEFECT_DIR, f) for f in os.listdir(DEFECT_DIR)]

    # Sort by map file

    defect_files.sort(key=lambda x: str(os.path.basename(x).split('_')[0]))

    # 90% train
    N = int(len(defect_files)*0.9)

    defect_files_train.extend(defect_files[0:N])
    defect_files_test.extend(defect_files[N:])

    defect_sep.append(defect_files[N:-1])

for NOMINAL_DIR in NOMINAL_DIRS:

    print(NOMINAL_DIR)

    nominal_files = [os.path.join(NOMINAL_DIR, f) for f in os.listdir(NOMINAL_DIR)]

    # Sort by map file
    nominal_files.sort(key=lambda x: str(os.path.basename(x).split('_')[0]))

    # 90% Train
    N = int(len(nominal_files)*0.9)

    nominal_files_train.extend(nominal_files[0:N])
    nominal_files_test.extend(nominal_files[N:])

    nominal_sep.append(nominal_files[N:-1])
print(nominal_files[1])
print(len(defect_files))

# Concatenate files
train_files = defect_files_train + nominal_files_train
test_files = defect_files_test + nominal_files_test

print(len(train_files))

for k in range(0,len(defect_sep)):
    conc_sep_files.append(defect_sep[k] + nominal_sep[k])

print("Shuffling")

print(train_files[-10:])

# Shuffle files in place. Very important!
np.random.shuffle(train_files)
np.random.shuffle(test_files)

print(train_files[-10:])

#Shuffle separated files
for i in range(0,len(conc_sep_files)):
    np.random.shuffle(conc_sep_files[i])

print("# of train files: " + str(len(train_files)))
print("# of test files: " + str(len(test_files)))

print("Done!")



# We use a base model of ResNet50 taking in a 256x128 grayscale image
#base_model = MobileNet(include_top=False, input_shape=(256,128,1), weights=None)
base_model = ResNet50(include_top=False, input_shape=(256,128,1), weights=None)
#base_model = EfficientNetV2B0(include_top=False, input_shape=(256,128,3), weights='imagenet')

# Bottom layer uses GAP and one sigmoid output (binary).
x = base_model.output
x = GlobalAveragePooling2D()(x)
print(x)

predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)


# Optimize on crossentropy using Adam
# Used to be adam, am changing to SGD to use a non-adaptive optimizer. Added the AUC metric to compare
# model metrics per project. AUC is printed in the console when training/evaluated.

#model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['acc'])

init_optimizer = SGD(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=init_optimizer, metrics=[AUC()])
model.summary()


# Add a Gaussian normalization layer to the top of the network
inputs =  Input((256,128,1))
y = LayerNormalization(axis=(1,2,3))(inputs) 

y = model(y)

# Our final model with the normalization added on top
model2 = Model(inputs=inputs, outputs=y)
model2.compile(loss='binary_crossentropy', optimizer=init_optimizer, metrics=[AUC()])
model2.summary()

# Generator to load in images
class DataGenerator(Sequence):

    def __init__(self, otdr_files, batch_size):
        self.otdr_files = otdr_files
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.otdr_files) / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_x = []
        batch_y = []

        for f in self.otdr_files[idx*self.batch_size : (idx+1)*self.batch_size]:
            

            x = (np.array(Image.open(f)).reshape(256,128,1)).astype('float32')


            batch_x.append(x)
            
            # Determine the class from the directory name
            if "defects" in os.path.basename(os.path.dirname(f)): #os.path.basename(os.path.dirname(f)) == DEFECT_DIR:
                batch_y.append(1)
            else:
                batch_y.append(0)

            
        return np.array(batch_x), np.array(batch_y)

training_generator = DataGenerator(train_files, 32)
val_generator = DataGenerator(test_files, 32)

data_generator_sep_list=[]

for m in range(0,len(conc_sep_files)):
    data_generator_sep_list.append(DataGenerator(conc_sep_files[m],32))

# Train model with 1000 steps per epoch (batch size 32).

model2.load_weights(r"C:\Users\BKONG\OneDrive - Xylem Inc\Documents\ultra-ai\model_building\ultra-ai-image-model-2021-12-15.hdf5")
#model2.load_weights(r"C:\Users\BKONG\OneDrive - Xylem Inc\Documents\ultra-ai\attempt 13 weights\weights.32-0.22.hdf5")

# Train model with 10000 steps per epoch (batch size 32).

model2.fit(x = training_generator, epochs=500, validation_data=val_generator, steps_per_epoch=10000, #validation_steps=1600,
                   callbacks=[EarlyStopping(patience=40, restore_best_weights=True),
                               ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5")])

#print(model2.evaluate(x = training_generator,steps=10000,callbacks=[]))

##Predictions Starts
# path=r"C:\models_v3\my_serving_model\0000000123"
# serving_model = tf.keras.models.load_model(path)

# #Seeing predictions for first project
# pred = serving_model.predict(x=data_generator_sep_list[1])
# print(type(pred))
# print(pred)
# print(pred[0][0])
##Prediction Ends



#original way of saving is below, can still use this
tf.saved_model.save(model2, './my_serving_model')

#2nd way of saving model is below, saves keras metadata too (optional, didn't end using this)
#tf.keras.models.save_model(model2, './my_serving_model')

##Below for loop is used to evaluate the accuracy and AUC values for each project separately
for j in range(0,len(conc_sep_files)):
    print("Project # ",j,"Evaluation Below")
    #print(len(conc_sep_files[j]))
    model2.evaluate(x = data_generator_sep_list[j],steps=10000)
    print("*****")

#model2.save('best2.hdf5')


