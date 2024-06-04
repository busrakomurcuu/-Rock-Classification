# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:45:14 2024

@author: BUSRA
"""

from tensorflow.keras.preprocessing import image 

import numpy as np 

import sys 

import os 

import matplotlib.pyplot as plt 

  

from tensorflow.keras.applications.vgg16 import VGG16 

from tensorflow.keras.applications.resnet50 import ResNet50 

from tensorflow.keras.applications.xception import Xception 

from tensorflow.keras.applications.inception_v3 import InceptionV3 

from tensorflow.keras.applications.mobilenet import MobileNet 

  

from tensorflow.keras.models import Sequential  

from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, GRU 

from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense 

from tensorflow.keras.layers import TimeDistributed,Bidirectional,GlobalAveragePooling2D 

from tensorflow.keras.layers import LSTM,GRU,SimpleRNN,BatchNormalization 

from tensorflow.keras import backend as K  

from tensorflow.keras.callbacks import ModelCheckpoint 

from tensorflow.keras.utils import to_categorical 

from sklearn.metrics import confusion_matrix 

from sklearn.preprocessing import LabelEncoder 

from sklearn.model_selection import train_test_split as sp 

from sklearn.model_selection import StratifiedKFold 

from sklearn.metrics import roc_auc_score, accuracy_score 

img_height = 100 

img_width = 100 

  

def  get_images(path):  

   image_list = [] 

   class_list = []     

   for dirname in os.listdir(path): 

    

        new_path =os.path.join(path,dirname) 

        for dirname, dirnames1,filenames in os.walk(new_path): 

            for filename in filenames: 

                img = image_to_vector(os.path.join(new_path,filename)) 

                image_list.append(img) 

                class_list.append(dirname) 

   return np.array(image_list),class_list 

  

def image_to_vector(img_file_path): 

  

    img = image.load_img(img_file_path, target_size=(img_height, img_width)) 

    x = image.img_to_array(img) 

  

    return x 

  

def create_checkpoint(): 

    filepath = "best.hdf5" 

    checkpoint = ModelCheckpoint(filepath, 

                            monitor='val_loss', 

                            verbose=2,  

                            save_best_only=True, 

                            mode='min')  

    return checkpoint 

  

def create_cnn_model1(): 

    model = Sequential()  

     

    model.add(Conv2D(32, (3, 3), input_shape = (img_height,img_width,3),kernel_initializer='VarianceScaling'))  

    model.add(Conv2D(32, (3, 3),kernel_initializer='VarianceScaling' ))   

    model.add(BatchNormalization()) 

    model.add(Activation('relu')) #gizli katman  

    model.add(MaxPooling2D(pool_size =(2, 2)))  

     

    model.add(Conv2D(64, (3, 3),kernel_initializer='VarianceScaling' )) 

    model.add(Conv2D(64, (3, 3),kernel_initializer='VarianceScaling' ))  

    model.add(BatchNormalization()) 

    model.add(Activation('relu'))  

    model.add(MaxPooling2D(pool_size =(2, 2)))  

     

    model.add(Conv2D(128, (3, 3),kernel_initializer='VarianceScaling' ))  

    model.add(Conv2D(128, (3, 3),kernel_initializer='VarianceScaling' ))  

    model.add(BatchNormalization()) 

    model.add(Activation('relu'))  

    model.add(MaxPooling2D(pool_size =(2, 2)))  

    

    model.add(GlobalAveragePooling2D()) 

    model.add(Dropout(0.5))  

     

  

    model.add(Flatten())  

  

    model.add(Dense(units=100, activation='sigmoid'))  

  

  

    model.add(Dense(2))  

    model.add(Activation('softmax'))  

    model.compile(loss ='categorical_crossentropy',  

                     optimizer ='adam',  

                   metrics =['accuracy'])  

    return model 

  

filePath = r'C:\Users\BÜŞRA\.spyder-py3\kurs\veriseti'   

  

  

features,classes = get_images(filePath) 

  

features = features / 255  

  

  

encoder = LabelEncoder()  

classes = encoder.fit_transform(classes) 

  

  

rnd_seed = 116 

np.random.seed(117)  

kf =StratifiedKFold(n_splits=10, random_state=rnd_seed, shuffle =True)  

  

  

  

scores = [] 

scores_auc = [] 

confussionMatrix = [] 

total = np.zeros((2,2)) 

filepath = "best.hdf5" 

  

  

for train_index, test_index in kf.split(features,classes):  

    classes1 = to_categorical(classes, num_classes=2) 

    x_train_val =features[train_index] 

    x_test = features[test_index] 

    y_train_val = classes1[train_index] 

    y_test = classes1[test_index] 

     

    x_train,x_val,y_train,y_val = sp(x_train_val, y_train_val,random_state=5, test_size=.1) 

     

  

    model = create_cnn_model1() 

   

    checkpoint = create_checkpoint() 

    print('Train... Model') 

    model.fit(x_train , y_train, epochs=50, batch_size=10, validation_data=(x_val,y_val),verbose=2,callbacks=[checkpoint]) 

    model.load_weights("best.hdf5") 

    y_pred = model.predict(x_test) 

     

  

    conf_mat = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)) 

  

     

    scores.append(accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))) 

    total = total + conf_mat[:2, :2] 

    confussionMatrix.append(conf_mat) 

     

  

import seaborn as sns 

  

  

average_conf_matrix = total / 10   

  

  

plt.figure(figsize=(8, 6)) 

sns.heatmap(conf_mat, annot=True, fmt=".0f", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"]) 

plt.title('Average Confusion Matrix') 

plt.xlabel('Predicted Labels') 

plt.ylabel('True Labels') 

plt.show() 

  

  

import numpy as np 

import matplotlib.pyplot as plt 

  

def relu(x): 

    return np.maximum(0, x) 

  

  

x_values = np.linspace(-5, 5, 100) 

  

  

y_values = relu(x_values) 

  

  

 

 

plt.figure(figsize=(8, 6)) 

plt.plot(x_values, y_values, label='ReLU Function', color='blue') 

plt.title('ReLU Activation Function') 

plt.xlabel('Input') 

plt.ylabel('Output') 

plt.axhline(0, color='black',linewidth=0.5) 

plt.axvline(0, color='black',linewidth=0.5) 

plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5) 

plt.legend() 

plt.show() 

  

  

print('Scores from each Iteration: ', scores) 

print('Average K-Fold Score :' , np.mean(scores)) 

  

print(conf_mat) 