# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 22:14:01 2019

@author: tanma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle


number = 5000
folder = "C:\\Users\\tanma.TANMAY-STATION\\Desktop\\ASTRA 19/images"
    
df = pd.read_csv('exoTrain.csv')

labels = df["LABEL"]
df = df.drop('LABEL',axis=1)

from sklearn.preprocessing import MinMaxScaler
df = MinMaxScaler(feature_range = (-1,1)).fit_transform(df.T)
df = df.T
df = pd.DataFrame(df)

fig = plt.figure(figsize=(28,28))
x = np.array(range(3197))
    
for i in range(number):
    plt.scatter(x,df[labels==1].iloc[i,:])
    plt.ylim(-1,1)
    plt.savefig('images/'+str(i))
    plt.close()

image_arr = []
for i in range(number):
    path = os.path.join(folder,str(i)+'.png')
    image = cv2.imread(path,0)
    image = cv2.resize(image,(28,28))
    image_arr.append(image)

image_np = np.zeros(shape = (number,28,28,1))
for i in range(len(image_arr)):
    image_np[i] = np.reshape(image_arr[i],(28,28,1))
        
with open ("image_arr.pkl","wb") as file:
    image_np = pickle.dump(image_np,file)
