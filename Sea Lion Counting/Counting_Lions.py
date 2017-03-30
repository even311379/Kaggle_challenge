# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:18:09 2017
Kaggle competition - Sea Lion Counting

@author: user
"""
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from skimage.io import imshow
train_info = pd.read_csv('Train//train.csv')
#print(train_info.head())

#Lion_count = train_info.iloc[:,1:].sum()
#
#Lion_count.plot(kind='bar')
#plt.show()

'''
red: adult males
magenta: subadult males
brown: adult females
blue: juveniles
green: pups
'''
#Define RGB ranges
col_min = {"red": np.array([160, 0, 0]),
           "magenta": np.array([200, 0, 200]),
           "brown": np.array([60, 30, 0]),
           "blue": np.array([0, 40, 150]),
           "green": np.array([0, 140, 0])
           }

col_max = {"red": np.array([255, 50, 50]),
           "magenta": np.array([255, 55, 255]),
           "brown": np.array([130, 70, 40]),
           "blue": np.array([40, 80, 255]),
           "green": np.array([80, 255, 80])
           }

def color_check(p):
    c = p
    for color in ["red","magenta","brown","blue","green"]:
        if sum(p >= col_min[color]) ==3 & sum(p <= col_max[color]) ==3:
            c = color
        else:
            continue
    return c

#ID = 0
#lux,luy = 1055,1363
#size = 10
#
#train_0 = cv2.cvtColor(cv2.imread('Train//'+str(ID)+'.jpg'),cv2.COLOR_BGR2RGB)
#trainD_0 = cv2.cvtColor(cv2.imread('TrainDotted//'+str(ID)+'.jpg'),cv2.COLOR_BGR2RGB)

#train_0 = cv2.imread('Train//'+str(ID)+'.jpg')
#trainD_0 = cv2.imread('TrainDotted//'+str(ID)+'.jpg')


#f, ax = plt.subplots(1,2,figsize=(16,8))
#(ax1, ax2) = ax.flatten()
#
#ax1.imshow(train_0[lux:lux+size,luy:luy+size])
#ax2.imshow(trainD_0[lux:lux+size,luy:luy+size])
#plt.show()

# Flow: Doing sliding window to chop each small image

# Template matching based on the "dot"
#adult_male_template = trainD_0[lux:lux+size,luy:luy+size]



# identify the whether it contains dot or not
# This will be the target image used in training algorithm

# Fit algorithm

ID = 1
lux,luy = 1055,1363
size = 10

train_0 = cv2.cvtColor(cv2.imread('Train//'+str(ID)+'.jpg'),cv2.COLOR_BGR2RGB)
trainD_0 = cv2.cvtColor(cv2.imread('TrainDotted//'+str(ID)+'.jpg'),cv2.COLOR_BGR2RGB)

#img = train_0 [1350:1900, 3000:3400]
#img_dot = trainD_0 [1350:1900, 3000:3400]


img = train_0
img_dot = trainD_0



r1, g1, b1 = 25,25, 25 # Original value
r2, g2, b2 = 230, 230, 230 # Value that we want to replace it with

red, green, blue = img_dot[:,:,0], img_dot[:,:,1], img_dot[:,:,2]
mask = (red < r1) & (green < g1) & (blue < b1)
img_dot[:,:,:3][mask] = [r2, g2, b2]

#img_c = np.copy(img)

diff = cv2.absdiff(img_dot, img)
gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

# Let's change threshold!!
ret,th1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)


#plt.figure(figsize=(16,8))
#plt.imshow(th1, 'gray')

__ ,cnts ,__ = cv2.findContours(th1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

'''
## Fuck the contour finding!! it's always error, the best way to handle should be filtered bad images by whether it contains the dot!!
'''
counts = []

small_lion = []
for c in cnts:
    if len(c) > 3:
        (x,y),rad = cv2.minEnclosingCircle(c)
        try:
            counts.append(color_check(img_dot[int(y),int(x)]))
            img_temp = img_dot[int(y)-20:int(y)+20,int(x)-20:int(x)+20]
            if img_temp.mean()<200:
                small_lion.append(img_temp)
        except:
            pass



    
print("Sea Lions Found: {}".format(len(small_lion)))        
       
#    print(img_dot[int(y),int(x)])
pd.Series(counts).value_counts()


n_images_total = len(small_lion)
n_images_per_row = 5

fig = plt.figure(figsize=(16,36))
for i in range(n_images_total):
    ax = fig.add_subplot((n_images_total/n_images_per_row)+1,n_images_per_row,i+1)
    plt.grid=False
    imshow(small_lion[i])    



#plt.figure(figsize=(16,8))
#plt.imshow(diff)

#f, ax = plt.subplots(3,1,figsize=(18,35))
#plt.grid = False
#(ax1, ax2, ax3) = ax.flatten()
#ax1.imshow(img_dot)
#ax2.imshow(th1, 'gray')
#ax3.imshow(diff)
#plt.show()