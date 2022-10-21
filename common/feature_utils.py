#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import pylab
import torch
data_dist=torch.load('1_50_raw.mat')
cur_path=os.getcwd()
data_path=os.path.join(cur_path,'data')
dir_list=os.listdir(data_path)
data_type=['Acceleration','Emg','Gyroscope']
data_path
data_dist['data'][0].shape


# In[4]:


import numpy as np 

def featureRMS(data):# 平方和，一维cos
    return np.sqrt(np.mean(data**2, axis=0))

def featureMAV(data):# 时间窗口的大小，滑动距离，网格搜索，强化、对抗学习调参，SENet自己架构，多尺度卷积，channel 空间 time 残差dense，Transformer，结合，RNN，LSTM卷积
    return np.mean(np.abs(data), axis=0) 

def featureWL(data):
    return np.sum(np.abs(np.diff(data, axis=0)),axis=0)/data.shape[0]

def featureZC(data, threshold=10e-7):
    numOfZC = []
    channel = data.shape[1]
    length  = data.shape[0]
    
    for i in range(channel):
        count = 0
        for j in range(1,length):
            diff = data[j,i] - data[j-1,i]
            mult = data[j,i] * data[j-1,i]
            
            if np.abs(diff)>threshold and mult<0:
                count=count+1
        numOfZC.append(count/length)
    return np.array(numOfZC)

def featureSSC(data,threshold=10e-7):
    numOfSSC = []
    channel = data.shape[1]
    length  = data.shape[0]
    
    for i in range(channel):
        count = 0
        for j in range(2,length):
            diff1 = data[j,i]-data[j-1,i]
            diff2 = data[j-1,i]-data[j-2,i]
            sign  = diff1 * diff2
            
            if sign>0:
                if(np.abs(diff1)>threshold or np.abs(diff2)>threshold):
                    count=count+1
        numOfSSC.append(count/length)
    return np.array(numOfSSC)




import math

timeWindow = 30
strideWindow = 20

def feature_extra_emg(emg):
    '''多电极肌电信号提取'''
    global featureData
    emgfeatureData=[]
    emg=emg.T
    length = math.floor((emg.shape[0]-timeWindow)/strideWindow)
    for i in range(emg.shape[1]):
        iemg=emg[:,i]
        iemg=iemg.reshape(emg.shape[0],1)
        for j in range(length):
            rms = featureRMS(iemg[strideWindow*j:strideWindow*j+timeWindow,:])
            mav = featureMAV(iemg[strideWindow*j:strideWindow*j+timeWindow,:])
            wl  = featureWL( iemg[strideWindow*j:strideWindow*j+timeWindow,:])
            zc  = featureZC( iemg[strideWindow*j:strideWindow*j+timeWindow,:])
            ssc = featureSSC(iemg[strideWindow*j:strideWindow*j+timeWindow,:])
            featureStack = np.hstack((rms,mav,wl,zc,ssc))
            emgfeatureData.append(featureStack)
    return np.array(emgfeatureData)


# In[6]:


import matplotlib.pyplot as plt
def feature_extra_gyr(gyr):
    '''多极陀螺仪信号提取'''
    poly_list=[]
    x=np.arange(0,timeWindow)
    gyr=gyr.T
    length = math.floor((gyr.shape[0]-timeWindow)/timeWindow)
    for i in range(gyr.shape[1]):
        single_poly_list=[]
        igyr=gyr[:,i]
        igyr=igyr.reshape(igyr.shape[0],1)
        for j in range(length):
            func=np.poly1d(np.polyfit(x,igyr[timeWindow*j:timeWindow*j+timeWindow][:,0],deg=3))
            funcx=func(x)
            #funcx=funcx[np.arange(0,timeWindow,10)]
            single_poly_list.extend(funcx)
            func=None
        indice_to_48=np.linspace(0,119,48,dtype=int)
        single_poly_list_compress=np.array(single_poly_list)[indice_to_48]
        poly_list.append(single_poly_list_compress)
    return np.array(poly_list).T


def feature_extra_acc(acc):
    '''多极加速度信号提取'''
    poly_list=[]
    x=np.arange(0,timeWindow)
    acc=acc.T
    length = math.floor((acc.shape[0]-timeWindow)/timeWindow)
    for i in range(acc.shape[1]):
        single_poly_list=[]
        iacc=acc[:,i]
        #plt.plot(iacc)
        #plt.show()
        iacc=iacc.reshape(iacc.shape[0],1)
        for j in range(length):
            func=np.poly1d(np.polyfit(x,iacc[timeWindow*j:timeWindow*j+timeWindow][:,0],deg=3))
            funcx=func(x)
            single_poly_list.extend(funcx)
            func=None
        indice_to_48=np.linspace(0, 119, 48, dtype=int)
        single_poly_list_compress=np.array(single_poly_list)[indice_to_48]
        poly_list.append(single_poly_list_compress)
    return np.array(poly_list).T



feature_data_dist={}
featureDataTotal=[]
featureLabelTotal=[]
print(len(data_dist['data']))
for num in range(len(data_dist['data'])):
    emg_feature=feature_extra_emg(np.array(data_dist['data'][num])[3:11,])
    acc_feature=feature_extra_acc(np.array(data_dist['data'][num])[0:3,])
    gyr_feature=feature_extra_gyr(np.array(data_dist['data'][num])[11:14,])
    label=data_dist['target'][num]
    featureDataTotal.append(np.hstack((emg_feature,gyr_feature,acc_feature)))
    featureLabelTotal.append(label)
feature_data_dist={}
feature_data_dist['data']=featureDataTotal
feature_data_dist['target']=featureLabelTotal


# In[9]:


feature_data_dist['data'][0].shape






