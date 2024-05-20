#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import random


# In[2]:


#Diabetes
data= np.loadtxt(r'Diabetes.txt',delimiter='\t',dtype=str)
data2=data[ : , :-1]
data2= data2.astype(float)
x_train2,x_test2,y_train2,y_test2=train_test_split(data2[ : , :-1],data2[ : ,-1],test_size=0.3,random_state=123)
y_train2=y_train2.reshape((-1,1))
train2=np.hstack((x_train2,y_train2))


# In[3]:


#Ionosphere
daata= np.loadtxt(r'Ionosphere.txt',delimiter='\t',dtype=str)
a=[]
for i in daata:
    a.append(i.split(','))
data4=np.array(a)   
for i in range(len(data4)):
    if data4[i,-1]=='g':
        data4[i,-1] = 1
    else:
        data4[i,-1] = -1
data4= data4.astype(float)
x_train4,x_test4,y_train4,y_test4=train_test_split(data4[ : , :-1],data4[ : ,-1],test_size=0.3,random_state=123)
y_train4=y_train4.reshape((-1,1))
train4=np.hstack((x_train4,y_train4))


# In[4]:


#Sonar
daataa= np.loadtxt(r'Sonar.txt',delimiter='\t',dtype=str)
aa=[]
for i in daataa:
    aa.append(i.split(','))
data5=np.array(aa)   
for i in range(len(data5)):
    if data5[i,-1]=='R':
        data5[i,-1] = 1
    else:
        data5[i,-1] = -1
data5= data5.astype(float)
x_train5,x_test5,y_train5,y_test5=train_test_split(data5[ : , :-1],data5[ : ,-1],test_size=0.3,random_state=123)
y_train5=y_train5.reshape((-1,1))
train5=np.hstack((x_train5,y_train5))


# In[5]:


def classifier(x_train,y_train,x_test,y_test,weights):
    model= DecisionTreeClassifier(max_depth=1,max_features=1)
    model.fit(x_train,y_train,weights)
    h=model.predict(x_train)
    pre=model.predict(x_test)
    return h,pre


# In[6]:


def boost(k,x_train,y_train,x_test,y_test):
    D=1/x_train.shape[0]
    W=np.full((k+1, x_train.shape[0]),D)
    alphas=[]
    pred_list=[]
    for t in range(k):
        error=0
        h,pre=classifier(x_train,y_train,x_test,y_test,W[t])
        h=h.reshape((-1,1))
        pred_list.append(pre)
        #print(pre.shape)
        for i in range(y_train.shape[0]):
            if h[i]!=y_train[i]:
                error+=W[t][i]
        #print(error) 
        if error != 0:
            a=0.5*np.log((1-error)/error)
            alphas.append(a) 
        zt=sum(W[t])    
        for i in range(y_train.shape[0]):
            weight=W[t][i]*np.exp(-1*a*y_train[i]*h[i])
            W[t+1][i]=weight/zt  
    pred = np.zeros(y_test.shape)
    for k in range(x_test.shape[0]):
        H= 0
        for t in range(21):
            H += alphas[t] * pred_list[t][k]
        pred[k] = np.sign(H)  
    acc = accuracy_score(y_test, pred)    
    return acc


# In[7]:


def accuracy(x_train,y_train,x_test,y_test):
    T=[21, 31, 41, 51]
    accs=[]
    for i in T:
        accs.append(boost(i,x_train,y_train,x_test,y_test))
    for i in range(len(T)):
        print('The accuracy of model with ',T[i],'classifiers is',accs[i]*100) 


# In[8]:


def noisy(data,k):
    sample=data[ : , :-1]
    label=data[ : ,-1]
    count=np.round(sample.shape[1]*k)
    features=np.random.randint(0,sample.shape[1]-1,int(count))
    for i in range(int(count)):
        N=np.random.randn(sample.shape[0])
        sample[ : ,features[i]]=N
    label=label.reshape((-1,1))
    #data_train=np.hstack((sample,label))
    return sample,label 


# In[9]:


def acc_noisy(k):
    train_data=[train2,train4,train5]
    datasets=['Diabetes','Ionosphere','Sonar']
    Xtests=[x_test2,x_test4,x_test5]
    Ytests=[y_test2,y_test4,y_test5]
    for i in range(len(train_data)):
        sample,label=noisy(train_data[i],k)
        print(datasets[i],'with',k*100,'% noise')
        accuracy(sample,label,Xtests[i],Ytests[i])    


# In[10]:


print('Diabetes dataset')
accuracy(x_train2,y_train2,x_test2,y_test2)
print('Ionosphere dataset')
accuracy(x_train4,y_train4,x_test4,y_test4)
print('Sonar dataset')
accuracy(x_train5,y_train5,x_test5,y_test5)


# In[11]:


noises=[0.1,0.2,0.3]
for i in noises:
    acc_noisy(i)

