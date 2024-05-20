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


#'BreastTissue'
data1= np.loadtxt(r'BreastTissue.txt',delimiter='\t',dtype=str)
data1= data1.astype(float)
x_train1,x_test1,y_train1,y_test1=train_test_split(data1[ : , :-1],data1[ : ,-1],test_size=0.3,random_state=123)
y_train1=y_train1.reshape((-1,1))
train1=np.hstack((x_train1,y_train1))


# In[3]:


#Diabetes
data= np.loadtxt(r'Diabetes.txt',delimiter='\t',dtype=str)
data2=data[ : , :-1]
data2= data2.astype(float)
x_train2,x_test2,y_train2,y_test2=train_test_split(data2[ : , :-1],data2[ : ,-1],test_size=0.3,random_state=123)
y_train2=y_train2.reshape((-1,1))
train2=np.hstack((x_train2,y_train2))


# In[4]:


#'Glass'
dataa= np.loadtxt(r'Glass.txt',delimiter='\t',dtype=str)
data3=dataa[ : , :-1]
data3= data3.astype(float)
x_train3,x_test3,y_train3,y_test3=train_test_split(data3[ : , :-1],data3[ : ,-1],test_size=0.3,random_state=123)
y_train3=y_train3.reshape((-1,1))
train3=np.hstack((x_train3,y_train3))


# In[5]:


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


# In[6]:


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


# In[7]:


#Wine
dat= np.loadtxt(r'Wine.txt',delimiter='\t',dtype=str)
aaa=[]
for i in dat:
    aaa.append(i.split(','))
data6=np.array(aaa)
data6= data6.astype(float)
x_train6,x_test6,y_train6,y_test6=train_test_split(data6[ : , :-1],data6[ : ,-1],test_size=0.3,random_state=123)
y_train6=y_train6.reshape((-1,1))
train6=np.hstack((x_train6,y_train6))


# In[8]:


def classifier(x_train,y_train,x_test,y_test):
    model= DecisionTreeClassifier()
    model.fit(x_train,y_train)
    c=model.predict(x_test)
    return c


# In[9]:


def new_data(data,x_test,y_test):
    w=list(np.arange(0,data.shape[0]))   #768 ta
    q=[]
    dataset=np.zeros((data.shape[0],data.shape[1]))
    for i in range(data.shape[0]):
        q.append(random.choice(w))
    #print(q[-1])    
    for i in range(len(q)):
        dataset[i]=data[q[i]]
    x_train=dataset[ :538, :-1]
    y_train=dataset[ :538,-1]
    pre=classifier(x_train,y_train,x_test,y_test)
    return pre


# In[10]:


def majority(k,data,x_test,y_test):
    d=[]
    majority_list=[]
    total_predict=np.zeros((k,len(y_test)))
    for j in range(k):
        total_predict[j]=new_data(data,x_test,y_test)
    for i in range(len(y_test)):
        (unique, counts) = np.unique(total_predict[ : ,i], return_counts=True)
        f = np.asarray((unique, counts)).T
        d.append(f)
    for i in d:
        majority=np.argmax(i[ : ,1])
        majority_list.append(i[majority,0])
    majority_array=np.array(majority_list)
    acc=accuracy_score(y_test,majority_array)
    return acc     


# In[11]:


def accuracy(data,x_test,y_test):
    classifiers=[11,21,31,41]
    accs=[]
    for i in classifiers:
        accs.append(majority(i,data,x_test,y_test))
    for i in range(len(classifiers)):
        print('The accuracy of model with ',classifiers[i],'classifiers is',accs[i])    


# In[12]:


def noisy(data,k):
    sample=data[ : , :-1]
    #print(sample.shape)
    label=data[ : ,-1]
    count=np.round(sample.shape[1]*k)
    #count=3
    features=np.random.randint(0,sample.shape[1]-1,int(count))
    for i in range(int(count)):
        N=np.random.randn(sample.shape[0])
        sample[ : ,features[i]]=N
    label=label.reshape((-1,1))
    #print(label.shape)
    data_train=np.hstack((sample,label))
    #print(data_train.shape)
    return data_train 


# In[13]:


datasets=['BreastTissue','Diabetes','Glass','Ionosphere','Sonar','Wine']
#datas=[data1,data2,data3,data4,data5,data6]
datas_train=[train1,train2,train3,train4,train5,train6]
Xtests=[x_test1,x_test2,x_test3,x_test4,x_test5,x_test6]
Ytests=[y_test1,y_test2,y_test3,y_test4,y_test5,y_test6]
print('noiseless')
for i in range(len(datasets)):
    print(datasets[i],'dataset')
    accuracy(datas_train[i],Xtests[i],Ytests[i])


# In[14]:


def acc_noisy(k):
    datasets=['BreastTissue','Diabetes','Glass','Ionosphere','Sonar','Wine']
    #datas=[data1,data2,data3,data4,data5,data6]
    datas_train=[train1,train2,train3,train4,train5,train6]
    train_sample=[]
    for i in datas_train:
        train_sample.append(noisy(i,k))
    Xtests=[x_test1,x_test2,x_test3,x_test4,x_test5,x_test6]
    Ytests=[y_test1,y_test2,y_test3,y_test4,y_test5,y_test6]
    print(k*100 ,'% noise')
    for i in range(len(datasets)):
        print(datasets[i],'dataset')
        accuracy(train_sample[i],Xtests[i],Ytests[i])


# In[15]:


acc_noisy(0.1)


# In[16]:


acc_noisy(0.2)


# In[17]:


acc_noisy(0.3)

