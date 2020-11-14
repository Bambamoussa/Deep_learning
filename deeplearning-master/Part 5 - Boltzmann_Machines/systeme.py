#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:27:58 2020

@author: bamba
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
"""
movies=pd.read_csv("ml-1m/movies.dat",sep="::",header=None,engine="python",encoding="latin-1")


users=pd.read_csv("ml-1m/users.dat",sep="::",header=None,engine="python",encoding="latin-1")

ratings=pd.read_csv("ml-1m/ratings.dat",sep="::",header=None,engine="python",encoding="latin-1")

training_set=pd.read_csv("ml-100k/u1.base",delimiter="\t",header=None)

training_set=np.array(training_set,dtype="int")


test_set=pd.read_csv("ml-100k/u1.test",delimiter="\t",header=None)

test_set=np.array(test_set,dtype="int")

nb_users=int(max(max(training_set[:,0]),max(test_set[:,0])))

nb_movies=int(max(max(training_set[:,1]),max(test_set[:,1])))

def convert(data):
    new_data=[]
    for id_users in range(1,nb_users+1):
        id_movies=data[data[:,0]==id_users,1]
        
        id_ratings=data[data[:,0]==id_users,2]
        
        ratings=np.zeros(nb_movies)
        ratings[id_movies-1]=id_ratings
        new_data.append(list(ratings))
        
    return new_data

training_set=convert(training_set)  

training_set=torch.FloatTensor(training_set)
test_set=torch.FloatTensor(test_set)

test_set[test_set==0]=-1
test_set[test_set==1]=0
test_set[test_set==2]=0
test_set[test_set>=3]=1


class RBM():
    def __init__(self,nv,nh):
        self.W=torch.randn(nv,nh)
        self.a=torch.randn(1,nh)
        self.b=torch.randn(1,nv)
        
    def sample_h(self,x):
        wx=torch.mm(x,self.W.t())
        activation=wx +self.a.expand_as(wx)
        p_h_given_v=torch.sigmoid(activation)
        return p_h_givenv,torch.bernoulli(p_h_given_v)
    
    def sample_v(self,y):
        wy=torch.mm(y,self.W)
        activation=wy +self.a.expand_as(wy)
        p_h_given_h=torch.sigmoid(activation)
        return p_h_given_h,torch.bernoulli(p_h_given_h)
    
    def train(self,v0,vk,ph0,phk):
        self.W += torch.mm(v0.t(),ph0)-torch.mm(vk.t(),phk)
        self.b +=torch.sum((v0-vk),0)
        self.a +=torch.sum((ph0-phk),0)

nv=len(training_set[0])
nh=100
batch_size=100
rbm=RBM(nv,nh)

nb_epoch=10
for epoch in range(1,nb_epoch+1):
    train_loss=0
    s=0.
    for id_user in range(0,nb_users -batch_size,batch_size):
        v0=training_set[id_user:id_user+batch_size]
        vk=v0
        ph0,_=rbm.sample_h(v0)
        for k in range(10):
            _,hk=rbm.sample_h(vk)
            _,vk=rbm.sample_v(hk)
            
            vk[v0<0]=v0[v0<0]
        phk,_=rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk)
        train_loss+=torch.mean(torch.abs(vk[v0>=0]-vk[v0>=0]))
        s+=1.
    print("epoch:",str(epoch)+ "loss:",+str(train_loss/s))
    
    """