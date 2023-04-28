# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 17:38:37 2022

@author: 97091
"""
import torch
# from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from torch_geometric.nn import GraphConv
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.nn import ReLU
from torch.nn import Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,SAGEConv,GATConv,GNNExplainer
from torch_geometric.nn import global_mean_pool
from sklearn.model_selection import KFold
import math


datas = pd.read_csv('edge.csv').values
index=[i for i in range(0,datas.shape[0]+1,2)]
edges=[]
for i in range(0,len(index)-1):
    temp=datas[index[i]:index[i+1]]
    edges.append(temp)
edge=[]    
for k in range(0,len(edges)):
    a=[]
    for e in edges[k]:
        x=[item for item in e if math.isnan(item)==False]
        a.append(x)
    edge.append(a)


def process_csv(path,path1,path2,edges,path3):
    
    
    
    
     datas = pd.read_csv(path).values
     
     indi= pd.read_csv(path1).values
     
     label=pd.read_csv(path2).values
     
     weight=pd.read_csv(path3).values
     

     


     a={i:list(np.where(indi==i)[0]) for i in np.unique(indi)}
     BATCH=[]
     for k in a.keys():
      
      x=torch.tensor(datas[a[k],:],dtype=torch.float)
      
      
      
      
      edge_index=torch.tensor(edges[k],dtype=torch.long)
      l=torch.tensor(label[k],dtype=torch.long)
      
      we=torch.tensor(weight[k],dtype=torch.float)
     
      
      data=Data(x=x,y=l,edge_index=edge_index,edge_weight=we)
      BATCH.append(data)
    
     return BATCH
dataset=process_csv('graph.csv','index.csv','score.csv',edge,'weight.csv')

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 =GraphConv(2, hidden_channels) 
        self.conv2 =GraphConv(hidden_channels, hidden_channels)  
        self.conv3 =GraphConv(hidden_channels, 3) 
        self.importance=Linear(hidden_channels,3)
       
        
        
        
        
       
        

    def forward(self, x, edge_index,edge_weight, batch):
        x = self.conv1(x=x, edge_index=edge_index,edge_weight=edge_weight)
        x = x.relu()
        x = self.conv2(x=x, edge_index=edge_index,edge_weight=edge_weight)
        x = x.relu()
        x = self.conv3(x=x, edge_index=edge_index,edge_weight=edge_weight)
        # i=torch.unique(batch)

        
        # index=[np.where(batch.numpy()==item.numpy())[0][-1] for item in i]
        # x=x[index]
        # importance=self.importance(x)
        # print(x,batch)

       
        
        
        
        
            
           
        # im=torch.nn.functional.softmax(importance)
        # im=torch.nn.functional.normalize(importance, p=2.0, dim=1, eps=1e-12, out=None)
        
        
        
        
       
        
        
        
        
        
        
        return x

model = GNN(hidden_channels=128)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

criterion1=torch.nn.CrossEntropyLoss()    


def test(loader):
    model.eval()
    correct=0
   
    loader=DataLoader(loader, batch_size=6)
   
    for data in loader:
       
        out=model(data.x,data.edge_index,data.edge_weight,data.batch)
        
        pred1=out.argmax(dim=1)
       
        
        correct+=int((pred1==data.y).sum())
   
  
    return correct/len(loader.dataset)
def train_model(datasets):
    model.train()
    datas=DataLoader(datasets, batch_size=1, shuffle=True)
    for _ in range(200):
      for data in datas:
        out=model(data.x,data.edge_index,data.edge_weight,data.batch)
        
        loss1=criterion1(out,data.y)
       
        
        loss1.backward()
     
        optimizer.step()
        optimizer.zero_grad()
def inverse_weight(distance):
    distance=[1/item for item in distance]
    weight=[round(item/sum(distance),2) for item in distance]
    return  weight
    
train_model(dataset)
acc=test(dataset)
x=torch.tensor([
                [0,1],
                [0,2],
                [0,3],
                [0,1],
                [0,2],
                [0,3],
                [0,1],
                [1,0]],dtype=torch.float)
inverse=[0.0707, 0.0762, 0.0922, 0.1118]
edge_index=torch.tensor([[0,1,2,3,4,5,6],[7,7,7,7,7,7,7]],dtype=torch.long)
edge_weight=torch.tensor([0.13,0.14,0.13,0.12,0.11,0.12,0.13],dtype=torch.float)
batch=[0]*x.size()[0]
batch=torch.tensor(batch)
label=model(x,edge_index,edge_weight,batch)
print(label[-1])
