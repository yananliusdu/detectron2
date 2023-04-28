# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:48:40 2023

@author: 97091

GNN training
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

# data path
path='/media/yanan/One Touch/detectron2_data_Ma/detectron2_data/crops'

#model path
model_path='/media/yanan/One Touch/detectron2_data_Ma/detectron2_data'

def create_link(num1,num2):
   
    ed=[i for i in range(num1) if i!=num2]
    ed1=[num2]*(num1-1)
    return [ed,ed1]


def inverse_weight(distance):
    distance=[1/item for item in distance]
    weight=[round(item/sum(distance),2) for item in distance]
    return  weight

def creat_weight(edge,center):
    ori=center[edge[1][0]]
    dist =[ np.linalg.norm(np.array(ori)-np.array(center[d])) for d in edge[0]]
    inverse_weight(dist)
    return inverse_weight(dist)
    

path_list=os.listdir(path)
path_list.sort()
BATCH=[]
count1=0
for f in path_list:
    di=os.path.join(path, f)
    item=os.listdir(di)
    file=[]
    x=[]
    center=[]
    for filename in item:
     if os.path.splitext(filename)[1] == '.npy':
        data=np.load(os.path.join(di,filename),allow_pickle=True)
        p_arr = data.reshape(-1)[0]['feature'].cpu().detach().numpy()
        x.append(p_arr)
        center.append(data.reshape(-1)[0]['center'])
     if os.path.splitext(filename)[1] == '.txt':
        with open(os.path.join(di,filename)) as f1:
            # l=f1.read().split(' ')
            # label=[int(i) for i in l]
            lines = f1.readlines()
            l = lines[1].split(' ')
            label=[int(i) for i in l]

    edge=[[],[]]
    Weight=[]
    count1=count1+len(center)
    for i in range(len(center)):
        temp=create_link(len(center),i)
        weight=creat_weight(temp,center)
        for z in range(len(center)-1):
            edge[0].append(temp[0][z])
            edge[1].append(temp[1][z])
            Weight.append(weight[z])
    x=torch.tensor(x,dtype=torch.float)
    label=torch.tensor(label,dtype=torch.long)
    edge=torch.tensor(edge,dtype=torch.long)
    we=torch.tensor(Weight,dtype=torch.float)
    data=Data(x=x,y=label,edge_index=edge,edge_weight=we)
    BATCH.append(data)


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 =GraphConv(1024, hidden_channels)
        # self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 =GraphConv(hidden_channels, 128)
        # self.bn2 = nn.BatchNorm1d(128)
        self.conv3 =GraphConv(128, 4)

    def forward(self, x, edge_index,edge_weight, batch):
        x = self.conv1(x=x, edge_index=edge_index,edge_weight=edge_weight)
        # x = self.bn1(x)
        x = x.relu()
        x = self.conv2(x=x, edge_index=edge_index,edge_weight=edge_weight)
        # x = self.bn2(x)
        x = x.relu()
        # x=readou
        # pro=self.mlp(x)
        x = self.conv3(x=x, edge_index=edge_index,edge_weight=edge_weight)
        return x

model = GNN(hidden_channels=256)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

criterion1=torch.nn.CrossEntropyLoss()

def test(loader):
    model.eval()
    correct=0
    count1=0
    for t in loader:
        count1+=len(t.y)
    loader=DataLoader(loader, batch_size=1)

    for data in loader:
        out=model(data.x,data.edge_index,data.edge_weight,data.batch)
        pred1=out.argmax(dim=1)
        correct+=int((pred1==data.y).sum())
    acc = correct/count1
    print(acc)
    return acc


def train_model(datasets):
    model.train()
    datas=DataLoader(datasets, batch_size=32, shuffle=True)
    for _ in range(300):
      for data in datas:
        out=model(data.x,data.edge_index,data.edge_weight,data.batch)
        loss1=criterion1(out,data.y)
        loss1.backward()
     
        optimizer.step()
        optimizer.zero_grad()
    # Save the trained model
    torch.save(model.state_dict(), model_path + '/object_order.pt')
def sample(ind,data):
    sets=[]
    for i in ind:
        sets.append(data[i])
    return sets
shuffle_indexes=np.random.permutation(len(BATCH))

test_size=int(len(BATCH)*0.2)     #规定测试集个数
 
test_ind=shuffle_indexes[:test_size]  #测试集索引
train_ind=shuffle_indexes[test_size:] #训练集索引

train_set=[BATCH[i] for i in train_ind]
test_set=[BATCH[i] for i in test_ind]

train_model(train_set)
acc=test(test_set)
