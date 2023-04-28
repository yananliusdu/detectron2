# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:51:44 2023

@author: 97091

GNN inference test
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:48:40 2023

@author: 97091
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv
from torch_geometric.loader import DataLoader
import cv2
import torch.nn.functional as F
import torch.nn as nn

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


def show_order(out,image,boxes):
    cant = (0, 0, 255)
    can=(0, 255,0 )
    can_txt = (255,0,0)
    for o,box in zip(out,boxes):
        if o==0:
            cv2.rectangle(image, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), cant, 2)
        if o==1:
            cv2.rectangle(image, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), can, 2)
            cv2.putText(image, str(o), (int(box[0]),int(box[1])+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, can_txt, 2)
        if o==2:
            cv2.rectangle(image, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), can, 2)
            cv2.putText(image, str(o), (int(box[0]),int(box[1])+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, can_txt, 2)
        if o==3:
            cv2.rectangle(image, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), can, 2)
            cv2.putText(image, str(o), (int(box[0]),int(box[1])+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, can_txt, 2)
    return image


def draw_result(out,image,boxes):
    cant = (0, 0, 255)
    can=(0, 255,0 )
    for o,box in zip(out,boxes):
        if o==0:
            cv2.rectangle(image, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), cant, 2)
        if o==1:
            cv2.rectangle(image, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), can, 2)
    return image

path='/media/yanan/One Touch/detectron2_data_Ma/detectron2_data/test'
path_list=os.listdir(path)
path_list.sort()
BATCH=[]
count1=0
f='108'
di=os.path.join(path, f)
item=os.listdir(di)
file=[]
x=[]
center=[]
boxes=[]
for filename in item:
     if os.path.splitext(filename)[1] == '.npy':
        
        data=np.load(os.path.join(di,filename),allow_pickle=True)
        p_arr = data.reshape(-1)[0]['feature'].cpu().detach().numpy()
        x.append(p_arr)
        center.append(data.reshape(-1)[0]['center'])
        boxes.append(data.reshape(-1)[0]['box'])
     if os.path.splitext(filename)[1] == '.txt':
        with open(os.path.join(di,filename)) as f1:
            l=f1.read().split(' ')
            label=[int(i) for i in l]
     if os.path.splitext(filename)[0] == 'ori':
        image1= cv2.imread(os.path.join(di,filename))

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
# label=torch.tensor(label,dtype=torch.long)
edge=torch.tensor(edge,dtype=torch.long)
we=torch.tensor(Weight,dtype=torch.float)
data=Data(x=x,edge_index=edge,edge_weight=we)


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
model.load_state_dict(torch.load('/media/yanan/One Touch/detectron2_data_Ma/detectron2_data/object_order.pt'))

batch=[0]*data.x.size()[0]
batch=torch.tensor(batch)
out=model(data.x,data.edge_index,data.edge_weight,batch)
out=torch.argmax(out,dim=1).cpu().detach().numpy()
image=show_order(out,image1,boxes)
cv2.imwrite('/media/yanan/One Touch/detectron2_data_Ma/detectron2_data/show_results/' + f + '_order' + '.png', image)
cv2.imshow('result',image)
cv2.waitKey(0)
