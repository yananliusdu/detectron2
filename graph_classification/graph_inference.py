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
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import cv2
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear

# part2-0, part3-1, part4-2
tsk_dimension = 32
tsk_part2 = np.array([1 for j in range(tsk_dimension)])
tsk_part3 = np.array([0 for j in range(tsk_dimension)])
tsk_part4 = np.array([0 for j in range(tsk_dimension)])
f='20'
label = 0
def tsk_architecture(label, dimension):
    ar = [0]*3
    ar[label] = 1
    p2 = np.array([ar[0] for j in range(dimension)])
    p3 = np.array([ar[1] for j in range(dimension)])
    p4 = np.array([ar[2] for j in range(dimension)])
    return p2, p3, p4

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
        obj_label = data.reshape(-1)[0]['label']
        tsk_part2, tsk_part3, tsk_part4 = tsk_architecture(label, tsk_dimension)
        if obj_label == 0:
            c = np.concatenate((p_arr, tsk_part2))
        elif obj_label == 1:
            c = np.concatenate((p_arr, tsk_part3))
        elif obj_label == 2:
            c = np.concatenate((p_arr, tsk_part4))
        x.append(c)
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


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(1024 + tsk_dimension, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 128)
        self.conv3 = GCNConv(128, 64)
        self.lin = Linear(64, 3)

    def forward(self, x, edge_index, edge_weight, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


model = GCN(hidden_channels=256)
model.load_state_dict(torch.load('/media/yanan/One Touch/detectron2_data_Ma/detectron2_data/object_to_grasp.pt'))

batch=[0]*data.x.size()[0]
batch=torch.tensor(batch)
out=model(data.x,data.edge_index,data.edge_weight,batch)
out=torch.argmax(out,dim=1).cpu().detach().numpy()
image=draw_result(out,image1,boxes)
cv2.imwrite('/media/yanan/One Touch/detectron2_data_Ma/detectron2_data/show_results/' + f  + '_grasp' + '.png', image)
cv2.imshow('result',image)
cv2.waitKey(0)
