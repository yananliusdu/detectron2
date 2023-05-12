# -*- coding: utf-8 -*-
"""
Created on Wed May 10 17:32:34 2023

@author: 97091
"""
import numpy as np

import cv2
import torch
from torch_geometric.data import Data

import os
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
    # inverse_weight(dist)
    return inverse_weight(dist)

def draw_result(out,image,boxes):
    cant = (0, 0, 255)
    can=(0, 255,0 )
   
    for o,box in zip(out,boxes):
        if o==0:
            cv2.rectangle(image, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), cant, 2)
        if o==1:
            cv2.rectangle(image, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), can, 2)
        
    return image
def transelate(output):
        # output=list(output.cpu().detach().numpy())
        idx_to_word= {1:"yes", 2:"no", 3:"move",4:'b',5:'c',
                                     6:'help'} 
        output=list(filter(lambda x: x != 0, output))
        
        output=[idx_to_word[i] for i in output]
        return output

def create_graph(f):
    path=r'/home/yanan/project-2/project/test_tmp'
    path_list=os.listdir(path)
    path_list.sort()
    
    count1=0
    di=os.path.join(path, str(f))
    item=sorted(os.listdir(di))
   
    x=[]
    center=[]
    boxes=[]
    goal=[]
    for filename in item:
     if os.path.splitext(filename)[1] == '.npy':
        
        data=np.load(os.path.join(di,filename),allow_pickle=True)
        p_arr = np.append(data.reshape(-1)[0]['feature'].cpu().detach().numpy(),[data.reshape(-1)[0]['label']]*64)
        x.append(p_arr)
        center.append(data.reshape(-1)[0]['center'])
        boxes.append(data.reshape(-1)[0]['box'])
    
     if os.path.splitext(filename)[0] == 'goal':
        with open(os.path.join(di,filename)) as f1:
            g=f1.read().split(' ')
            goal=[int(i) for i in g]


    P=[]
    for p,g in zip(x,goal):
     P.append(np.append(p,[g]*64))
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
    x=torch.tensor(P,dtype=torch.float)
    edge=torch.tensor(edge,dtype=torch.long)
    we=torch.tensor(Weight,dtype=torch.float)
    data=Data(x=x,edge_index=edge,edge_weight=we)
    return data,boxes
def output(data,encoder,decoder,image,boxes):
    batch=[0]*data.x.size()[0]
    batch=torch.tensor(batch)
    out=encoder(data.x,data.edge_index,data.edge_weight,batch)
    out=torch.argmax(out,dim=1).cpu().detach().numpy()


    for i in range(len(data.x)):
        embedding=encoder.node_embeedings(data.x, data.edge_index,data.edge_weight, i)
        output=decoder.Predict(embedding.unsqueeze(0))
        print(transelate(output))
    image=draw_result(out,image,boxes)
    return image
