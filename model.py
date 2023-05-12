# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:11:39 2023

@author: 97091
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 =GraphConv(1152, hidden_channels) 
        self.conv2 =GraphConv(hidden_channels, 128)  
        self.conv3 =GraphConv(128, 2) 
      
       
        
        
        
        
       
        

    def forward(self, x, edge_index,edge_weight, batch):
        x = self.conv1(x=x, edge_index=edge_index,edge_weight=edge_weight)
        x = x.relu()
        x = self.conv2(x=x, edge_index=edge_index,edge_weight=edge_weight)
        x = x.relu()
        
        
        x = self.conv3(x=x, edge_index=edge_index,edge_weight=edge_weight)
       
        
        
        
        
       
        
        
        
        
        
        
        return x
    def node_embeedings(self,x,edge_index,edge_weight,obj1):
        x = self.conv1(x=x, edge_index=edge_index,edge_weight=edge_weight)
        x = x.relu()
        x = self.conv2(x=x, edge_index=edge_index,edge_weight=edge_weight)
        x = x.relu()
        # out=torch.argmax(self.conv3(x=x, edge_index=edge_index,edge_weight=edge_weight),dim=1)
        # x=torch.cat((x,out.reshape(out.shape[0],1)*torch.ones(1,16)),dim=1)
        return x[obj1]
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super( DecoderRNN , self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size 
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding( self.vocab_size , self.embed_size )
        self.lstm  = nn.LSTM(    input_size  =  self.embed_size , 
                             hidden_size = self.hidden_size,
                             num_layers  = self.num_layers ,
                             batch_first = True 
                             )
        self.fc = nn.Linear( self.hidden_size , self.vocab_size  )
        

    def init_hidden( self, batch_size ):
      return ( torch.zeros( self.num_layers , batch_size , self.hidden_size  ),
      torch.zeros( self.num_layers , batch_size , self.hidden_size  ))
    
    def forward(self, features, captions):            
      captions = captions[:,:-1]      
      self.batch_size = features.shape[0]
      self.hidden = self.init_hidden( self.batch_size )
      embeds = self.word_embedding( captions )
      
      
      inputs = torch.cat( ( features , embeds ) , dim =1  )      
      lstm_out , self.hidden = self.lstm(inputs , self.hidden)      
      outputs = self.fc( lstm_out )      
      return outputs

    def Predict(self, inputs):        
        final_output = []
        batch_size = inputs.shape[0]         
        hidden = self.init_hidden(batch_size) 
       
        inputs=inputs.unsqueeze(1)
        while True:
            lstm_out, hidden = self.lstm(inputs, hidden) 
            outputs = self.fc(lstm_out)  
            outputs = outputs.squeeze(1) 
            _, max_idx = torch.max(outputs, dim=1) 
            final_output.append(max_idx.cpu().numpy()[0].item())             
            if (len(final_output) >=3):
                break
            
            inputs = self.word_embedding(max_idx) 
            inputs = inputs.unsqueeze(1) 
                    
        return final_output  