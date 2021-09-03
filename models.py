# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 15:00:55 2021

@author: jdetu
"""
import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    
    def __init__(self, obs_shape, num_actions):
        super(Model, self).__init__()
        # TODO: increase model architecture
        #assert len(obs_shape) == 1, "Error in Observation shape."
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        
        self.net = nn.Sequential(
            nn.Linear(obs_shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, num_actions))
        
        self.opt = optim.Adam(self.net.parameters(), lr = 1e-4)
        
    def forward(self, x):
        return self.net(x)
        
    
class ConvModel(nn.Module):

    def __init__(self, obs_shape, num_actions, lr = 1e-4):
        #assert len(obs_shape) == 3
        super(ConvModel, self).__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        #input shape 84x84
        self.conv_net = torch.nn.Sequential(
            nn.Conv2d(1, 16, (8, 8), stride = (4,4)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (4, 4), stride = (2,2)),
            nn.ReLU()
            )
        
        with torch.no_grad(): # calc conv net neuron size
            dummy = torch.zeros((1, *obs_shape))
            #print(dummy.shape)
            x = self.conv_net(dummy)
            s = x.shape
            fc_size = s[1] * s[2] * s[3]
            
        self.fc_net = torch.nn.Sequential(
            nn.Linear(fc_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
            )
        
        self.opt = optim.Adam(self.parameters(), lr = lr)
            
        
    def forward(self, x):
        conv_latent = self.conv_net(x/255.0)
        return self.fc_net(conv_latent.view((conv_latent.shape[0], -1)))
    
class ConvModel2(nn.Module):

    def __init__(self, obs_shape, num_actions, lr = 1e-4):
        #assert len(obs_shape) == 3
        super(ConvModel2, self).__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        #input shape 84x84
        self.conv_net = torch.nn.Sequential(
            nn.Conv2d(4, 16, (8, 8), stride = (4,4)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (4, 4), stride = (2,2)),
            nn.ReLU()
            )
        
        with torch.no_grad(): # calc conv net neuron size
            dummy = torch.zeros((1, *obs_shape))
            #print(dummy.shape)
            x = self.conv_net(dummy)
            s = x.shape
            fc_size = s[1] * s[2] * s[3]
            
        self.fc_net = torch.nn.Sequential(
            nn.Linear(fc_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
            )
        
        self.opt = optim.Adam(self.parameters(), lr = lr)
            
        
    def forward(self, x):
        conv_latent = self.conv_net(x/255.0)
        return self.fc_net(conv_latent.view((conv_latent.shape[0], -1)))
    
if __name__ == '__main__':
    m = ConvModel((4, 84, 84), 4)
    tensor = torch.zeros((1, 4, 84, 84))
    print(m.forward(tensor))
    