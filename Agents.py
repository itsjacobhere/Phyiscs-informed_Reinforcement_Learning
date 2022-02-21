
import numpy as np
from PINN_3D import PINN
import scipy
from pyDOE import lhs
import cv2

#import libraries
import gym

import time
import wandb
import numpy as np


import torch 
from matplotlib import pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm 
from dataclasses import dataclass
from typing import Any
from models import Model as Stock_NN
from models import ConvModel2 as ConvModel


import matplotlib.pyplot as plt
is_ipython = 'inline' in plt.get_backend()
if is_ipython: from IPython import display
if is_ipython: display.clear_output(wait=True)
    
    
l2 = [20,20,20,20,20,20,20, 20]
sensor_coords = np.array([[0.1, -0.9],[0.9,0.9],[0.5,0.45]])

# loads matlab solution from raissi et al, data used in init conditions
y = np.linspace(-1,1,256).reshape(-1,1)
x = np.linspace(0,0.99,100).reshape(-1,1)                                  
usol = np.zeros([256,100]) #data['usol']                            

Y, X = np.meshgrid(y,x)                         

ic = usol.copy()

initial_temp = 0

ic.fill(initial_temp)

#plt.pcolormesh(X,Y,ic.T,)
#plt.colorbar()

ic = ic.T.reshape(100,1,256)

timelen = 4

time = np.linspace(0,1, timelen)
time = time.reshape(-1,1)

Y, X, T= np.meshgrid(y,x, time)

# test data is the 3D coordinates in [y,x,t]
X_u_test = np.hstack((Y.flatten()[:,None], X.flatten()[:,None], T.flatten()[:,None]))

# Domain bounds
lb = X_u_test[0] 
ub = X_u_test[-1] 



def trainingdata(ic = ic, n_bc = 100, n_coll = 10000, n_ic = 100, temp = 0, 
                 sensor_coords = None, sensor_values = None):
    '''generates test sample coordinates for initial run of PINN.'''
    
    # initial conditions: 100 x 256 where t = 0
    init_x = np.hstack((Y[:,:,0][:,None], X[:,:,0][:,None], T[:,:,0][:,None]))
    init_u = ic

    #where x = 0 for t and y
    leftedge_x = np.hstack((Y[0,:][:,None], X[0,:][:,None], T[0,:][:,None])) #L1
    leftedge_u = np.array([0]*y.shape[0]).reshape(-1,1) #usol[:,0][:,None] #* initial_temp #np.full([256,1], 0) #np.full([256,1], 0) 

    #where x = 1 for all t and y
    rightedge_x = np.hstack((Y[0,:][:,None], X[-1,:][:,None], T[-1,:][:,None])) #L1
    rightedge_u =  np.array([0]*y.shape[0]).reshape(-1,1) #usol[:,0][:,None] #* initial_temp  #np.full([256,1], 0)# np.full([256,1], 0)#

    #bottom where y = -1
    bottomedge_x = np.hstack((Y[:,0][:,None], X[:,0][:,None], T[:,0][:,None])) #L2
    bottomedge_u = np.array([temp]*x.shape[0]).reshape(-1,1) #usol[-1,:][:,None] #np.full([100,1], 0) #

    #top where y = 1
    topedge_x = np.hstack((Y[:,-1][:,None], X[:,0][:,None], T[:,-1][:,None])) #L3
    topedge_u = np.array([temp]*x.shape[0]).reshape(-1,1) #usol[0,:][:,None] #np.full([100,1], 0)  #

    all_bc_x = np.vstack([
                               leftedge_x, 
                               rightedge_x, 
                               bottomedge_x, 
                               topedge_x]) 

    all_bc_u = np.vstack([
                             leftedge_u, 
                             rightedge_u, 
                             bottomedge_u, 
                             topedge_u])   

    #choose random n_bc points for training
    idx = np.random.choice(all_bc_x.shape[0], n_bc, replace=False) 

    bc_x = all_bc_x[idx, :] 
    bc_u = all_bc_u[idx,:]     

    id_x = np.random.choice(100, n_ic, replace=False) 
    id_y = np.random.choice(256, n_ic, replace=False)
    
    ic_x = init_x[id_x,:,id_y]
    ic_u = init_u[id_x,:,id_y]

    # create collocation points
    store = []
    for i in range(time.shape[0]):
        coll_points = lb[:2] + (ub[:2] - lb[:2]) * lhs(2,n_coll)
        # assert collocation points have been sampled from the correct range...
        assert((coll_points[:, 1] >= 0.0).all() and (coll_points[:, 1] <= 1.0).all())
        assert((coll_points[:, 0] >= -1.0).all() and (coll_points[:, 1] <= 1.0).all())
        store.append(coll_points) # coll points for every frame t

    # convert to array    
    s = np.array(store)

    t_ = np.array(([time]*n_coll))
    f_x = np.concatenate((s, t_.reshape(time.shape[0], n_coll, 1)), axis = 2).reshape(n_coll, 3, time.shape[0]) 
    
    # flip coordinates # this is done very badly, fix it
    X_u_copy = bc_x.copy()
    ytemp = X_u_copy[:,0,:]
    xtemp = X_u_copy[:,1,:]
    X_u_copy[:,0,:] = xtemp
    X_u_copy[:,1,:] = ytemp
    
    
    f_x = np.vstack((f_x, X_u_copy)) # append boundary coords to collocation coords
    
    if sensor_coords is not None:
        sensor_coords = np.append(np.flip(sensor_coords), np.array([[0],[0],[0]]),axis = 1 )
        ic_x = np.append(ic_x,sensor_coords, axis = 0)
        ic_u = np.append(ic_u, sensor_values, axis = 0)
        
    return f_x, bc_x, bc_u, ic_x, ic_u


# create the training data
n_bc = 100 # num boundary condition exemplars to sample
n_coll = 10000 # num coll points in each time frame to constrain f
n_ic = 100 # num init condition exemplars to sample
f_x, bc_x, bc_u, ic_x, ic_u = trainingdata(ic, n_bc, n_coll, n_ic, temp = 0)

import random
class Exp_Replay:
    """Experience replay, old samples are removed beyond specified limit
    stores collection of experience tuples (sars), this combats experience correlation"""
    def __init__(self, buffer_n = int(1e5)):   
        
        self.buffer_n = buffer_n
        self.buffer = [None]*buffer_n
        self.idx = 0
        
    def insert(self, sars):
        i = self.idx % self.buffer_n
        self.buffer[i] = sars
        self.idx +=1 # update index
        
    def sample(self, n_sample):
        if self.idx < self.buffer_n:
            return random.sample(self.buffer[:self.idx], n_sample)
        return random.sample(self.buffer, n_sample)
    
# for data storage
@dataclass
class Sars: # store experience tuples
    state: Any
    action: int
    reward: float
    next_state : Any
    done : bool

def add_sensor_vals(u_pred,obs):
    x_idx = np.round(sensor_coords[:,0]*84).reshape(-1,1)
    y_idx = np.round(((sensor_coords[:,1]  - -1) / (1 - -1)) * 84).reshape(-1,1)
    idx = np.append(x_idx,y_idx, axis = 1 ).astype(int) # 0 to 1
    for i in range(u_pred.shape[-1]):
        u_pred[idx[:,0],idx[:,1],i] = obs.reshape(3)
    return u_pred


class DQN_Agent:
    
    def __init__(self, env,
                 learning_rate = 1e-4, 
                 discount_rate = 0.99,
                 eps_max = 0.9, 
                 eps_min = 0.01,
                 eps_decay = 1e-6, 
                 boltzman_exploration = False,
                 min_rb_size = int(2e4), 
                 sample_size = 100,
                 model_train_freq = 100,
                 tgt_update_freq = 5000,
                 max_epoch = 0, 
                 load_model = None,
                 load_PINN = None,
                 use_PINN = True,
                 device = 'cuda:0',
                 name = 'Breakout',
                 description = '__'):
        
        self.lr = learning_rate
        self.gamma = discount_rate
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        
        self.boltzman = boltzman_exploration 
        self.min_rb = min_rb_size
        self.sample_size = sample_size 
        self.model_train_freq = model_train_freq 
        self.tgt_update_freq = tgt_update_freq
        self.max_epoch = max_epoch 
        self.load_model = load_model
        self.load_PINN = load_PINN
        self.use_PINN = use_PINN
        self.device = device
        self.name = name
        self.id = np.nan
        self.descrip = description 
        self.explore = True
        
        self.log = {'loss': [],
                    'avg_reward': [],
                    'eps': [],
                    'step_num': [],
                    'PINN':[]}
        
        if self.use_PINN: self.pinn = PINN(layers = l2, device = device, sensor_coords = sensor_coords) 
        
        # init env
        self.env = env #gym.make('stocks-v0', frame_bound=(15, 200), window_size=15)
        
        
        self.step_count = 0
        
        return
    
    def choose_action(self, eps):
        
        if self.boltzman: # use boltzman exploration
                logits = self.m(torch.Tensor(self.last_observation).unsqueeze(0).to(self.device))[0]
                action = torch.distributions.Categorical(logits = logits).sample().item()
        else:
            if np.random.random() < eps: # explore action space
                action = self.env.action_space.sample()
            else: # greedy action
                action = self.m(torch.Tensor(self.last_observation)
                           .unsqueeze(0).to(self.device)).max(-1)[-1].item()
        return action
    
    def run_episode(self, episode = 0, render = False, explore = True,
                   load_model = None,
                   load_PINN = None):
        '''runs one episode in the taining process.'''
        self.explore = explore 
        if not explore:
            self.max_epoch = 0
            self.train(episode_num = 1, use_wandb = False)
        
        # compute decaying exploration rate as a function of episode
        eps = (self.eps_max - self.eps_min) * np.exp(-self.eps_decay*self.step_count) + self.eps_min
        
        if load_PINN is not None and self.use_PINN:
            self.pinn = PINN(layers = l2, device = self.device, sensor_coords = sensor_coords)
            self.pinn.load_state_dict(torch.load(load_PINN))
        if load_model is not None:
            # instantiate prediction network
            self.m = ConvModel(self.u_pred_shape, #self.env.observation_space.shape,
                           self.env.action_space.n).to(self.device)
            self.m.load_state_dict(torch.load(load_model))
        
        
        if self.use_PINN:
            ic = usol.copy()
            initial_temp = 0
            ic.fill(initial_temp)
            ic = ic.T.reshape(100,1,256)

            f_x, bc_x, bc_u, ic_x, ic_u = trainingdata(ic, n_bc, n_coll, n_ic, temp = 0)
            obs = self.env.reset()
            LBFGS = True
            self.pinn.train(bc_x[:,:,0], bc_u, f_x[:,:,0], ic_x, ic_u,
                      epochs = 1, 
                      LBFGS = LBFGS,
                      K = .01)
            u_pred = self.pinn.predict(X_u_test, load_model = None, time = timelen)
            ic = u_pred[:,:,-1].reshape(100,1,256) # save last frame as ic to input next prediction
            u_pred = cv2.resize(u_pred, (84,84)) # resize 
            u_pred = np.transpose(u_pred,(2,0,1))

            self.last_observation = u_pred #self.env.reset()
        else:
            self.last_observation = self.env.reset()
        
        done = False
        
        prev_action = None
        loss = np.nan
        rolling_reward = 0
        while not done: # until episode ends
            if not render: self.tq.update()
            #print(self.step_num)
            
            # choose action
            if not explore: eps = 0
            action = self.choose_action(eps)
            
            # observe state reward by taking action
            obs, reward, done, info = self.env.step(action)
            rolling_reward += reward # sum reward for episode
            
            if self.use_PINN:
                LBFGS = False
                if self.step_count % 150 == 0:
                    LBFGS = True

                if prev_action == 1:
                    a = 20
                elif prev_action == 0:
                    a = -20
                else:
                    a = 0

                f_x, bc_x, bc_u, ic_x, ic_u = trainingdata(ic, n_bc, n_coll, n_ic, temp = a,
                                                          sensor_coords = sensor_coords,
                                                          sensor_values = obs)
                #print(obs[2])

                prev_action = action
                self.pinn.train(bc_x[:,:,0], bc_u, f_x[:,:,0], ic_x, ic_u,
                      epochs = 1, 
                      LBFGS = LBFGS,
                      K = .01,
                      max_iter = 1,
                      source = np.mean(obs),
                      sensor_values = obs) # a

                u_pred = self.pinn.predict(X_u_test, load_model = None, time = timelen)

                ic = u_pred[:,:,-1].reshape(100,1,256)
                u_pred = cv2.resize(u_pred, (84,84))
                u_pred = add_sensor_vals(u_pred,obs)
                u_pred = np.transpose(u_pred,(2,0,1))
                #self.log['PINN'].append(u_pred)
                obs = u_pred
            
            # insert experience tuple at top of buffer
            self.rb.insert(Sars(self.last_observation, action, reward, obs, done))

            self.last_observation = obs # update observation
            
            #  counters
            self.steps_since_train += 1
            self.step_num += 1
            self.step_count += 1
            
            if render:
                self.env.render()
                clear_output(wait=True)
            
            
            # train prediction network
            if  explore and self.steps_since_train > self.model_train_freq and self.rb.idx > self.min_rb:
                #print('Logging')
                # train model neural network
                loss = self.train_NN(self.m, 
                                     self.rb.sample(self.sample_size), 
                                     self.tgt,
                                     self.env.action_space.n,
                                     self.device)
                self.steps_since_train = 0 # reset train counter
                
                
                wandb.log({'loss': loss.detach().cpu().item(), 
                           'epsilon': eps, 
                           'avg_reward': self.episode_rewards[-1]}, 
                          step = self.step_num)
                self.save_reward = np.mean(self.episode_rewards[-1])

                self.epochs_since_tgt_update +=1

                # update target nn
                if self.epochs_since_tgt_update > self.tgt_update_freq:
                    self.tgt.load_state_dict(self.m.state_dict())
                    self.epochs_since_tgt_update = 0

                self.epoch += 1  
            
            #self.log['loss'].append(loss.detach().cpu().item())
            self.log['avg_reward'].append(rolling_reward)
            self.log['eps'].append(eps)
            self.log['step_num'].append(self.step_num)
                    
        return rolling_reward # return episode rewards
    
    def train_NN(self, 
                 model,
                 transition, 
                 tgt, 
                 num_actions, 
                 device):
        '''trains model passed'''
        
        curr_states = torch.stack([torch.Tensor(s.state) for s in transition]).to(device)
        rewards = torch.stack([torch.Tensor([s.reward]) for s in transition]).to(device)
        next_states = torch.stack([torch.Tensor(s.next_state) for s in transition]).to(device)
        actions = [s.action for s in transition]
        if_done = torch.stack([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in transition]).to(device)
        
        

        with torch.no_grad(): # get best next actions with target network
            next_qvals = tgt(next_states).max(-1)[0] #(N, num_actions)

        model.opt.zero_grad()
        qvals = model(curr_states) # shape: (N, num_actins), get qvals of current state
        H_actions = torch.nn.functional.one_hot(torch.LongTensor(actions), num_actions).to(device)
        
        # MSE loss function
        #loss = ((rewards + if_done[:,0]*next_qvals - torch.sum(qvals * H_actions, -1))**2).mean()

        f_loss = torch.nn.SmoothL1Loss()
        target = torch.sum(qvals * H_actions, -1)
        inputs = rewards.squeeze() + if_done[:,0]*self.gamma*next_qvals # Bellman optimality
        loss = f_loss(target, inputs )
        loss.backward()
        model.opt.step()

        return loss
    
    def train(self, max_epoch = np.inf, identifier = 0, episode_num = 0, use_wandb = True):
        '''begin training by running episodes until max or interrupted'''
        
        self.id = identifier
        self.max_epoch = max_epoch 
        
        if self.name is None:
            self.name = str(self.descrip) + '_eps_' + str(self.eps_max) + '_rb_' + str(self.min_rb) + '_samples_' + str(self.sample_size) + '_tgt_' + str(self.tgt_update_freq) + '_id_' + str(self.id) + '_.pth'
        
        # init w and b for data viz in dashboard
        if use_wandb: wandb.init(project = "Dissertation_Final", name = self.name)    
        
        #instantiate PINN
        
        self.pinn.train(bc_x[:,:,0], bc_u, f_x[:,:,0], ic_x, ic_u,
                  epochs = 1, 
                  LBFGS = False,
                  K = .01)
        u_pred = self.pinn.predict(X_u_test, load_model = None, time = timelen)
        u_pred = cv2.resize(u_pred, (84,84))
        u_pred = u_pred.reshape(4,84,84)
        self.u_pred_shape = u_pred.shape
        
        if self.load_PINN is not None and self.use_PINN:
            self.pinn.load_state_dict(torch.load(self.load_PINN))
        
        # instantiate prediction network
        self.m = ConvModel(u_pred.shape, #self.env.observation_space.shape,
                           self.env.action_space.n).to(self.device)
        if self.load_model is not None:
            self.m.load_state_dict(torch.load(self.load_model))
        
        # instantiate target network
        self.tgt = ConvModel(u_pred.shape, #self.env.observation_space.shape, 
                        self.env.action_space.n).to(self.device)
        self.tgt.load_state_dict(self.m.state_dict()) 
        
        # instantiate buffer
        self.rb = Exp_Replay()
        
        # init counterstw
        self.epoch = 0
        self.steps_since_train = 0
        self.epochs_since_tgt_update = 0
        self.step_num = -self.min_rb
        self.step_count = 0
        self.episode_rewards = [np.nan]
        episode = episode_num
        
        self.tq = tqdm()
        
        if not self.explore: return
        
        ra = str(np.random.random())[2:8]
        
        try:
            while episode < self.max_epoch:
                #if episode % 5 == 0: print (episode)
                self.episode_rewards.append(self.run_episode(episode))
                episode += 1
                
                if episode % 10 == 0:
                    self.cnn_model = f"models/CNN_3D_Diffusion_{episode}_episodes_{self.id}.pth"
                    self.pinn_model = f"models/PINN_3D_Diffusion_{episode}_episodes_{self.id}.pth"
                    torch.save(self.tgt.state_dict(), self.cnn_model)
                    torch.save(self.pinn.state_dict(), self.pinn_model)
                
            if use_wandb:   
                r = str(np.random.random())[2:8] # random marker
                self.cnn_model = f"models/CNN_3D_Diffusion_{self.step_count}_{r}_{self.id}.pth"
                self.pinn_model = f"models/PINN_3D_Diffusion_{self.step_count}_{r}_{self.id}.pth"
                torch.save(self.tgt.state_dict(), self.cnn_model)
                torch.save(self.pinn.state_dict(), self.pinn_model)
                print('Training Completed')
        except KeyboardInterrupt: # save model on interrupt
            r = str(np.random.random())[2:8]
            self.cnn_model = f"models/CNN_3D_Diffusion_{self.step_count}_{r}_{self.id}.pth"
            self.pinn_model = f"models/PINN_3D_Diffusion_{self.step_count}_{r}_{self.id}.pth"
            torch.save(self.tgt.state_dict(), self.cnn_model)
            torch.save(self.pinn.state_dict(), self.pinn_model)
            print('Training Interrupted')
    
