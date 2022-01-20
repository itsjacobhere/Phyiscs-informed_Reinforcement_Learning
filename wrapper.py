import wandb

# fenics Finite Element
from fenics import *
from mshr import *
from dolfin import *

# typical libaries
import numpy as np
import matplotlib.pyplot as plt

#% matplotlib notebook
from IPython.display import Image
from IPython.display import set_matplotlib_formats
from IPython.display import clear_output
set_matplotlib_formats('png', 'pdf')
get_ipython().run_line_magic('matplotlib', 'inline')

import gym
from gym import spaces

from wandb.integration.sb3 import WandbCallback

from tqdm import tqdm
from PINN_3D import PINN
from pyDOE import lhs         #Hypercube Sampling
import scipy.io
import torch
#Set default dtype to float32
torch.set_default_dtype(torch.float)
 
y = np.linspace(-1,1,256).reshape(-1,1)
x = np.linspace(0,0.99,100).reshape(-1,1)                                    
usol = np.zeros([256,100])                            

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

np.zeros([256,100]).shape

# test data is the 3D coordinates in [y,x,t]
X_u_test = np.hstack((Y.flatten()[:,None], X.flatten()[:,None], T.flatten()[:,None]))

# Domain bounds
lb = X_u_test[0] 
ub = X_u_test[-1] 

def sample_coords(ic = ic, n_bc = 100, n_coll = 10000, n_ic = 100, temp = 0, 
                 sensor_coords = None, sensor_values = None):

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
    for i in range(timelen):
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
        
        #print(ic_x.shape, sensor_coords.shape)
        #print(ic_u.shape, sensor_values.shape)
        
    return f_x, bc_x, bc_u, ic_x, ic_u

obs = np.array([0,0,0]).reshape(-1,1)

# create the training data
n_bc = 100 # num boundary condition exemplars to sample
n_coll = 10000 # num coll points in each time frame to constrain f
n_ic = 100 # num init condition exemplars to sample
sensor_coords = np.array([[0.1, -0.9],[0.9,0.9],[0.5,0.45]])

f_x, bc_x, bc_u, ic_x, ic_u = sample_coords(ic, n_bc, n_coll, n_ic, temp = 0,
                                          sensor_coords = sensor_coords,
                                          sensor_values = obs)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# various neural network architectures
l1 = [50,50,50,50,50,50,50,50,50,50]
l2 = [20,20,20,20,20,20,20, 20]
l3 = [20,50,100,100,100,100,100,100,50,20]




# instantiate PINN neural network
#pinn = PINN(layers = l2, device = device)  

def add_sensor_vals(u_pred,obs):
    x_idx = np.round(sensor_coords[:,0]*84).reshape(-1,1)
    y_idx = np.round(((sensor_coords[:,1]  - -1) / (1 - -1)) * 84).reshape(-1,1)
    idx = np.append(x_idx,y_idx, axis = 1 ).astype(int) # 0 to 1
    for i in range(u_pred.shape[-1]):
        u_pred[idx[:,0],idx[:,1],i] = obs.reshape(3)
    return u_pred

from environment import heat_diffusion
import cv2

import wandb

class PINN_env(gym.Env):
 '''wrapper for PDE-governed environment and external RL agents. 
 Runs PINN to interface with input feature extraction neural network of external agent
 '''
    metadata = {'render.modes': ['human']}
    
    def __init__(self, env, verbose = False,
                standardize_obs = False,
                norm_obs = False,
                discrete_bc = False,
                mean_obs = True,
                use_wandb = False,
                max_loss = np.inf,
                min_delta = 0,
                max_iter = 100,
                flow_vel = 0,
                K = 0.01):
        super(PINN_env, self).__init__()
        from gym import spaces
        
        self.verbose = verbose
        self.norm_obs = norm_obs
        self.stand_obs = standardize_obs
        self.discrete_bc = discrete_bc
        self.mean_obs = mean_obs
        self.use_wandb = use_wandb
        self.max_episode_timesteps = 200
        self.max_loss = max_loss
        self.min_delta = min_delta
        self.max_iter = max_iter
        self.flow_vel = flow_vel
        self.k = K
        self.env = env
        
        HEIGHT = 84 #100
        WIDTH = 84 #256
        N_CHANNELS = 4
        
        self.info = {'energy':[],
                     'action':[None],
                     'reward':[],
                     'cost':[],
                     'PINN_loss':[]} 
    
        
        self.observation_space = spaces.Box(low=0, high=255, shape=
                    (N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)
        self.action_space = env.action_space
        self.pinn = PINN(layers = l2, sensor_coords = env.sensors, device = 'cuda')
        self.reward_range = env.reward_range
        
        ic = np.zeros([256,100])
        initial_temp = 0
        ic.fill(initial_temp)
        self.ic = ic.T.reshape(100,1,256)
        
        self.prev_action = None
        
        self.tq = tqdm()
        
    def step(self, action):
        #print(action)
        
        obs, reward, done, info = self.env.step(action)
        self.sensor_temps = obs 
        
        assert len(self.env.sensors) == len(obs)
        
        #prev_action = self.info['action'][-1] 
        if self.env.continuous and self.prev_action is not None and not self.discrete_bc:
                a = int(self.env.power_to_temp(self.prev_action))
        elif self.prev_action is not None and (self.prev_action == 1 or self.prev_action > 0): # previous action instead
                a = 20
        elif self.prev_action is not None and (self.prev_action == 0 or self.prev_action <= 0):
                a = -20
        else:
                a = 0
        
        #self.ic += np.mean(obs)
        
        f_x, bc_x, bc_u, ic_x, ic_u = sample_coords(self.ic, n_bc, n_coll, n_ic, 
                                                      temp = a,
                                                      sensor_coords = self.env.sensors,
                                                      sensor_values = obs)
        LBFGS = False
        if self.env.counter % 100 == 0:
            LBFGS = True
            
        if self.flow_vel != 0:
            self.flow_vel = action[1]
        
        max_idx = np.argmax(abs(obs))
        s = obs[max_idx][0]
        if self.mean_obs: s = np.mean(obs)
        loss = self.pinn.train(bc_x[:,:,0], bc_u, f_x[:,:,0], ic_x, ic_u,
                  epochs = self.max_iter, 
                  LBFGS = LBFGS,
                  K = self.k,
                  max_iter = 100,
                  source = s,
                  sensor_values = -obs,
                  max_loss = self.max_loss,
                  min_delta = self.min_delta,
                  flow_velocity = self.flow_vel)
        
        u_pred = self.pinn.predict(X_u_test, time = timelen)
        
        self.ic = u_pred[:,:,-1].reshape(100,1,256)
        self.u_pred = cv2.resize(u_pred, (84,84))
        #if self.norm_obs: self.u_pred = (self.u_pred - self.u_pred.mean())/self.u_pred.std()
        obs = add_sensor_vals(self.u_pred,obs)
        obs = np.transpose(obs,(2,0,1))
        if self.norm_obs:
            obs = (obs - obs.min())/(obs.max()-obs.min())
        if self.stand_obs:
            obs = (obs - obs.mean())/obs.std()
        
        
        self.info['energy'].append(self.env.energy_flux)
        self.info['action'].append(action)
        self.info['reward'].append(reward)
        self.info['PINN_loss'].append(loss[-1])
        #self.info['PINN'].append(obs)
        
        if done and self.use_wandb: 
            wandb.log({'PINN_loss':loss[-1].cpu().detach().numpy()},
                          step = self.env.step_tot)
        if self.verbose: self.tq.update()
        
        self.prev_action = action if not self.env.continuous else action[0]
        
        return obs, reward, done, info
    
    def reset(self, ic_temp = None):
        obs = self.env.reset(ic_temp = ic_temp)
        a = 0
        self.info['action'].append(None)
        self.prev_action = None
        
        ic = usol.copy()
        initial_temp = 0
        ic.fill(initial_temp)
        self.ic = ic.T.reshape(100,1,256)
        
        f_x, bc_x, bc_u, ic_x, ic_u = sample_coords(self.ic, n_bc, n_coll, n_ic, temp = a,
                                                      sensor_coords = self.env.sensors,
                                                      sensor_values = obs)
        
        self.pinn.train(bc_x[:,:,0], bc_u, f_x[:,:,0], ic_x, ic_u,
                  epochs = 1, 
                  LBFGS = True,
                  K = self.k,
                  max_iter = 1,
                  source = 0,
                  sensor_values = obs)
        
        obs = self.pinn.predict(X_u_test, time = timelen)
        obs = cv2.resize(obs, (84,84))
        obs = np.transpose(obs,(2,0,1))
        if self.norm_obs:
            obs = (obs - obs.min())/(obs.max()-obs.min())
        if self.stand_obs:
            obs = (obs - obs.mean())/obs.std()
            
        return obs
    
    def save_PINN(self, episode = 0, id = 0):
        self.pinn_model = f"models/PINN_3D_wrap_Diffusion_{episode}_episodes_{id}_.pth"
        torch.save(self.pinn.state_dict(), self.pinn_model)
    
    def load_PINN(self, filename):
        self.pinn = PINN(layers = l2, device = device, sensor_coords = sensor_coords)
        self.pinn.load_state_dict(torch.load(filename))
        return
    
    def render(self):
        self.env.render()
        return
    
    def test_epsisode(self, model, #env = None, 
                      render = True, 
                      load_PINN = None,
                      ic_temp = None):
        
        tq = tqdm()
        
        #model = model.load(model_name)        
        if load_PINN is not None:
            self.pinn = PINN(layers = l2, device = device, sensor_coords = sensor_coords)
            self.pinn.load_state_dict(torch.load(load_PINN))
        else:
            print('No PINN model passed')
            return
        
        obs = self.reset(ic_temp = ic_temp)
        done = False
        reward = 0
        reward_history = []
        while not done:
            
            action = model.act(states = obs, independent = True)
            obs, r, done, _ = self.step(action)
            
            reward += r
            reward_history.append(r)
            
            if render:
                self.render()
                clear_output(wait=True)
            else:
                tq.update()
                print('action: ', action)
        
        return reward_history
    
    
