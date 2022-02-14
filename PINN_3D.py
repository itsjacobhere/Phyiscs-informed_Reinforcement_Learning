

import torch
from tqdm import tqdm
import numpy as np

class PINN(torch.nn.Module):
    '''
    3-Dimensional Physics-Informed Neural Network (PINN) - 
    Converges on solution for heat diffusion PDE in (X, Y, T) dimensions
    '''
    def __init__(self, layers = [10,10], device = None, sensor_coords = [[0,0]]):
        super(PINN, self).__init__() # inherit methods from torch
        
        # set to use GPU if available else cpu
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else: 
            self.device = device
            
        # set nn architecture:    
        
        # store hidden layer activation function
        self.hidden_activation = torch.nn.Tanh().to(device)
        
        # initialize nn architecture:
        self.input_layer = torch.nn.Linear(3,layers[0]).to(device)
        self.input_activation = torch.nn.Tanh().to(device)
        self.hidden = torch.nn.ModuleList([torch.nn.Linear(layers[i], layers[i+1]).to(device) for i in range(len(layers)-1)])
        self.output_layer = torch.nn.Linear(layers[-1],1).to(device)
        self.output_activation = torch.nn.Tanh().to(device)
        
        self.sensor_coords = np.append(np.flip(sensor_coords), 
                                       np.zeros(len(sensor_coords)).reshape(-1,1), axis = 1)
        
        # init weights and biases:
        
        torch.nn.init.xavier_normal_(self.input_layer.weight.data, gain=1.66)
        torch.nn.init.zeros_(self.input_layer.bias.data)
        
        for i in range(len(layers) - 1):
            
            # set weights from normal distribution 
            torch.nn.init.xavier_normal_(self.hidden[i].weight.data, gain=1.66)
            
            # init biases as zero
            torch.nn.init.zeros_(self.hidden[i].bias.data)
            
        torch.nn.init.xavier_normal_(self.output_layer.weight.data, gain=1.66)
        torch.nn.init.zeros_(self.output_layer.bias.data)
        
        
    def forward(self, x_in): 
        """Feed forward function through neural network."""
        # convert to tensor
        if torch.is_tensor(x_in) != True:         
            x_in = torch.from_numpy(x_in)
        
        # input layer
        x = self.input_layer(x_in)
        x = self.input_activation(x)
        
        # loop through hidden layers
        for i in range(len(self.hidden)):
            x = self.hidden_activation(self.hidden[i](x))
        
        x_out = self.output_layer(x)
        
        return x_out
        
    def MSE(self, y_pred, y_test):
        return torch.mean((y_pred - y_test)**2)
    
    def train_step(self, closure = True):
        '''Takes one train step, called from Train method for number of epochs.'''
        
        if closure:
            self.optimizer.zero_grad()
            
        # thermal diffisivity
        K = self.K
            
        # predict on initial condition w/ nn
        ic_pred = self.forward(self.ic_x)
        self.mse_ic = self.MSE(ic_pred, self.ic_u)
        
        # predict solution to boundary condition
        bc_pred = self.forward(self.x_bc) #[x,t]
        self.mse_u = self.MSE(bc_pred, self.u_bc)
        
        # predict u w/ network
        self.x.requires_grad = True 
        u_pred = self.forward(self.x)
        
        # differentiate using auto grad:
        
        # 1st deriv wrt X = [y,x,t]
        deriv1 = torch.autograd.grad(u_pred,
                                    self.x, #[y,x,t]
                                    torch.ones([self.x.shape[0], 1]).to(self.device),
                                    retain_graph = True,
                                    create_graph = True)[0]
        
        # 2nd deriv wrt X
        deriv2 = torch.autograd.grad(deriv1,
                                    self.x, 
                                    torch.ones(self.x.shape).to(self.device),
                                    create_graph = True)[0]
        
        du_dy, du_dx, du_dt = deriv1[:,[0]], deriv1[:,[1]], deriv1[:,[2]]
        d2u_dy2, d2u_dx2, d2u_dt2  = deriv2[:,[0]], deriv2[:,[1]], deriv2[:,[2]]
        
        # minimize f by incorporating into the loss
        f = du_dt - K * d2u_dx2 - K * d2u_dy2 + self.vel*du_dx + self.vel*du_dy - self.source # == 0
        
        self.mse_f = self.MSE(f, self.f_hat)
        self.loss = self.mse_u + self.mse_f + self.mse_ic
        
        if closure:
            self.loss.backward()
            
        return self.loss
    
    def train(self, x_bc, u_bc, x, ic_x, ic_u,
              learning_rate = 1e-4,
              epochs = int(1e4),
              LBFGS = True,
              K = 0.1,
              max_iter = 1,
              source = 0,
              sensor_values = [[0],[0],[0]],
              flow_velocity = 0,
              max_loss = np.inf,
              min_delta = np.inf):
        
        self.vel = flow_velocity
        
        x = np.append(x, self.sensor_coords, axis = 0) # add coords to end of x coords
        
        self.source = source
        self.K = K
        
        # boundary conditions
        self.x_bc = x_bc if torch.is_tensor(x_bc) else torch.from_numpy(x_bc).float().to(self.device)  
        self.u_bc = u_bc if torch.is_tensor(u_bc) else torch.from_numpy(u_bc).float().to(self.device)
        
        # initial conditions
        self.ic_x = ic_x if torch.is_tensor(ic_x) else torch.from_numpy(ic_x).float().to(self.device)  
        self.ic_u = ic_u if torch.is_tensor(ic_u) else torch.from_numpy(ic_u).float().to(self.device)  
        
        # f_hat(x) = 0
        self.x = x if torch.is_tensor(x) else torch.from_numpy(x).float().to(self.device)
        self.f_hat = torch.zeros(x.shape[0],1).to(self.device) # PDE is minimized to equal zero
        
        self.f_hat[-len(sensor_values):] = torch.tensor(sensor_values) # set last values equal to obs
        
        # store loss history
        self.loss_u = []
        self.loss_f = []
        loss_t = []
        
        if LBFGS == True:
            # same optimizer used in Raissi et al, quasi-Newton method
            self.optimizer = torch.optim.LBFGS(self.parameters(), 
                                  lr=0.1, 
                                  max_iter = max_iter, 
                                  max_eval = None, 
                                  tolerance_grad = 1e-05, 
                                  tolerance_change = 1e-09, 
                                  history_size = 100, 
                                  line_search_fn = 'strong_wolfe') 
            self.optimizer.step(self.train_step) # uses closure
        
        # declar adam opt
        optimizer = torch.optim.Adam(self.parameters(), 
                       lr= learning_rate,
                       betas=(0.9, 0.999), 
                       eps=1e-08, 
                       weight_decay=0, 
                       amsgrad=False)
        #tq = tqdm()
        
        iters = 0
        while iters <= epochs: #tqdm
            
            # perform train step
            loss = self.train_step(closure = False)
            loss_t.append(loss)
            
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # exit conditions
            del_loss = np.abs(loss_t[0].cpu().detach().numpy()-loss.cpu().detach().numpy())
            if loss.cpu().detach().numpy() < max_loss and del_loss > min_delta:break
            #if epochs > 10: tq.update()
            #if iters % 100 == 0: print(iters)
            iters +=1
            
        #print(iters)   
        return loss_t
    
    def predict(self, x_test, load_model = None, time = 5):
        
        # uses pretrained model
        if load_model is not None:
            self.load_state_dict(torch.load(load_model))
        
        if torch.is_tensor(x_test) != True: # convert to tensor send to device
            x_test = torch.from_numpy(x_test).float().to(self.device)
        
        # feedforwards input into nn
        u_pred = self.forward(x_test) 
        u_pred = u_pred.cpu().detach().numpy()
        u_pred = u_pred.reshape(100,256,time) 
        
        return u_pred
    
    def error(self, x_test, u_true):
        u_pred = self.forward(x_test)
        return (torch.linalg.norm((u_true-u_pred),2)/torch.linalg.norm(u_true,2)).item() # l2 error
