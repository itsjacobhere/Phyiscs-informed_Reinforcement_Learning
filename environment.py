# fenics Finite Element
from fenics import *
from mshr import *
from dolfin import *

# typical libaries
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import Image
from IPython.display import set_matplotlib_formats
from IPython.display import clear_output
set_matplotlib_formats('png', 'pdf')
get_ipython().run_line_magic('matplotlib', 'inline')

from tqdm import tqdm

import gym
from gym import spaces

import wandb

class heat_diffusion(gym.Env):
    
    def __init__(self, dt = 5e-4, sensor_coords = [[0,0]],
                continuous = False,
                noisy_IC = False,
                noisy_source = False,
                norm_reward = False,
                scale_reward = False,
                verbose = False,
                wandb_name = None,
                project_name = 'Default_project',
                Cp = 0.07):
        super(heat_diffusion, self).__init__() # inherit from openAI gym
        
        self.continuous = continuous
        self.tol = 0.5
        self.penalty = 0
        self.cost_mult = 1
        self.switching_cost = 0.5
        self.streak_thresh = 10
        
        self.Cp = Cp
        
        self.noisy_IC = noisy_IC
        self.norm_reward = norm_reward
        self.verbose = verbose
        self.noisy_source = noisy_source
        self.scale_reward = scale_reward
        self.wandb_name = wandb_name
        self.ic_temp = None
        
        
        if wandb_name is not None: wandb.init(project = project_name, name = wandb_name)
        
        
        # increase or decrease the boundary temp
        self.action_space = spaces.Discrete(2)
        if continuous:
            self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]) )#, dtype=np.float32)
        self.action = np.nan
        self.info = {'energy':[np.nan],
                     'action':[None],
                     'reward':[],
                     'cost':[]}
        #self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
        
        # 4 continuous temp sensors at specified locations
        self.observation_space = spaces.Box(
              low = -np.inf, high = np.inf, shape=(len(sensor_coords), 1), dtype=np.float16)
        
        self.sensors = sensor_coords
        
        self.dt = dt
        
        meshsize = 100
        Wall = Rectangle( dolfin.Point(0.0, -1.0), dolfin.Point(1.0, 1.0) )
        block = Circle(Point(0.5,0), 0.05) #Rectangle( dolfin.Point(0.45, -0.05), dolfin.Point(0.55, 0.05) ) #
        domain = Wall - block  
        domain.set_subdomain(5, block)
        self.mesh = generate_mesh(domain, meshsize)
        
        self.tq = tqdm()
        self.step_tot = 0
        self.start_sim()
        
    def show_mesh(self):
        plot(self.mesh)
    
    def power_to_temp(self, power):
        return 40/(1+np.exp(-5*power)) - 20
    
    def start_sim(self):
        
        self.streak_counter = 0
        self.counter = 0
        dt = self.dt
        mesh = self.mesh
        
        self.V = FunctionSpace(mesh, 'CG', 1)
        
        self.t = 0
        self.step_num = 0
        
        gD_bottom = Expression('0', degree = 1) 
        gD_top = Expression('0', degree = 1) 
        gD_left = Expression('sin(x[1]) + 0', degree = 1) 
        gD_right = Expression('-sin(x[1]) + 0', degree = 1) 
        
        self.gN = Constant(0.0) # heat flux from person
        
        tol = 1e-6 #very small value 
        class bottom(SubDomain):
            def inside(self, x, on_boundary):
                return abs(x[1] + 1) < tol
        class top(SubDomain):
            def inside(self, x, on_boundary):
                return abs(x[1] - 1) < tol
        class  left(SubDomain):
            def inside(self, x, on_boundary):
                return abs(x[0]) < tol
        class right(SubDomain):
            def inside(self, x, on_boundary):
                return abs(x[0] - 1) < tol
        
        class Person(SubDomain):
            def inside(self, x, on_boundary):
                a = 0.5 # x center
                b = 0.0 # y center
                return abs( (x[0] - a)**2 + (x[1] - b)**2 - 0.05**2) < 0.001
            
        bottom = bottom()    
        top = top()
        left = left()
        right = right()
        
        person = Person()
        
        # Initialize mesh function for boundary domains and mark
        self.boundaries = MeshFunction("size_t", mesh , 1) #
        boundaries = self.boundaries
        boundaries.set_all(0) #
        
        
        bottom.mark(boundaries,1)
        top.mark(boundaries,2)
        left.mark(boundaries,3)
        right.mark(boundaries, 4)
        
        person.mark(boundaries,5)
        
        ds = Measure("ds", domain = mesh, subdomain_data = boundaries) #
        dS = Measure("dS", domain = mesh, subdomain_data = boundaries) 
        
        
        bc_bottom = DirichletBC (self.V , gD_bottom, boundaries, 1)
        bc_top = DirichletBC(self.V, gD_top, boundaries, 2)
        bc_left = DirichletBC(self.V, gD_left, boundaries, 3)
        bc_right = DirichletBC(self.V, gD_right, boundaries, 4)
        bc_person = DirichletBC(self.V, self.gN, boundaries, 5)
        self.bcs = [bc_bottom, bc_top, bc_left, bc_right, bc_person] #store boundaries in list
        
        
        
        # Define initial conditions
        self.u_D = Constant(20) 
        if self.noisy_IC: 
            if np.random.random() < 0.5:
                self.u_D = Constant(np.random.randint(-20,-10)) #Expression('0', degree=1)
            else: 
                self.u_D = Constant(np.random.randint(10,20)) #Expression('0', degree=1)
                
        if self.ic_temp is not None:
            self.u_D = Constant(self.ic_temp)
        self.u_n = interpolate(self.u_D, self.V)

        rho = Constant(0.8)
        Cp = Constant(self.Cp) #0.1, 0.07
        k = Constant(0.9) #0.98

        # Define variational problem
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        f = Constant(0)
        if self.noisy_source: 
            f = Constant(np.random.uniform(-3,3)) 
        
        
        F = k*u*v*dx + rho*Cp*dt*inner(grad(u), grad(v))*dx - (self.u_n + dt*f)*v*dx \
                    - dt*self.gN*v*ds(5) 
        
        a, self.L = lhs(F), rhs(F)
        
        self.A = assemble(a)
        
        self.u = Function(self.V)
    
    def norm(self,x):
            return 2/ (1+np.exp(-0.2*x) ) - 1
        
    def step(self, action):
        
        self.action = action if not self.continuous else self.power_to_temp(action[0])
        self.change_bcs(self.action)
        self.counter += 1
        self.step_tot +=1
        
        # Update current time
        self.t += self.dt
        self.u_D.t = self.t
        
        self.fenics_compute()
        
        self.step_num += 1
        
        obs = np.array(self.get_obs()).reshape(3,1)
        
        reward = self.get_reward()
        done = self.is_done()
        
        self.info['energy'].append(self.energy_flux)
        self.info['action'].append(action)
        self.info['reward'].append(self.get_reward())
        
        if self.verbose: self.tq.update()
        self.obs = obs
               
        return obs, reward, done, self.info
    
    def fenics_compute(self):
        
        b = assemble(self.L)
        #A = assemble(self.a)
        
        for bc in self.bcs:
            bc.apply(self.A,b)
        
        # Compute solution
        solve(self.A, self.u.vector(), b)

        n = FacetNormal(self.mesh)
        
        flux = (inner(grad(self.u_n('+')), n('+')))
        
        self.energy_flux = assemble(flux*dS(subdomain_id=5, subdomain_data=self.boundaries))
        
        # Update previous solution
        self.u_n.assign(self.u)
    
    def change_bcs(self, action):
        
        if not self.continuous:
            if action == 1:
                action = 20
            elif action == 0:
                action = -20
            
        gD_bottom = Constant(action) #Expression(str(action), degree = 1) 
        def bottom(x):
            return abs(x[1] + 1) < 1e-6
        bc_bottom = DirichletBC (self.V , gD_bottom, bottom )
        self.bcs[0] = bc_bottom
        
        gD_top = Constant(action) #Expression(str(action), degree = 1) 
        def top(x):
            return abs(x[1] - 1) < 1e-6
        bc_top = DirichletBC (self.V , gD_top, top )
        self.bcs[1] = bc_top
        
        return 
    
    def get_obs(self):
        return [self.u(p) for p in self.sensors]
    
    def get_reward(self):
        
        fail_penalty = 0
        
        if self.info['energy'][-1] <= self.tol:
            self.streak_counter += 1
        else:
            self.streak_counter = 0
            
        def sigmoid(x):
            return 2 / (1 + np.exp(-0.025*x+2)) -0.4
        
        self.bonus = sigmoid(self.streak_counter)
        
        if all(np.abs(self.info['energy'][-10:]) >= 10):
            fail_penalty = 10000
                
        if self.continuous:
            self.energy_cost = self.cost_mult * (np.abs(self.action)/20)
        elif self.action != self.info['action'][-1]:
            self.energy_cost = self.switching_cost
            
        r = -(self.energy_flux)**2 - self.energy_cost - self.penalty + self.bonus - fail_penalty
        
        if self.scale_reward: 
            r = r/10
        elif self.norm_reward: 
            r = self.norm(r)
        else:
            pass
        return r
    
    def is_done(self):
        done = False
        
        
        if all(np.abs(self.info['energy'][-10:]) >= 10):
            done = True
        
        elif self.counter >= 200: 
            done = True
            
            
        '''elif len(self.info['energy']) > 20 and all(np.abs(self.info['energy'][-20:]) < self.tol) :
            done = True
            self.bonus = 50'''
            
        if done and self.wandb_name is not None: 
            wandb.log({'episode_reward':np.sum(self.info['reward'][-self.counter:])},
                          step = self.step_tot)
            if self.verbose:
                print('Episode Reward: ', np.sum(self.info['reward'][-self.counter:]))
        return done
    
    def render(self):
        self.verbose = False
        plt.rcParams['figure.figsize'] = [15, 10]
        a = plot(self.u, vmin = -20, vmax = 20, cmap = 'viridis')
        plt.colorbar(a)
        plt.scatter(np.array(self.sensors)[:,0], np.array(self.sensors)[:,1], c = 'red')
        plt.title('time = {:.2f} hours at step = {}, taking action {}'.format(self.t, 
                                                                                self.step_num, 
                                                                                self.action))
        plt.show()
        
        return 
    
    def reset(self, ic_temp = None):
        
        self.info = {'energy':[np.nan],
                     'action':[None],
                     'reward':[],
                     'cost':[]}
        
        self.ic_temp = ic_temp
        #print('resetting')
        self.start_sim()
        obs = self.get_obs()
        self.counter = 0
        
        return np.array(obs).reshape(3,1)
    
    
    
    
    
    
    
    
    
#_________________________________________________________________________________________











from tqdm import tqdm
class Convection(gym.Env):
    
    def __init__(self, dt = 5e-4, sensor_coords = [[0,0]],
                continuous = False,
                noisy_IC = False,
                noisy_source = False,
                norm_reward = False,
                scale_reward = False,
                verbose = False,
                wandb_name = None,
                project_name = 'Default_project',
                Cp = 5,
                k = 5,
                theta = 5,
                Pe = 0.1):
        super(Convection, self).__init__() # inherit from openAI gym
        
        self.continuous = continuous
        self.tol = 0.5
        self.penalty = 0
        self.cost_mult = 100
        self.switching_cost = 0.5
        self.streak_thresh = 10
        
        self.noisy_IC = noisy_IC
        self.norm_reward = norm_reward
        self.verbose = verbose
        self.noisy_source = noisy_source
        self.scale_reward = scale_reward
        self.wandb_name = wandb_name
        self.ic_temp = None
        
        self.Cp = Cp
        self.k = k
        self.theta = theta
        self.Pe = Pe
        
        if wandb_name is not None: wandb.init(project = project_name, name = wandb_name)
        
        # increase or decrease the boundary temp
        self.action_space = spaces.MultiDiscrete([2,3])
        if self.continuous:
            self.action_space = spaces.Box(low=np.array([-1,0]), high = np.array([1,1]))
            
        self.action = np.nan
        self.info = {'energy':[np.nan],
                     'action':[np.array([None, None])],
                     'reward':[],
                     'cost':[]}
        #self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
        
        # 4 continuous temp sensors at specified locations
        self.observation_space = spaces.Box(
              low = -np.inf, high = np.inf, shape=(len(sensor_coords), 1), dtype=np.float16)
        
        self.sensors = sensor_coords
        
        self.dt = dt
        
        meshsize = 70
        Wall = Rectangle( dolfin.Point(0.0, -1.0), dolfin.Point(1.0, 1.0) )
        block = Circle(Point(0.5,0), 0.05) #Rectangle( dolfin.Point(0.45, -0.05), dolfin.Point(0.55, 0.05) ) #
        domain = Wall - block  
        domain.set_subdomain(5, block)
        self.mesh = generate_mesh(domain, meshsize)
        
        
        
        self.tq = tqdm()
        self.step_tot = 0
        self.start_sim()
        
    def show_mesh(self):
        plot(self.mesh)
    
    def power_to_temp(self, power):
        return 40/(1+np.exp(-5*power)) - 20
    
    def power_to_fan(self, power):
        return 1/(1.25+np.exp(-10*power + 5)) 
    
    def start_sim(self):
        
        self.counter = 0
        dt = self.dt
        mesh = self.mesh
        self.t = 0
        self.step_num = 0
        
        nu = 0.008
        
        # Define function spaces (P2-P1)
        self.V = VectorFunctionSpace(mesh, "Lagrange", 2)
        self.Q = FunctionSpace(mesh, "Lagrange", 1)
        
        # Define trial and test functions
        u = TrialFunction(self.V)
        p = TrialFunction(self.Q)

        v = TestFunction(self.V)
        q = TestFunction(self.Q)

        
        tol = 1e-6 #very small value 
        class bottom(SubDomain):
            def inside(self, x, on_boundary):
                return abs(x[1] + 1) < tol
        class top(SubDomain):
            def inside(self, x, on_boundary):
                return abs(x[1] - 1) < tol
        class  left(SubDomain):
            def inside(self, x, on_boundary):
                return abs(x[0]) < tol
        class right(SubDomain):
            def inside(self, x, on_boundary):
                return abs(x[0] - 1) < tol
        
        class Person(SubDomain):
            def inside(self, x, on_boundary):
                a = 0.5 # x center
                b = 0.0 # y center
                return abs( (x[0] - a)**2 + (x[1] - b)**2 - 0.05**2) < 0.0015
            
        #instantiate boundary 'object' classes
        bottom = bottom()    
        top = top()
        left = left()
        right = right()
        
        person = Person()
        
        # Initialize mesh function for boundary domains and mark
        self.boundaries = MeshFunction("size_t", mesh , mesh.topology().dim()-1,0) #
        boundaries = self.boundaries
        boundaries.set_all(0) #
        
        
        bottom.mark(boundaries,1)
        top.mark(boundaries,2)
        left.mark(boundaries,3)
        right.mark(boundaries, 4)
        
        person.mark(boundaries, 5)
        
        # Define boundaries
        inflow   = 'near(x[1], -1.0)'
        inflow_top = 'near(x[1], 1.0)'
        outflow  = 'on_boundary && (x[0] == 0 || x[0]==1) && x[1]>-0.25 && x[1]<0.25'#'near(x[1], 1.0)'
        walls    = 'near(x[0], 0) || near(x[0], 1.0)'
        cylinder = 'on_boundary && x[0]>0.45 && x[0]<0.55 && x[1]>0.0 && x[1] < 0.4 && x[1]>0.6 && x[1] < 1.0'

        # Define inflow profile
        inflow_profile = ('0','0.0*4.0*1.2*x[0]*(1.0 - x[0]) / pow(1.0, 2)')
        inflow_profile_top = ('0','-0.0*4.0*1.2*x[0]*(1.0 - x[0]) / pow(1.0, 2)')
        
        # Define boundary conditions
        bcu_inflow = DirichletBC(self.V, Expression(inflow_profile,degree=2), inflow)
        bcu_inflow_top = DirichletBC(self.V, Expression(inflow_profile_top,degree=2), inflow_top)
        bcu_walls = DirichletBC(self.V, Constant((0, 0)), walls)
        bcu_cylinder = DirichletBC(self.V, Constant((0, 0)), boundaries, 5)
        bcp_outflow = DirichletBC(self.Q, Constant(0), outflow)
        self.bcu = [bcu_inflow,bcu_inflow_top, bcu_walls, bcu_cylinder]
        self.bcp = [bcp_outflow]
        
        
        
        # Create functions
        u0 = Function(self.V)
        u1 = Function(self.V)
        p1 = Function(self.Q)

        
        # Define coefficients
        k1 = Constant(dt) #Constant(dt)
        f = Constant((0, 0))

        # Tentative velocity step
        F1 = (1/k1)*inner(u - u0, v)*dx \
                + inner(grad(u0)*u0, v)*dx \
                + nu*inner(grad(u), grad(v))*dx \
                - inner(f, v)*dx

        
        a1, self.L1 = lhs(F1), rhs(F1)
        
        # Pressure update
        a2 = inner(grad(p), grad(q))*dx
        self.L2 = -(1/k1)*div(u1)*q*dx

        # Velocity update
        a3 = inner(u, v)*dx
        self.L3 = inner(u1, v)*dx - k1*inner(grad(p1), v)*dx

        # Assemble matrices
        self.A1 = assemble(a1)
        self.A2 = assemble(a2)
        self.A3 = assemble(a3)

        # Use amg preconditioner if available
        self.prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

        # Use nonzero guesses  for CG with non-symmetric BC
        parameters['krylov_solver']['nonzero_initial_guess'] = True

        
        #------------------------------------------------------------
        self.W = FunctionSpace(mesh, "Lagrange", 1)
        
        gD_bottom = Expression('0', degree = 1) 
        gD_top = Expression('0', degree = 1) 
        gD_left = Expression('0', degree = 1) 
        gD_right = Expression('0', degree = 1) 
        
        self.gN = Constant(0.0) # heat flux from person
        
        bc_bottom = DirichletBC (self.W , gD_bottom, boundaries, 1)
        bc_top = DirichletBC(self.W, gD_top, boundaries, 2)
        bc_left = DirichletBC(self.W, gD_left, boundaries, 3)
        bc_right = DirichletBC(self.W, gD_right, boundaries, 4)
        bc_person = DirichletBC(self.W, self.gN, boundaries, 5)
        self.bcs = [bc_bottom,bc_top, bc_right, bc_person,bc_left]
        
        #----
        
        #  initial condition
        #ic = Constant(5.0) #Expression("5.0", degree=0)
        
        # Define initial conditions
        ic = Constant(20) 
        if self.noisy_IC: 
            if np.random.random() < 0.5:
                ic = Constant(np.random.randint(-20,-10)) #Expression('0', degree=1)
            else: 
                ic = Constant(np.random.randint(10,20)) #Expression('0', degree=1)
        
        if not self.noisy_IC:
            ic = Constant(20)
        
        if self.ic_temp is not None:
            ic = Constant(self.ic_temp)

            
        # Equation coefficients
        Pe = Constant(self.Pe) # Peclet number
        #velocity = Expression(("x[1]", "0"), degree=3) # convecting velocity
        local_shear_rate = Constant([0.1,0.1]) #Expression(("0.1", "0.1"), degree=3)  #local shear rate #**
        
        ds = Measure("ds", domain = mesh, subdomain_data = boundaries)
        dS = Measure("dS", domain = mesh, subdomain_data = boundaries) 
        
        # Define boundary measure on Neumann part of boundary
        dsTop = Measure("ds", subdomain_id=2, subdomain_data=boundaries)
        dsBottom = Measure("ds", subdomain_id=1, subdomain_data=boundaries)
        
        def separation_v(u):
            return local_shear_rate*(1.0 - u)
        
        # Define steady part of the equation
        def operator(u, v, velocity):
            return  1.0/Pe*inner(u.dx(1), v.dx(1))*dx  \
                    - dot(velocity, grad(u))*v*dx \
                    + dot(local_shear_rate, grad(u*(1.0-u)))*v*dx \
                    - u*(1.0-u)*v*dsTop + u*(1.0-u)*v*dsBottom \
                    + dot(local_shear_rate, grad(u*(1.0-u)))*v*dx  # 2nd convection term
        
        # Define trial and test function and solution at previous time-step
        uw = Function(self.W) # set as nonlinear term
        vw = TestFunction(self.W)
        u0w = Function(self.W)
        
        theta = Constant(self.theta)
        rho = Constant(1.25)
        Cp = Constant(self.Cp)
        k = Constant(self.k) #1.1
        
        # time discretized equation
        self.F = k*uw*vw*dx + rho*Cp*(1/dt)*inner(uw-u0w, vw)*dx \
                + theta*operator(uw, vw, u1) \
                + (1.0-theta)*operator(u0w, vw, u1) - dt*self.gN*vw*ds(5) 
        
        #a, self.L = lhs(self.F), rhs(self.F)
        #self.A = assemble(a)
        
        u0w.interpolate(ic)
        
        self.u = u
        self.u0 = u0
        self.u1 = u1
        self.p1 = p1
        
        self.u0w = u0w
        self.uw = uw
        self.vw = vw
        
    def norm(self,x):
            return 2/ (1+np.exp(-0.05*x) ) - 1 #0.2
        
    def step(self, action):
        
        self.action = action
        self.action[0] = action[0] if not self.continuous else self.power_to_temp(action[0])
        self.action[1] = action[1] if not self.continuous else self.power_to_fan(action[1])
        self.change_bcs(self.action)
        self.counter += 1
        self.step_tot +=1
        
        # Update current time
        self.t += self.dt
        
        unstable_pen = self.fenics_compute()
            
        self.step_num += 1
        
        obs = self.get_obs()
        reward = self.get_reward(unstable_pen)
        
        self.info['energy'].append(self.energy_flux)
        self.info['action'].append(action)
        self.info['reward'].append(reward)
        
        done = self.is_done()
        
        if self.verbose: self.tq.update()
        self.obs = obs
        
        return np.array(obs).reshape(3,1), reward, done, self.info
    
    def fenics_compute(self):
        
        # Compute convection solution
        try:
            solve(self.F==0, self.uw, self.bcs)#, tol = 1e-6)
            self.failed_conv = False
        except:
            unstable_pen = 20000
            self.failed_conv = True
            
        n = FacetNormal(self.mesh)
        
        flux = (inner(grad(self.u0w('+')), n('+')))#*dS(5)
        
        self.energy_flux = assemble(flux*dS(subdomain_id=5, subdomain_data=self.boundaries))
        
        # Update previous solution
        self.u0w.assign(self.uw) # update temp
        
        #---
        # Solve Navier-Stokes
        try:
            # Compute tentative velocity step
            b1 = assemble(self.L1)
            [bc.apply(self.A1, b1) for bc in self.bcu]
            solve(self.A1, self.u1.vector(), b1, "bicgstab", "default")

            # Pressure update
            b2 = assemble(self.L2)
            [bc.apply(self.A2, b2) for bc in self.bcp]
            [bc.apply(self.p1.vector()) for bc in self.bcp]
            solve(self.A2, self.p1.vector(), b2, "bicgstab", self.prec)

            # Velocity update
            b3 = assemble(self.L3)
            [bc.apply(self.A3, b3) for bc in self.bcu]
            solve(self.A3, self.u1.vector(), b3, "bicgstab", "default")

            # Move to next time step
            self.u0.assign(self.u1) # update velocity
            
            unstable_pen = 0
            self.failed_NS = False
        except:
            unstable_pen = 20000
            self.failed_NS = True
            
        return unstable_pen
    
    def change_bcs(self, action):
        
        if not self.continuous:
            if action[0] == 1:
                action[0] = 10

            elif action[0] == 0:
                action[0] = -10
        
        gD_bottom = Constant(str(action[0])) #Expression(str(action[0]), degree = 1) 
        def bottom(x):
            return abs(x[1] + 1) < 1e-6
        bc_bottom = DirichletBC(self.W , gD_bottom, bottom)
        self.bcs[0] = bc_bottom
        
        gD_top = Constant(str(action[0])) #Expression(str(action[0]), degree = 1) 
        def top(x):
            return abs(x[1] - 1) < 1e-6
        bc_top = DirichletBC(self.W , gD_top,top)
        self.bcs[1] = bc_top
        
        speed = action[1]
        if not self.continuous:
            if action[1] == 1:
                speed = 0.7
            elif action[1] == 0:
                speed = 0.3
            elif action[1] == 2:
                speed = 0
            
        # Define inflow profile
        inflow   = 'near(x[1], -1.0)'
        inflow_top = 'near(x[1], 1.0)'
        inflow_profile = ('0',f'{str(speed)}*4.0*1.2*x[0]*(1.0 - x[0]) / pow(1.0, 2)')
        inflow_profile_top = ('0',f'-{str(speed)}*4.0*1.2*x[0]*(1.0 - x[0]) / pow(1.0, 2)')
        
        bcu_inflow = DirichletBC(self.V, 
                                 Constant([0,speed]), #Expression(inflow_profile,degree=2), 
                                 inflow)
        bcu_inflow_top = DirichletBC(self.V, 
                                     Constant([0,-speed]), #Expression(inflow_profile_top,degree=2), 
                                     inflow_top)
        self.bcu[0], self.bcu[1] = bcu_inflow, bcu_inflow_top
        return 
    
    def get_obs(self):
        return [self.uw(p) for p in self.sensors]
    
    def get_reward(self, pen = 0):
        fail_penalty = 0
        if self.info['energy'][-1] <= self.tol:
            self.streak_counter += 1
        else:
            self.streak_counter = 0
            
        def sigmoid(x):
            return 2 / (1 + np.exp(-0.025*x+2)) -0.4
        
        self.bonus = 10*sigmoid(self.streak_counter)
        
        if all(np.abs(self.info['energy'][-10:]) >= 25):
            fail_penalty = 10000
        
        if self.continuous:
            self.energy_cost = self.cost_mult * (np.abs(self.action[0])/20)
            self.fan_cost =  self.cost_mult * (np.abs(self.action[1]))
        elif self.action[0] != self.info['action'][-1][0]:
            self.energy_cost = self.switching_cost
        
        r = -(self.energy_flux)**2 - self.energy_cost - self.fan_cost - self.penalty + self.bonus - fail_penalty - pen
        
        if self.scale_reward: 
            r = r/10
        elif self.norm_reward: 
            r = self.norm(r)
        else: 
            pass
        
        return r
    
    def is_done(self):
        done = False
        
        if self.failed_conv or self.failed_NS:
            done = True
            if self.failed_conv:
                print('failed conv')
            elif self.failed_NS:
                print('failed NS')
        
        if all(np.abs(self.info['energy'][-10:]) >= 25):
            done = True
            print('energy exceeded')
        
        elif self.counter >= 200: 
            done = True
            #print('episode over')
            
        if done and self.wandb_name is not None: 
            wandb.log({'episode_reward':np.sum(self.info['reward'][-self.counter:]),
                       'episode_len':self.counter},
                          step = self.step_tot)
            if self.verbose:
                print('Episode Reward: ', np.sum(self.info['reward'][-self.counter:]),
                      'Episode Length: ', self.counter)
                
        return done
    
    def render(self, mode = 'both', scale = 50, size = [7,10]):
        
        plt.rcParams['figure.figsize'] = size
        '''if mode == 'heat':
            plot(self.uw)#, vmin = -20, vmax = 20)
        elif mode == 'velocity':
            plot(self.u1, scale = 350)
        else:
            plot(self.u1, scale = 350)#, vmin = -20, vmax = 20)
            plot(self.uw)'''
        #plt.colorbar(a)
        
        a = plot(self.uw, cmap = 'inferno', vmin = -20, vmax = 20)
        plot(self.u1, scale = scale)
        plt.colorbar(a)
        
        plt.scatter(np.array(self.sensors)[:,0], np.array(self.sensors)[:,1], c = 'red')
        plt.title('time = {:.2f} hours at step = {}, taking action {}'.format(self.t, 
                                                                                self.step_num, 
                                                                                self.action))
        plt.show()
        
        return 
    
    def reset(self, ic_temp = None):
        self.info = {'energy':[np.nan],
                     'action':[np.array([None, None])],
                     'reward':[],
                     'cost':[]}
        
        self.ic_temp = ic_temp
        
        #print('resetting')
        self.start_sim()
        obs = self.get_obs()
        self.counter = 0
        
        return np.array(obs).reshape(3,1)