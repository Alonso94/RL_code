import numpy as np
import time
from tqdm import trange

import torch
import torch.nn as nn
from scipy.stats import truncnorm
from reacher_env import Reacher
import random
import matplotlib.pyplot as plt

# from model import ensemble

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def swish(x):
    return x*torch.sigmoid(x)

def truncated_norm(size,std):
    val=truncnorm.rvs(a=-2,b=2,size=size,scale=std)
    return torch.tensor(val,dtype=torch.float).to(device)

def get_w_b(ensemble_size,input_size,output_size):
    w=truncated_norm(size=(ensemble_size,input_size,output_size),
                     std=1.0/(2.0*np.sqrt(input_size)))
    w=nn.Parameter(w).to(device)
    b=torch.zeros(ensemble_size,1,output_size,dtype=torch.float)
    b = nn.Parameter(b).to(device)
    return w,b

def shuffle_rows(arr):
    idx=np.argsort(np.random.uniform(size=arr.shape),axis=-1)
    return arr[np.arange(arr.shape[0])[:,None],idx]

class ensemble(nn.Module):
    def __init__(self,ensemble_size,input_size,output_size):
        super().__init__()
        self.ensemble_size=ensemble_size
        self.input_size=input_size
        self.output_size=output_size

        self.layer0_w,self.layer0_b=get_w_b(ensemble_size,input_size,200)
        self.layer1_w, self.layer1_b = get_w_b(ensemble_size, 200, 200)
        self.layer2_w, self.layer2_b = get_w_b(ensemble_size, 200, 200)
        self.layer3_w, self.layer3_b = get_w_b(ensemble_size, 200, output_size)

        self.inputs_mu=nn.Parameter(torch.zeros(input_size),requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros(input_size), requires_grad=False)

        self.max_logvar=nn.Parameter(torch.ones(1,output_size,dtype=torch.float32)/2.0).to(device)
        self.min_logvar = nn.Parameter(-torch.ones(1, output_size , dtype=torch.float32) * 10.0).to(device)

    def compute_decays(self):
        layer0_d=0.00025*(self.layer0_w**2).sum()/2.0
        layer1_d = 0.0005 * (self.layer1_w ** 2).sum() / 2.0
        layer2_d = 0.0005 * (self.layer2_w ** 2).sum() / 2.0
        layer3_d = 0.00075 * (self.layer3_w ** 2).sum() / 2.0
        decays=layer0_d+layer1_d+layer2_d+layer3_d
        return decays

    def get_input_stats(self,data):
        mu=np.mean(data,axis=0,keepdims=True)
        sigma=np.std(data,axis=0,keepdims=True)
        sigma[sigma<1e-12]=1.0
        self.inputs_mu.data=torch.from_numpy(mu).to(device).float()
        self.inputs_sigma.data=torch.from_numpy(sigma).to(device).float()

    def forward(self, input):
        x=(input-self.inputs_mu)/self.inputs_sigma
        x=x.matmul(self.layer0_w)+self.layer0_b
        x=swish(x)
        x = x.matmul(self.layer1_w) + self.layer1_b
        x = swish(x)
        x = x.matmul(self.layer2_w) + self.layer2_b
        x = swish(x)
        x = x.matmul(self.layer3_w) + self.layer3_b
        mean=x[:,:,:self.output_size]
        logvar=x[:,:,:self.output_size]
        logvar=self.max_logvar-nn.functional.softplus(self.max_logvar-logvar)
        logvar=self.min_logvar+nn.functional.softplus(logvar-self.min_logvar)
        return mean,logvar

class MPC:
    def __init__(self,env):
        self.env=env
        self.action_dim=env.action_space.shape[0]
        self.state_dim=env.observation_space.shape[0]
        # MPC parameters
        self.horizon=25
        self.action_buffer=np.array([]).reshape(0,self.action_dim)
        self.previous_solution=np.tile(np.zeros(self.action_dim),[self.horizon])
        # Ensemble parameters
        self.E=5
        self.input_size=self.action_dim+self.state_dim
        self.output_size=self.state_dim
        self.model=ensemble(self.E,self.input_size,self.output_size).to(device)
        self.model.optim=torch.optim.Adam(self.model.parameters(),lr=0.001)
        self.n_train_iter=100
        self.epochs=5
        self.batch_size=32
        self.init_population=2000
        self.has_trained=False
        # CEM parameters
        self.solution_dim=self.horizon*self.action_dim
        self.population_size=400
        self.n_elites=40
        self.max_iter=5
        self.alpha=0.1
        self.var_min=0.001
        self.action_range=[-0.2,0.2]
        # np.ile repeat the value
        self.init_variance=np.tile(np.ones(self.action_dim)*0.25,[self.horizon])
        # propagation parameters
        self.n_particles=20
        self.train_in=np.array([]).reshape(0,self.action_dim+self.state_dim)
        self.train_out=np.array([]).reshape(0,self.state_dim)
        # for plotting
        self.x=0
        self.xx=[]
        self.returns=[]

    def shufle_rows(self,array):
        idx=np.argsort(np.random.uniform(size=array.shape),axis=-1)
        return array[np.arange(array.shape[0])[:,None],idx]

    def rollout(self, render=True, max_length=150):
        state = self.env.reset()
        next_state = state
        done = False
        ret=0
        times = []
        next_states, states, actions, rewards = [], [], [], []
        for t in range(max_length):
            self.x+=1
            start = time.time()
            action = self.act(state,t)
            end = time.time()
            times.append(start - end)
            states.append(state)
            actions.append(action)
            next_state, reward, done, _ = self.env.step(action)
            next_states.append(next_state)
            ret += reward
            if render:
                self.env.render()
            if done:
                break
        self.xx.append(self.x)
        self.returns.append(ret)
        plt.figure()
        plt.plot(self.xx, self.returns)
        plt.xlabel('Training step')
        plt.ylabel('Cumulative rewards')
        plt.show()
        print("Average action selection time = ", np.mean(times))
        return states, actions, next_states, ret

    def collect_data(self, n_rollouts=1):
        inputs, outputs = [], []
        for i in range(n_rollouts):
            states, actions, next_states, ret = self.rollout()
            input = np.concatenate([states, actions], axis=-1)
            inputs.append(input)
            outputs.append(np.array(next_states) - np.array(states))
        return inputs, outputs

    def train_the_model(self):
        if not self.has_trained:
            # prepare inputs and output
            D_inputs,D_outputs=self.collect_data()
            self.train_in=np.concatenate([self.train_in]+D_inputs,axis=0)
            self.train_out=np.concatenate([self.train_out]+D_outputs,axis=0)
        self.model.get_input_stats(self.train_in)
        # sample from the data
        # E datasets, one for each neural network in the ensemble
        idx=np.random.randint(self.train_in.shape[0],size=[self.E,self.train_in.shape[0]])
        epoch_range=trange(self.epochs)
        num_batches=int(np.ceil(idx.shape[-1]/self.batch_size))
        for _ in epoch_range:
            for b in range(num_batches):
                # take a batch
                b_indx=idx[:,b*self.batch_size:(b+1)*self.batch_size]
                # define the loss of the uncertainty and decay
                loss=0.01*(self.model.max_logvar.sum()-self.model.min_logvar.sum())
                loss+=self.model.compute_decays()
                # move data to GPU
                train_in=torch.from_numpy(self.train_in[b_indx]).to(device).float()
                train_out=torch.from_numpy(self.train_out[b_indx]).to(device).float()
                # compute the output of the model
                mean,logvar=self.model(train_in)
                inv_var=torch.exp(-logvar)
                # compute the losses (to a scalar)
                train_losses=((mean-train_out)**2)*inv_var+logvar
                train_losses=train_losses.mean(-1).mean(-1).sum()
                # compute the full loss
                loss+=train_losses
                # train the model
                self.model.optim.zero_grad()
                loss.backward()
                self.model.optim.step()
            idx=self.shufle_rows(idx)
            # just to get the loss
            train_in = torch.from_numpy(self.train_in[idx[:4000]]).to(device).float()
            train_out = torch.from_numpy(self.train_out[idx[:4000]]).to(device).float()
            mean, _ = self.model(train_in)
            train_losses = ((mean - train_out) ** 2).mean(-1).mean(-1)
        self.has_trained=True

    def run_the_whole_system(self,num_trials=100):
        if not self.has_trained:
            self.train_the_model()
        for i in range(num_trials):
            states, actions, next_states, ret=self.rollout()
            input = np.concatenate([states, actions], axis=-1)
            output=np.array(next_states) - np.array(states)
            self.train_in = np.concatenate([self.train_in, input], axis=0)
            self.train_out = np.concatenate([self.train_out, output], axis=0)
            self.train_the_model()


    def act(self,state,t):
        if not self.has_trained:
            #random if not trained
            return np.random.uniform(low=self.action_range[0],high=self.action_range[1],size=self.action_dim)
        if self.action_buffer.shape[0]>0:
            # execute the first action
            action=self.action_buffer[0]
            self.action_buffer=self.action_buffer[1:]
            return action
        self.current_state=state
        # obtain a solution using CEM
        solution=self.obtain_solution(self.previous_solution)
        # action_buffer = first action from solution
        # previous solution = [rest of actions, zero action]
        all_exept_first=np.copy(solution)[self.action_dim:]
        self.previous_solution=np.concatenate([all_exept_first,np.zeros(self.action_dim)])
        self.action_buffer=solution[:self.action_dim].reshape(-1,self.action_dim)
        return self.act(state,t)

    def obtain_solution(self,previous_solution):
        # CEM
        t=0
        mean=previous_solution
        var=self.init_variance
        truncated_normal=truncnorm(-2,2,loc=np.zeros_like(mean),scale=np.ones_like(var))
        while t<self.max_iter and np.max(var)>self.var_min:
            samples=truncated_normal.rvs(size=[self.population_size,self.solution_dim])
            samples=samples*np.sqrt(var)+mean
            samples=samples.astype(np.float32)
            costs=self.propagate_to_find_costs(samples)
            elites=samples[np.argsort(costs)][:self.n_elites]
            new_mean=np.mean(elites,axis=0)
            new_var=np.var(elites,axis=0)
            mean=self.alpha*mean+(1-self.alpha)*new_mean
            var=self.alpha*var+(1-self.alpha)*new_var
            t+=1
        return mean

    @torch.no_grad()
    def propagate_to_find_costs(self,action_sequences):
        # prepare the action sequences
        n_action_sequences=action_sequences.shape[0]
        action_sequences=torch.from_numpy(action_sequences).to(device).float()
        # reshape -> [400,25,1]
        action_sequences=action_sequences.view(-1,self.horizon,self.action_dim)
        # transpose -> [25,400,1]
        action_sequences=action_sequences.transpose(0,1)
        # expand -> [25,400,1,1]
        action_sequences=action_sequences[:,:,None]
        # tile (make copies for particles) -> [25,400,20,1]
        action_sequences=action_sequences.expand(-1,-1,self.n_particles,-1)
        # reshape -> [25,8000,7]
        action_sequences=action_sequences.contiguous().view(self.horizon,-1,self.action_dim)
        # prepare the current state
        current_state=torch.from_numpy(self.current_state).to(device).float()
        current_state=current_state[None]
        current_state=current_state.expand(n_action_sequences*self.n_particles,-1)
        costs=torch.zeros(n_action_sequences,self.n_particles,device=device)
        for t in range(self.horizon):
            current_action=action_sequences[t]
            # print(current_action.shape)
            # print(current_state.shape)
            predicted_next_state=self.predict(current_state,current_action)
            cost=self.state_cost(predicted_next_state)+self.action_cost(current_action)
            cost=cost.view(-1,self.n_particles)
            costs+=cost
            current_state=predicted_next_state
        # NaN -> hgih cost
        costs[costs!=costs]=1e6
        return costs.mean(1).detach().cpu().numpy()

    def predict(self,state,action):
        input_state=state
        dim=state.shape[-1]
        # format the state [8000,10] -> [400,5,4,10]
        state=state.view(-1,self.E,self.n_particles//self.E,dim)
        # transpose [5,400,4,10]
        state=state.transpose(0,1)
        # reshape [5,1600,10]
        state=state.contiguous().view(self.E,-1,dim)

        # format the action [8000,7] -> [400,5,4,7]
        dim=action.shape[-1]
        action = action.view(-1, self.E, self.n_particles // self.E, dim)
        # transpose [5,400,40,7]
        action = action.transpose(0, 1)
        # reshape [5,1600,7]
        action = action.contiguous().view(self.E, -1, dim)

        #get the output of the model
        inputs=torch.cat((state,action),dim=-1)
        # mean [5,1600,10]
        mean,var=self.model(inputs)

        predicted_state=mean+torch.randn_like(mean,device=device)*var.sqrt()
        # [5,1600,10]
        #remove additional dimensions
        dim=predicted_state.shape[-1]
        # [5,400,4,10]
        predicted_state=predicted_state.view(self.E,-1,self.n_particles//self.E,dim)
        # [400,5,4,10]
        predicted_state=predicted_state.transpose(0,1)
        # [8000,10]
        predicted_state=predicted_state.contiguous().view(-1,dim)

        return input_state+predicted_state

    def state_cost(self,state):
        state=state.detach().cpu().numpy().transpose()
        dis=state[:,:3]-self.env.target
        cost=np.sum(np.square(dis),axis=-1)
        cost=torch.from_numpy(cost).to(device).float()
        return cost

    @staticmethod
    def action_cost(action):
        return 0.01*(action**2).sum(dim=1)

def set_global_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_global_seeds(0)
env=Reacher()
mpc=MPC(env)
mpc.run_the_whole_system()
