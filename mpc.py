import numpy as np
import time
from tqdm import trange

import torch
import torch.nn as nn
from scipy.stats import truncnorm
from env.reacher_env import Reacher
from env.env_sim_control import rozum_sim
import random
import matplotlib.pyplot as plt
# import tensorflow as tf
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
    w=nn.Parameter(w)
    b=torch.zeros(ensemble_size,1,output_size,dtype=torch.float32)
    b = nn.Parameter(b)
    return w,b

class ensemble(nn.Module):
    def __init__(self,ensemble_size,input_size,output_size):
        super().__init__()
        self.ensemble_size=ensemble_size
        self.input_size=input_size
        self.output_size=output_size

        self.layer0_w,self.layer0_b=get_w_b(ensemble_size,input_size,500)
        self.layer1_w, self.layer1_b = get_w_b(ensemble_size, 500, 500)
        self.layer2_w, self.layer2_b = get_w_b(ensemble_size, 500, 500)
        self.layer3_w, self.layer3_b = get_w_b(ensemble_size, 500, 2*output_size)

        self.inputs_mu=nn.Parameter(torch.zeros(1,input_size),requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros(1,input_size), requires_grad=False)

        self.max_logvar=nn.Parameter(torch.ones(1,output_size,dtype=torch.float32)/2.0)
        self.min_logvar = nn.Parameter(-torch.ones(1, output_size , dtype=torch.float32) * 10.0)

    def compute_decays(self):
        layer0_d=0.0001*(self.layer0_w**2).sum()/2.0
        layer1_d = 0.00025 * (self.layer1_w ** 2).sum() / 2.0
        layer2_d = 0.00025 * (self.layer2_w ** 2).sum() / 2.0
        layer3_d = 0.0005 * (self.layer3_w ** 2).sum() / 2.0
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
        logvar=x[:,:,self.output_size:]
        logvar=self.max_logvar-nn.functional.softplus(self.max_logvar-logvar)
        logvar=self.min_logvar+nn.functional.softplus(logvar-self.min_logvar)
        return mean,logvar

class MPC:
    def __init__(self,env,load=False):
        self.env=env
        self.action_dim=env.action_dim
        self.state_dim=env.state_dim
        self.action_range=env.action_range
        self.action_lb=self.env.action_space.low
        self.action_ub=self.env.action_space.high
        # MPC parameters
        self.horizon=5
        self.action_buffer=np.array([]).reshape(0,self.action_dim)
        self.previous_solution=np.tile((self.action_lb+self.action_ub)/2.0,[self.horizon])
        # Ensemble parameters
        self.E=10
        self.input_size=self.action_dim+self.state_dim
        self.output_size=self.state_dim
        self.model=ensemble(self.E,self.input_size,self.output_size).to(device)
        self.has_trained = False
        if load:
            self.model.load_state_dict(torch.load("/home/ali/RL_code/models/ensemble_e10_u6_s9+12.pth",map_location=device))
            self.model.eval()
            self.has_trained=True
        self.model.optim=torch.optim.Adam(self.model.parameters(),lr=0.001)
        self.init_rollouts = 1
        self.n_train_iter=50
        self.rol_per_iter=1
        self.epochs=5
        self.batch_size=64
        # CEM parameters
        self.solution_dim=self.horizon*self.action_dim
        self.opt_lb=np.tile(self.action_lb,[self.horizon])
        self.opt_ub = np.tile(self.action_ub, [self.horizon])
        self.population_size=400
        self.n_elites=40
        self.max_iter=5
        self.alpha=0.1
        self.var_min=0.001
        # np.ile repeat the value
        self.init_variance=np.tile(np.square(self.action_ub-self.action_lb)/16.0,[self.horizon])
        # propagation parameters
        self.n_particles=20
        self.train_in=np.array([]).reshape(0,self.action_dim+self.state_dim)
        self.train_out=np.array([]).reshape(0,self.state_dim)
        # for plotting
        self.x=0
        self.count=0
        self.count_done=0
        self.xx=[]
        self.returns=[]
        self.picked=[]
        self.ds=[]
        self.dist=[]

    def shufle_rows(self,array):
        idx=np.argsort(np.random.uniform(size=array.shape),axis=-1)
        return array[np.arange(array.shape[0])[:,None],idx]

    def rollout(self, render=False, plot=True, max_length=50,evaluate=False):
        state = self.env.reset()
        ret=0
        times = []
        next_states, states, actions, rewards = [], [], [], []
        train_range=trange(max_length)
        for t in train_range:
            start = time.time()
            action = self.act(state)
            end = time.time()
            times.append(end-start)
            if self.has_trained:
                self.x+=1
            states.append(state)
            actions.append(action)
            state, reward, done, _ = self.env.step(action)
            next_states.append(state)
            # state=next_state.copy()
            ret += reward
            if render:
                self.env.render()
            if done:
                break
        if self.has_trained and plot:
            self.xx.append(self.x)
            if ret>0:
                self.count+=1
                if ret>70:
                    self.count_done+=1
            self.picked.append(self.count)
            self.ds.append(self.count_done)
            self.returns.append(ret)
            plt.figure()
            plt.plot(self.xx, self.returns)
            plt.xlabel('Training step')
            plt.ylabel('Cumulative rewards')
            plt.show()
            print("\nAverage action selection time = ", np.mean(times))
        print("episode length = ",len(states))
        return states, actions, next_states, ret

    def collect_data(self, n_rollouts=1):
        inputs, outputs = [], []
        for i in range(n_rollouts):
            states, actions, next_states, ret = self.rollout()
            input_ = np.concatenate([np.array(states), np.array(actions)], axis=-1)
            inputs.append(input_)
            outputs.append(np.array(next_states) - np.array(states))
        return inputs, outputs

    def run_the_whole_system(self,num_trials=60):
        if not self.has_trained:
            # prepare inputs and output
            D_inputs, D_outputs = self.collect_data(n_rollouts=self.init_rollouts)
            self.train_in = np.concatenate([self.train_in] + D_inputs, axis=0)
            self.train_out = np.concatenate([self.train_out] + D_outputs, axis=0)
            self.train_the_model()
        for i in range(num_trials):
            print("start training ...")
            D_inputs, D_outputs = self.collect_data(n_rollouts=self.rol_per_iter)
            self.train_in = np.concatenate([self.train_in] + D_inputs, axis=0)
            self.train_out = np.concatenate([self.train_out] + D_outputs, axis=0)
            self.train_the_model()
        self.env.out.release()
        torch.save(self.model.state_dict(),"/home/ali/RL_code/models/ensemble_e10_u6_s9+12.pth")
        print("model saved!")
        plt.figure()
        plt.plot(self.xx, self.returns)
        plt.xlabel('Training step')
        plt.ylabel('Cumulative rewards')
        plt.show()
        plt.savefig("/home/ali/RL_code/images/rewards.png")
        plt.figure()
        plt.plot(self.xx, self.count)
        plt.xlabel('Training step')
        plt.ylabel('Cube picked')
        plt.show()
        plt.savefig("/home/ali/RL_code/images/pick.png")
        plt.show()
        plt.figure()
        plt.plot(self.xx, self.ds)
        plt.xlabel('Training step')
        plt.ylabel('done count')
        plt.show()
        plt.savefig("/home/ali/RL_code/images/done.png")

    def evaluate(self,max_length=50,render=False):
        state = self.env.reset()
        x=0
        train_range = trange(max_length)
        xx=[]
        yy=[]
        for t in train_range:
            action = self.act(state)
            x += 1
            state, reward, done, _ = self.env.step(action)
            d = np.linalg.norm(state[:3] - self.env.init_pose_cube)
            xx.append(x)
            yy.append(d)
            if render:
                self.env.render()
            if done:
                break
        plt.figure()
        plt.plot(xx, yy)
        plt.xlabel('time')
        plt.ylabel('distance')
        plt.savefig("step_response.png")
        plt.show()


    def train_the_model(self):
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
                # print(logvar)
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
        train_in = torch.from_numpy(self.train_in[idx[:5000]]).to(device).float()
        train_out = torch.from_numpy(self.train_out[idx[:5000]]).to(device).float()
        mean, _ = self.model(train_in)
        train_losses = ((mean - train_out) ** 2).mean(-1).mean(-1)
        print(train_losses.detach().cpu().numpy())
        self.has_trained=True

    def act(self,state):
        if not self.has_trained:
            #random if not trained
            return np.random.uniform(low=self.action_lb,high=self.action_ub,size=self.action_dim)
        if self.action_buffer.shape[0]>0:
        #     # execute the first action
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
        # solution=solutions[:self.action_dim]
        return self.act(state)

    def obtain_solution(self,previous_solution):
        # CEM
        t=0
        mean=previous_solution.copy()
        var=self.init_variance.copy()
        truncated_normal=truncnorm(a=self.action_range[0],b=self.action_range[1],loc=np.zeros_like(mean),scale=np.ones_like(var))
        # truncated_normal=truncnorm(-2,2,loc=np.zeros_like(mean),scale=np.ones_like(var))
        while (t<self.max_iter) and np.max(var)>self.var_min:
            lb=mean-self.opt_lb
            ub=self.opt_ub-mean
            cd_var=np.minimum(np.minimum(np.square(lb/2),np.square(ub/2)),var)

            samples=truncated_normal.rvs(size=[self.population_size,self.solution_dim])
            samples=samples*np.sqrt(cd_var)+mean
            samples=samples.astype(np.float32)

            costs=self.propagate_to_find_costs(samples)

            # arg sort from smaller to larger
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
            predicted_next_state=self.predict(current_state,current_action)
            state_cost=self.env.state_cost(predicted_next_state)
            cost=state_cost#+self.env.action_cost(current_action)
            cost=cost.view(-1,self.n_particles)
            costs+=cost
            current_state=predicted_next_state
        # print(costs)
        # x=input()
        # NaN -> high cost
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
        mean,logvar=self.model(inputs)
        var=torch.exp(logvar)
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

def set_global_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # tf.set_random_seed(seed)

set_global_seeds(0)
# env=Reacher()
# env=CartPole()
env=rozum_sim()
mpc=MPC(env,load=True)
# mpc.run_the_whole_system(num_trials=1)
mpc.evaluate()