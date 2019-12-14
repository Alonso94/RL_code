import numpy as np
import scipy.stats as stats
from scipy.io import savemat
import os
from tqdm import trange
import torch

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def shuffle_rows(arr):
    idx=np.argsort(np.random.uniform(size=arr.shape),axis=-1)
    return arr[np.arange(arr.shape[0])[:,None],idx]

class CEM:
    def __init__(self,problem_space,max_iter,population_size,num_elite,cost_function):
        self.problem_space=problem_space
        self.max_iter=max_iter
        self.population_size=population_size
        self.num_elite=num_elite
        self.cost_function=cost_function
        self.epsilon=0.001
        self.alpha=0.25
        self.upper_bound=None
        self.lower_bound=None

    def obtain_solution(self,init_mean,init_var):
        mean,var,t=init_mean,init_var,0
        # create a truncated norm
        X=stats.truncnorm(-2,2,loc=np.zeros_like(mean),sclae=np.ones_like(var))
        while (t<self.max_iter) and np.max(var)>self.epsilon:
            lb_dist=mean-self.lower_bound
            ub_dist=self.upper_bound-mean
            lb1=np.square(lb_dist/2)
            ub1=np.square(ub_dist/2)
            constrained_var=np.minimum(np.minimum(lb1,ub1),var)
            # get random variables from the distribution
            samples=X.rvs(size=[self.population_size,self.problem_space])*np.sqrt(constrained_var)+mean
            costs=self.cost_function(samples)
            elites=samples[np.argsort(costs)][:self.num_elite]
            new_mean=np.mean(elites,axis=0)
            new_var=np.var(elites,axis=0)
            mean=self.alpha*mean+(1-self.alpha)*new_mean
            var=self.alpha*var+(1-self.alpha)*new_var
            t+=1
        return mean

    def reset(self):
        pass

class MPC:
    def __init__(self,env,model):
        self.env=env
        self.dO=env.observation_space.shape[0]
        self.dU=env.action_space.shape[0]
        self.ac_lb,self.ac_ub=-1,1
        self.horizon=25
        self.num_particles=20

        # array of targets (lean to map obs->targ_proc(obs,next_obs))
        self.targ_proc=lambda obs, next_obs: next_obs
        # preprocess ovservation before sending them to the model
        self.obs_preproc=lambda obs: obs
        #action buffer
        self.ac_buf = np.array([]).reshape(0, self.dU)
        # previous solution (initialized to the mean)
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.horizon])
        # init variance
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.horizon])
        # training inputs (initialize shape)
        self.train_in = np.array([]).reshape(0, self.dU + self.obs_preproc(np.zeros([1, self.dO])).shape[-1])
        # training targets (initialize shape)
        self.train_targs = np.array([]).reshape(
            0, self.targ_proc(np.zeros([1, self.dO]), np.zeros([1, self.dO])).shape[-1]
        )
        self.model=model
        self.epochs=50
        self.batch_size=32
        self.trained=False
        self.optimizer=CEM(problem_space=self.horizon*self.dU,
                           population_size=400,
                           num_elite=40,
                           max_iter=5,
                           cost_function=self._compute_cost)

        self.update_fns=[self.update_goal]

    def update_goal(self):
        self.goal=self.env.goal

    def train(self,obs_traj,acts_traj,rews_traj):
        self.trained=True
        new_train_in=[]
        new_train_targs=[]
        for obs,acts in zip(obs_traj,acts_traj):
            new_obs=self.obs_preproc([obs[:-1]])
            new_train_in.append(np.concatenate([new_obs,acts],axis=-1))
            new_targs=self.targ_proc(obs[-1],obs[1:])
            new_train_targs.append(new_targs)
        self.train_in=np.concatenate([self.train_in]+new_train_in,axis=0)
        self.train_targs=np.concatenate([self.train_targs]+new_train_targs,axis=0)

        self.model.fit_input_stats(self.train_in)
        idxs=np.random.randint(self.train_in.shape[0],size=[self.model.num_nets,self.train_in.shape[0]])

        epoch_range=trange(self.epochs,unit="epoch(s)",desc="Model training")
        num_batches=int(np.ceil(idxs.shape[-1])/self.batch_size)

        for _ in epoch_range:
            for i in range(num_batches):
                batch_idxs=idxs[:,i*self.batch_size:(i+1)*self.batch_size]
                loss=0.01*(self.model.max_logvar.sum()-self.model.min_log_var.sum())
                loss+=self.model.compute_decays()

                train_in=torch.from_numpy(self.train_in[batch_idxs]).to(device).float()
                train_targ=torch.from_numpy(self.train_targs[batch_idxs]).to(device).float()

                mean,logvar=self.model(train_in)
                inv_var=torch.exp(-logvar)

                train_losses=((mean-train_targ)**2)*inv_var+logvar
                # take mean over the last 2 dimensions
                train_losses=train_losses.mean(-1).mean(-1).sum
                loss+=train_losses
                self.model.optim.zero_grad()
                loss.backward()
                self.model.optim.step()

            idxs=shuffle_rows(idxs)
            val_in = torch.from_numpy(self.train_in[idxs[:5000]]).to(device).float()
            val_targ = torch.from_numpy(self.train_targs[idxs[:5000]]).to(device).float()
            mean,_=self.model(val_in)
            mse_losses=((mean-val_targ)**2).mean(-1).mean(-1)

    def reset(self):
        self.prev_sol=np.tile((self.ac_lb+self.ac_ub)/2,[self.horizon])
        self.optimizer.reset()
        for update_fn in self.update_fns:
            update_fn()

    def act(self,obs,t):
        if not self.trained:
            return np.random.uniform(self.ac_lb,self.ac_ub,self.dU)
        if self.ac_buf.shape[0]>0:
            action=self.ac_buf[0]
            self.ac_buf=self.ac_buf[1:]
            return action
        self.curr_obs=obs
        sol=self.optimizer.obtain_solution(self.prev_sol,self.init_var)
        self.prev_sol=np.concatenate([np.copy(sol)[self.dU],np.zeros(self.dU)])
        self.ac_buf=sol[:self.dU].reshape(-1,self.dU)
        return self.act(obs,t)

    @torch.no_grad()
    def _compile_cost(self,ac_seq):
        nopt=ac_seq.shape[0]
        # size (400,25) (population size, solution dimension)
        ac_seq=torch.from_numpy(ac_seq).to(device).float()
        # (400,25,1)
        ac_seq=ac_seq.view(-1,self.horizon,self.dU)
        # (25,400,1)
        transposed=ac_seq.transpose(0,1)
        # (25,400,1,1)
        expanded=transposed[:,:,None]
        # (25,400,20,1)
        tiled=expanded.expand(-1,-1,self.num_particles,-1)
        # (25,8000,1)
        ac_seq=tiled.contiguous().view(self.horizon,-1,self.dU)

        # expand current observation
        curr_obs=torch.from_numpy(self.curr_obs).to(device).float()
        curr_obs=curr_obs[None]
        curr_obs=curr_obs.expand(nopt*self.num_particles,-1)

        costs=torch.zeros(nopt,self.num_particles,device=device)

        for t in range(self.horizon):
            curr_act=ac_seq[t]
            next_obs=self.predict_next_obs(curr_obs,curr_act)
            cost=self.obs_cost_fn(next_obs)+self.act_cost_fn(curr_act)
            cost=cost.view(-1,self.num_particles)
            costs+=cost
            curr_obs=self.obs_preproc(next_obs)

        costs[costs!=costs]=1e6

        return costs.mean(1).detach().cpu().numpy()

