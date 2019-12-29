import numpy as np
import gym
import time
import torch
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CartPole():
    def __init__(self):
        self.env = gym.make('CartPole-v1').env
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(shape=(5,),low=-1, high=1)
        self.action_dim = 1
        self.action_range=[-1,1]
        self.state_dim = self.observation_space.shape[0]
        self.target_ee_pose=torch.tensor([0.0,1.0]).to(device).float()


    def step(self, action):
        if action[0]>0.0:
            a=1
        else:
            a=0
        s,r,done,_=self.env.step(a)
        s_new=np.array([s[0],s[1],np.sin(s[2]),np.cos(s[2]),s[3]])
        reward=np.exp(-np.sum(np.square([s[0]-s_new[2],s_new[3]]-np.array([0.0,1.0]))))
        reward-=0.01*np.sum(np.square(action))
        done=False
        return s_new,reward,done,None

    def reset(self):
        s=self.env.reset()
        self.env.state = self.env.np_random.normal(0,0.05, size=(4,))
        s_new=np.array([s[0],s[1],np.sin(s[2]),np.cos(s[2]),s[3]])
        return s_new

    def render(self):
        self.env.render()
        # time.sleep(1/30)

    def state_cost(self,state):
        #[8000,10]
        # end effector position (x=x-sin(theta),y=cos(theta))
        theta=torch.atan2(state[:,2],state[:,3])
        ee_pose=torch.stack([state[:,0]-state[:,2],state[:,3]]).transpose(0,1)
        cost= ((ee_pose-self.target_ee_pose)**2).sum(dim=-1)
        cost -torch.exp(-cost)
        return cost

    @staticmethod
    def action_cost(action):
        return 0.01*(action**2).sum(dim=1)
CartPole()