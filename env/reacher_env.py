import numpy as np
import gym
import time
import torch
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Reacher():
    def __init__(self):
        self.env = gym.make('FetchReach-v1').env
        self.action_space = gym.spaces.Box(shape=(7,),low=-0.1,high=0.1)
        self.observation_space = gym.spaces.Box(shape=(10,),low=-np.inf, high=np.inf)
        self.action_dim = self.action_space.shape[0]
        self.state_dim = self.observation_space.shape[0]
        self.action_range=[-0.2,0.2]
        self.target=np.array([]).reshape(0,3)
        self.q = [0.0, -1.0, 0.0, 2.0, 0.0, 0.5, 0.0]

    def step(self, action):
        self.q+=action
        qpos = {
            'robot0:shoulder_pan_joint': self.q[0],
            'robot0:shoulder_lift_joint': self.q[1],
            'robot0:upperarm_roll_joint': self.q[2],
            'robot0:elbow_flex_joint': self.q[3],
            'robot0:forearm_roll_joint':self.q[4],
            'robot0:wrist_flex_joint':self.q[5],
            'robot0:wrist_roll_joint':self.q[6]
        }
        for name, value in qpos.items():
            self.env.sim.data.set_joint_qpos(name, value)
        self.env.sim.forward()
        time.sleep(1 / 30)
        ob=self.env._get_obs()
        s=np.concatenate((ob['achieved_goal']-self.target,self.q),axis=0)
        d=np.linalg.norm(ob["achieved_goal"]-ob["desired_goal"])
        done=False
        r=-d-0.01*np.square(action).sum()
        return s,r,done,None

    def reset(self):
        self.env.reset()
        self.q = [0.0, -1.0, 0.0, 2.0, 0.0, 0.5, 0.0]
        qpos = {
            'robot0:shoulder_pan_joint': 0.0,
            'robot0:shoulder_lift_joint': -1.0,
            'robot0:upperarm_roll_joint': 0.0,
            'robot0:elbow_flex_joint': 2.0,
            'robot0:forearm_roll_joint': 0.0,
            'robot0:wrist_flex_joint': 0.5,
            'robot0:wrist_roll_joint': 0.0
        }
        for name, value in qpos.items():
            self.env.sim.data.set_joint_qpos(name, value)
        ob = self.env._get_obs()
        self.target=ob["desired_goal"]
        return np.concatenate((ob['achieved_goal']-self.target,self.q),axis=0)

    def render(self):
        self.env.render()
        time.sleep(1/30)

    def state_cost(self,state):
        #[8000,10]
        state=state.detach().cpu().numpy()
        dis=state[:,:3]
        # dis=state-self.env.target.copy()
        cost=np.sum(np.square(dis),axis=-1)
        cost = torch.from_numpy(cost).to(device).float()
        return cost

    @staticmethod
    def action_cost(action):
        return 0.01*(action**2).sum(dim=1)