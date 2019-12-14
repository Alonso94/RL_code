import os
import sys
import time

from collections import OrderedDict

import numpy as np
import tensorflow as tf
import gym

gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.6,allow_growth=True)
config=tf.ConfigProto(gpu_options=gpu_options,
                      log_device_placement=True,
                      allow_soft_placement=True,
                      inter_op_parallelism_threads=1,
                      intra_op_parallelism_threads=1)

class Agent:
    def __init__(self,env,noise_std):
        self.env=env
        self.noise_std=noise_std

    def sample(self,horizon,policy):
        ts,rewards=[],[]
        obs,acts,reward_sum,done=[self.env.reset()],[],0,False

        policy.reset()
        for t in range(horizon):
            action=policy.act[obs[t],t]
            acts.append(action)
            if self.noise_std is None:
                observation,reward,done,infor=self.env.step(acts[t])
            else:
                action=acts[t]+np.random.normal(loc=0,scale=self.noise_std,size=self.env.action_space.shape[0])
                #action=np.clip(action,*self.env.action_range)
                action=np.minimum(np.maximum(action,self.env.action_space.low),self.env.action_space.high)
                observation,reward,done,info=self.env.step(action)
            obs.append(observation)
            reward_sum+=reward
            rewards.append(reward)
            if done:
                break
        print("Rollout length = ",len(acts))
        return {"obs":np.array(obs),
                "acts":np.array(acts),
                "ret":reward_sum,
                "rews":np.array(rewards)}


class model_based_exp:
    def __init__(self,env,horizon,stochastic,policy):
        self.env=env
        self.horizon=horizon
        self.policy = policy
        self.stochastic=stochastic
        noise_std=None
        self.agent=Agent(env,noise_std)

        self.n_train_iters=100
        self.n_rollout_per_iter=100
        self.n_init_rollouts=1

    def run(self):
        # traj_obs,traj_acts,traj_rets,traj_rews=[],[],[],[]
        samples=[]
        for i in range(self.n_init_rollouts):
            samples.append(self.agent.sample(self.horizon,self.policy))
            # traj_obs.append(samples[-1]["obs"])
            # traj_acts.append(samples[-1]["acts"])
            # traj_rews.append(samples[-1]["rews"])

        if self.n_init_rollouts>0:
            self.policy.train(
                [sample["obs"] for sample in samples],
                [sample["acts"] for sample in samples],
                [sample["rews"] for sample in samples]
            )

        print("Start_training")
        for i in range(self.n_train_iters):
            for j in range(self.n_rollout_per_iter):
                samples.append(self.agent.sample(self.horizon, self.policy))
            # traj_obs.extend([sample["obs"] for sample in samples])
            # traj_obs.extend([sample["acts"] for sample in samples])
            # traj_obs.extend([sample["rews"] for sample in samples])
            self.policy.train(
                [sample["obs"] for sample in samples],
                [sample["acts"] for sample in samples],
                [sample["rews"] for sample in samples]
            )