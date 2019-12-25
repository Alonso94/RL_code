import time
import numpy as np

def rollout(env,policy,render=False,max_length=100,random=False):
    state=env.reset()
    next_state=state
    done=False
    times=[]
    next_states,states,actions,rewards=[],[],[],[]
    while not done or len(states)<max_length:
        start=time.time()
        action=policy(state)
        end=time.time()
        times.append(start-end)

        state.append(state)
        actions.append(action)
        next_state,reward,done,_=env.step(action)
        next_states.append(next_state)
        rewards.append(reward)
        if render:
            env.render()
    if policy:
        print("Average action selection time = ", np.mean(times))
    return states,actions,next_states,rewards

def collect_data(env,policy,n_rollouts=10):
    inputs,outputs=[],[]
    for i in range(n_rollouts):
        states,actions,next_states,rewards=rollout(env,policy)
        input=np.concatenate([states,actions],axis=-1)
        inputs.append(input)
        outputs.append(next_states-states)
    return inputs,outputs