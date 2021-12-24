import torch
import numpy as np

def eval_policy(env, model, num_ep):
    ep_rewards = []
    ep_lengths = []
    num_successes = 0
    for i in range(num_ep):
        done = False
        ep_reward = 0
        ep_length = 0
        obs = env.reset()
        while not done:
            with torch.no_grad():
                action = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            if 'is_success' in info and info['is_success']:
                num_successes += 1
        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_length)

    metrics = dict(reward=np.mean(ep_rewards), stddev=np.std(ep_rewards), length=np.mean(ep_lengths))
    if 'is_success' in info:
        metrics['success_rate'] = num_successes/num_ep
    
    return metrics