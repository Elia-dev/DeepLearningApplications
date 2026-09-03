import numpy as np
import torch
from torch.distributions import Categorical

# Select an action from the policy given the current observation from the environment. Return the action and its log probability.
def select_action(env, obs, policy):
    """Sample an action from the policy's distribution and return (action, log_prob)."""
    dist = Categorical(policy(obs))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob.reshape(1)

def compute_returns(rewards, gamma):
    """Compute the discounted returns for a sequence of rewards"""
    returns = np.zeros_like(rewards, dtype=np.float32)
    G = 0
    # backward computation of returns
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns

def run_episode(env, policy, maxlen=500):
    """Run a single episode in the environment using the given policy.
       Return the observations, actions, log probabilities, and rewards collected during the episode."""
    observations = []
    actions = []
    log_probs = []
    rewards = []

    (obs, info) = env.reset()
    for i in range(maxlen):
        # Get the current observation, run the policy and select an action
        obs = torch.tensor(obs, dtype=torch.float32)
        (action, log_prob) = select_action(env, obs, policy)
        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob)

        # Step the environment with the selected action and collect the reward
        (obs, reward, term, trunc, info) = env.step(action)
        rewards.append(reward)
        if term or trunc:
            break
    return observations, actions, torch.cat(log_probs), rewards