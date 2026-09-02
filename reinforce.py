import torch
import torch.nn as nn
from utils import run_episode, compute_returns
from torch.distributions import Categorical

def evaluate_policy(policy, env, M: int, maxlen: int) -> tuple[float, float]:
    """Evaluate the policy by running M episodes in the environment and returning the average reward
        and average episode length."""
    policy.eval()
    ep_rewards, ep_lengths = [], []
    for _ in range(M):
        _, _, _, rewards = run_episode(env, policy, maxlen=maxlen)
        ep_rewards.append(sum(rewards))
        ep_lengths.append(len(rewards))
    return sum(ep_rewards) / M, sum(ep_lengths) / M

def evaluate_lander(policy, env, M, maxlen):
    landings = 0
    policy.eval()
    ep_rewards, ep_lengths = [], []
    for _ in range(M):
        _, _, _, rewards = run_episode(env, policy, maxlen=maxlen)
        ep_rewards.append(sum(rewards))
        ep_lengths.append(len(rewards))
        if sum(rewards) >= 200:   # landing bonus
            landings += 1
    return landings / M, sum(ep_rewards) / M, sum(ep_lengths) / M

def reinforce(policy, env, env_render=None, gamma=0.99, num_episodes=10,
              N=100, M=10, standardize=True, lr=1e-2, maxlen=500):

    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    running_rewards = [0.0]
    eval_rewards = []
    eval_lengths = []

    policy.train()
    for episode in range(num_episodes):
        (observations, actions, log_probs, rewards) = run_episode(env, policy, maxlen=maxlen)

        # Compute the discounted returns and update the running average of rewards
        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32)
        running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])

        # Standardize the returns
        if standardize:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Update the policy
        opt.zero_grad()
        loss = (-log_probs * returns).mean()
        loss.backward()
        opt.step()

        # Evaluate the policy every N episodes
        if episode % N == 0:
            avg_reward, avg_length = evaluate_policy(policy, env, M, maxlen)
            eval_rewards.append(avg_reward)
            eval_lengths.append(avg_length)
            print(f'Episode {episode} | Avg eval reward: {avg_reward:.1f} | Avg eval length: {avg_length:.1f}')

            if env_render:
                run_episode(env_render, policy, maxlen=maxlen)
            policy.train()

    policy.eval()
    return running_rewards, eval_rewards, eval_lengths


def reinforce_baseline(policy, value_net, env, env_render=None, gamma=0.99,
                        num_episodes=10, N=100, M=10, lr=1e-2, maxlen=500):
    """Train a policy using the REINFORCE algorithm with a value function baseline."""
    policy_opt = torch.optim.Adam(policy.parameters(), lr=lr)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=lr)
    value_loss_fn = nn.MSELoss()

    running_rewards = [0.0]
    eval_rewards = []
    eval_lengths = []

    policy.train()
    value_net.train()
    for episode in range(num_episodes):
        (observations, actions, log_probs, rewards) = run_episode(env, policy, maxlen=maxlen)

        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32)
        running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])

        obs_batch = torch.stack(observations).float()
        values = value_net(obs_batch).squeeze(-1)

        # Compute advantages: compute the difference between the returns and the value estimates,
        # so how much better or worse the actual return was compared to what the value network predicted.
        # .detach() so the gradient of the policy loss does not flow into the value network.
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute the policy loss and value loss
        policy_loss = (-log_probs * advantages).mean()
        value_loss = value_loss_fn(values, returns)

        # Update the policy and value networks
        policy_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        policy_opt.step()

        # Update the value network
        value_opt.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
        value_opt.step()

        # Evaluate the policy every N episodes
        if episode % N == 0:
            avg_reward, avg_length = evaluate_policy(policy, env, M, maxlen)
            eval_rewards.append(avg_reward)
            eval_lengths.append(avg_length)
            print(f'Episode {episode} | Avg eval reward: {avg_reward:.1f} | Avg eval length: {avg_length:.1f}')
            if env_render:
                run_episode(env_render, policy, maxlen=maxlen)
            policy.train()

    policy.eval()
    value_net.eval()
    return running_rewards, eval_rewards, eval_lengths

def reinforce_lunar(policy, value_net, env, env_render=None, gamma=0.99,
                        num_episodes=10, N=100, M=10, lr=1e-2, maxlen=500, entropy_coef: float = 0.01):
    """Train a policy using the REINFORCE algorithm with a value function baseline."""
    policy_opt = torch.optim.Adam(policy.parameters(), lr=lr)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=lr)
    value_loss_fn = nn.MSELoss()

    running_rewards = [0.0]
    eval_rewards = []
    eval_lengths = []

    policy.train()
    value_net.train()
    for episode in range(num_episodes):
        (observations, actions, log_probs, rewards) = run_episode(env, policy, maxlen=maxlen)

        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32)
        running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])

        obs_batch = torch.stack(observations).float()
        values = value_net(obs_batch).squeeze(-1)

        # Compute advantages: compute the difference between the returns and the value estimates,
        # so how much better or worse the actual return was compared to what the value network predicted.
        # .detach() so the gradient of the policy loss does not flow into the value network.
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute the policy loss and value loss
        dist = Categorical(policy(obs_batch))
        entropy = dist.entropy().mean() # average entropy across the batch
        policy_loss = (-log_probs * advantages).mean() - entropy_coef * entropy # maximize entropy to encourage exploration
        value_loss = value_loss_fn(values, returns)

        # Update the policy and value networks
        policy_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        policy_opt.step()

        # Update the value network
        value_opt.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
        value_opt.step()

        # Evaluate the policy every N episodes
        if episode % N == 0:
            landings, avg_reward, avg_length = evaluate_lander(policy, env, M, maxlen)
            eval_rewards.append(avg_reward)
            eval_lengths.append(avg_length)
            print(f'Episode {episode} | Avg eval reward: {avg_reward:.1f} | Avg eval length: {avg_length:.1f} | Landing success rate: {landings:.2%}')
            if env_render:
                run_episode(env_render, policy, maxlen=maxlen)
            policy.train()

    policy.eval()
    value_net.eval()
    return running_rewards, eval_rewards, eval_lengths
