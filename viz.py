import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

# Help from Claude because I was struggling to get the video to render since I run the code in a remote environment
# This function renders an episode of the environment as a video using the given policy
# It returns an HTML object that can be displayed at any moment
def render_episode_as_video(env_name, policy, max_steps=500):
    env_vis = gym.make(env_name, render_mode='rgb_array')
    obs, _ = env_vis.reset()
    frames = [env_vis.render()]

    with torch.no_grad():
        for _ in range(max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32)
            probs = policy(obs_t)
            action = torch.argmax(probs).item()
            obs, _, terminated, truncated, _ = env_vis.step(action)
            frames.append(env_vis.render())
            if terminated or truncated:
                break

    env_vis.close()

    fig = plt.figure(figsize=(6, 6))
    plt.axis('off')
    img = plt.imshow(frames[0])

    def update(i):
        img.set_data(frames[i])
        return [img]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=40)
    plt.close(fig)
    return HTML(anim.to_jshtml())
