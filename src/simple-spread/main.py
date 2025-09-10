import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
import imageio.v2 as imageio
from pettingzoo.mpe import simple_spread_v3

if __name__ == "__main__":
    
    output_dir = 'gifs/simple-spread'
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)

    rendering = False # Whether you want or not to render the experiment (N.B. without rendering the env is faster)

    env = simple_spread_v3.env(N = 10, render_mode="rgb_array" if rendering else None)
    env.reset(seed=42)

    frames = []
    steps = 400

    for agent in env.agent_iter(max_iter=steps):
        obs, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            action = None
        else:
            action = env.action_space(agent).sample()
        env.step(action)

        if rendering:
            # Get current frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)

    env.close()

    if rendering:
        # Save gif
        imageio.mimsave(f'{output_dir}/mpe_random.gif', frames, fps=15)