import argparse
import random
import numpy as np
import imageio
import os

from env_loader import EnvLoader
from crafter.env import Env

def get_env(skill_config, nproc):
    env_loader = EnvLoader(skill_config, nproc)
    venv, total_envs = env_loader.get_new_sample()
    venv.reset()

    return venv, total_envs


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    

    venv, _ = get_env(args.config_path,  args.num_repeats)
    images = venv.get_images()

    os.makedirs(args.save_path, exist_ok=True)

    for i in range(len(images)):
        frame = images[i]
        imageio.imsave(f"{args.save_path}/env_{i}.png", frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="./environment_visualizations/")

    args = parser.parse_args()

    main(args)
