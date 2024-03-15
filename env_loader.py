import json
import numpy as np
from functools import partial

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from crafter.env import Env

class EnvLoader():

    def __init__(self, config_path, num_repeats, **kwargs):
        self.config_path = config_path
        self.num_repeats = num_repeats

        self.kwargs = kwargs

        self.params = []
        
        with open(f"{config_path}", "r") as f:
            world_config = json.load(f)
        
        for env in world_config:
            inventory_items = env["inventory_settings"]
            temp = []

            for item in inventory_items:
                
                if "stone" in item and inventory_items[item] > 0:
                    temp.append("wood_pickaxe")

                if "iron" in item and inventory_items[item] > 0:
                    temp.append("stone_pickaxe")

                if "diamond" in item and inventory_items[item] > 0:
                    temp.append("iron_pickaxe")

            for item in temp:
                inventory_items[item] = 1

            self.params.append({
                "seed": np.random.randint(0, 2**31 - 1),
                "world_kwargs": env["environment_settings"],
                "initial_inventory": inventory_items,
            })

    
    def get_new_sample(self):
        params = []

        for j, param in enumerate(self.params):
            n = self.num_repeats

            for i in range(n):
                x = param.copy()
                x["seed"] = np.random.randint(0, 2**31 - 1)
            
                params.append(x)

        env_fns = [ partial(Env,
                            length=1024, 
                            seed=param["seed"],
                            world_kwargs=param["world_kwargs"], 
                            initial_inventory=param["initial_inventory"],
                            **self.kwargs
                        )
                        for param in params
                    ]
    
        venv = SubprocVecEnv(env_fns)
        venv = VecMonitor(venv)

        return venv, len(params)
