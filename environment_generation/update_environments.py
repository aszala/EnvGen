

import sys, os
sys.path.append(os.getcwd())

import json
import argparse
from glob import glob
import numpy as np
import openai as oai
from dotenv import dotenv_values


gpt_prompt = """
You are a environment designer agent for a game called "Crafter".
Your job is to design four environments which can be used to teach an agent how to play.
The agent already knows certain tasks, but it needs to learn how to do the rest.
The environments should not be super easy, but instead focus on putting the agent in a situation where it can learn on its own eventually.

How can we update the pretraining environments to improve agent performance overall?

You can control what the environments look like and what items the agent may start with.

Here is a list of things an agent would need to learn how to do:
collect_coal
collect_diamond
collect_drink
collect_iron
collect_sapling
collect_stone
collect_wood
defeat_skeleton
defeat_zombie
eat_cow
eat_plant
make_iron_pickaxe
make_iron_sword
make_stone_pickaxe
make_stone_sword
make_wood_pickaxe
make_wood_sword
place_furnace
place_plant
place_stone
place_table
wake_up

Here is a list how well the agent already knows some of the tasks:
collect_coal: {collect_coal}% +/- {collect_coal_error}%
collect_diamond: {collect_diamond}% +/- {collect_diamond_error}%
collect_drink: {collect_drink}% +/- {collect_drink_error}%
collect_iron: {collect_iron}% +/- {collect_iron_error}%
collect_sapling: {collect_sapling}% +/- {collect_sapling_error}%
collect_stone: {collect_stone}% +/- {collect_stone_error}%
collect_wood: {collect_wood}% +/- {collect_wood_error}%
defeat_skeleton: {defeat_skeleton}% +/- {defeat_skeleton_error}%
defeat_zombie: {defeat_zombie}% +/- {defeat_zombie_error}%
eat_cow: {eat_cow}% +/- {eat_cow_error}%
eat_plant: {eat_plant}% +/- {eat_plant_error}%
make_iron_pickaxe: {make_iron_pickaxe}% +/- {make_iron_pickaxe_error}%
make_iron_sword: {make_iron_sword}% +/- {make_iron_sword_error}%
make_stone_pickaxe: {make_stone_pickaxe}% +/- {make_stone_pickaxe_error}%
make_stone_sword: {make_stone_sword}% +/- {make_stone_sword_error}%
make_wood_pickaxe: {make_wood_pickaxe}% +/- {make_wood_pickaxe_error}%
make_wood_sword: {make_wood_sword}% +/- {make_wood_sword_error}%
place_furnace: {place_furnace}% +/- {place_furnace_error}%
place_plant: {place_plant}% +/- {place_plant_error}%
place_stone: {place_stone}% +/- {place_stone_error}%
place_table: {place_table}% +/- {place_table_error}%
wake_up: {wake_up}% +/- {wake_up_error}%


Here is the set of environments that the agent learned the above tasks in:
Environment 1:
```json
{env_1}
```
Environment 2:
```json
{env_2}
```
Environment 3:
```json
{env_3}
```
Environment 4:
```json
{env_4}
```


Here is a list of parameters you can control when making an environment:
target_biome: grassland | mountain | beaches | natural
coal_rarity: very common | common | rare
iron_rarity: very common | common | rare
diamond_rarity: very common | common | rare
tree_rarity: very common | common | rare

Here is a list of items the agent can start with:
sapling: 0-9
wood: 0-9
stone: 0-9
coal: 0-9
iron: 0-9
diamond: 0-9
wood_pickaxe: 0-1
stone_pickaxe: 0-1
iron_pickaxe: 0-1
wood_sword: 0-1
stone_sword: 0-1
iron_sword: 0-1


Here is a list of constraints:
agent must sleep in order to get the wake up achievement
during sleep the agent cannot move and may be woken up by attacking monsters or by the sun rising
natural biome will set the environment to have all the biomes
coal, iron, and diamond can only be found in a mountain biome
trees and cows can only be found in a grassland biome
skeletons can only be found in a mountain biome and will shoot arrows at the agent
zombies can be found anywhere and will track/chase the agent
agents can only drink water
diamond can only be mined with a iron_pickaxe
iron can only be mined with a stone_pickaxe
stone can only be mined with a wood_pickaxe
you need a table to craft items and a furnace to craft iron items


You do not need to use every parameter setting, you can skip it to set to its standard setting.
You can leave some of them as default if they are not specifically needed for the task.

Output in the following format and include a brief explanation of the purpose of each environment and explicitly mention what information you used to come up with the environment settings. For example, you can mention the agents' current knowledge of a specific task:

Environment 1:
```json
{
    "environment_settings": {
        ...
    },
    "inventory_settings": {
        ...
    }
}
```
Environment 2:
...
"""

update_prompt = """Those environments resulted in the agent improving up to these scores: 
collect_coal: {collect_coal}% +/- {collect_coal_error}%
collect_diamond: {collect_diamond}% +/- {collect_diamond_error}%
collect_drink: {collect_drink}% +/- {collect_drink_error}%
collect_iron: {collect_iron}% +/- {collect_iron_error}%
collect_sapling: {collect_sapling}% +/- {collect_sapling_error}%
collect_stone: {collect_stone}% +/- {collect_stone_error}%
collect_wood: {collect_wood}% +/- {collect_wood_error}%
defeat_skeleton: {defeat_skeleton}% +/- {defeat_skeleton_error}%
defeat_zombie: {defeat_zombie}% +/- {defeat_zombie_error}%
eat_cow: {eat_cow}% +/- {eat_cow_error}%
eat_plant: {eat_plant}% +/- {eat_plant_error}%
make_iron_pickaxe: {make_iron_pickaxe}% +/- {make_iron_pickaxe_error}%
make_iron_sword: {make_iron_sword}% +/- {make_iron_sword_error}%
make_stone_pickaxe: {make_stone_pickaxe}% +/- {make_stone_pickaxe_error}%
make_stone_sword: {make_stone_sword}% +/- {make_stone_sword_error}%
make_wood_pickaxe: {make_wood_pickaxe}% +/- {make_wood_pickaxe_error}%
make_wood_sword: {make_wood_sword}% +/- {make_wood_sword_error}%
place_furnace: {place_furnace}% +/- {place_furnace_error}%
place_plant: {place_plant}% +/- {place_plant_error}%
place_stone: {place_stone}% +/- {place_stone_error}%
place_table: {place_table}% +/- {place_table_error}%
wake_up: {wake_up}% +/- {wake_up_error}%

Could you generate new environments based on these scores?
"""


def call_api(log, engine="gpt-4-1106-preview", use_azure=True):
    if use_azure:
        resp = oai.ChatCompletion.create(
            engine=engine,
            messages=log,
            temperature=0
        )
    else:
        resp = oai.ChatCompletion.create(
            model=engine,
            messages=log,
            temperature=0
        )

    resp = resp['choices'][0]['message']['content']

    return resp


if __name__ == "__main__":
    config = dotenv_values(".env")

    oai.api_key = config['API_KEY']

    if 'API_TYPE' in config:
        oai.api_type = config['API_TYPE']

    if 'API_BASE' in config:
        oai.api_base = config['API_BASE']

    if 'API_VERSION' in config:
        oai.api_version = config['API_VERSION']

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--params_file", type=str, required=True)
    parser.add_argument("--save_name", type=str, required=True)
    parser.add_argument("--root_dir", type=str, default="./environment_generation/generated_environments/")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--num_evals", type=int, default=3)
    parser.add_argument("--use_azure", action="store_true")
    parser.add_argument("--engine", type=str, default="gpt-4-1106-preview")

    args = parser.parse_args()

    score_files = [
        f"{args.load_path}/stage_2/eval_{i+1}/success_rate_final.json" for i in range(args.num_evals)
    ]

    output_scores = {}

    for score_file in score_files:
        with open(score_file, "r") as f:
            scores = json.load(f)

            for task, score in scores.items():
                score = round(score)
                
                if task not in output_scores:
                    output_scores[task] = []
                
                output_scores[task].append(score)
                
    for task, scores in output_scores.items():
        output_scores[task] = np.mean(scores), np.std(scores)
        gpt_prompt = gpt_prompt.replace("{" + task + "}", str(round(output_scores[task][0])))
        gpt_prompt = gpt_prompt.replace("{" + task + "_error" + "}", str(round(output_scores[task][1])))

    with open(args.params_file, "r") as f:
        params = json.load(f)

        for i, env in enumerate(params):
            gpt_prompt = gpt_prompt.replace("{" + f"env_{i+1}" + "}", json.dumps(env, indent=4))

    save_path = f"{args.root_dir}/{args.save_name}/stage_{args.stage}/"

    os.makedirs(save_path, exist_ok=True)
    save_path += "envs"

    chatlog = [ {"role": "user", "content": gpt_prompt} ]

    stages = glob(f"{args.load_path}/stage_*")
    stages.sort(key=lambda x: int(x.split("/")[-1].replace("stage_", "")))

    for stage in stages:
        x = int(stage.split("/")[-1].replace("stage_", ""))

        if x == 2 or x % 2 != 0 or x > args.stage:
            continue

        with open(f"{args.root_dir}/{args.save_name}/stage_{x-2}/envs.txt", "r") as f:
            prev_res = f.readlines()
            prev_res = '\n'.join(prev_res)

        chatlog.append({"role": "assistant", "content": prev_res})

        score_files = [
            f"{args.load_path}/stage_{x}/eval_{i+1}/success_rate_final.json" for i in range(args.num_evals)
        ]

        output_scores = {}

        for score_file in score_files:
            with open(score_file, "r") as f:
                scores = json.load(f)

                for task, score in scores.items():
                    score = round(score)
                    
                    if task not in output_scores:
                        output_scores[task] = []
                    
                    output_scores[task].append(score)

        new_prompt = update_prompt   
        for task, scores in output_scores.items():
            output_scores[task] = np.mean(scores), np.std(scores)
            new_prompt = new_prompt.replace("{" + task + "}", str(round(output_scores[task][0])))
            new_prompt = new_prompt.replace("{" + task + "_error" + "}", str(round(output_scores[task][1])))

        chatlog.append({"role": "user", "content": new_prompt})

    resp = call_api(log=chatlog, engine=args.engine, use_azure=args.use_azure)

    with open(f"{save_path}.txt", "w") as f:
        f.write(resp)

    lines = resp.split("\n")
    for i, line in enumerate(lines):
        if "//" in line:
            lines[i] = line.split("//")[0].strip()
    
    envs = []
    s = ""
    track = False

    for line in lines:
        if "```json" in line:
            track = True
            continue

        if "```" in line:
            track = False
            # print("-------", s)
            x = json.loads(s)
            envs.append(x)
            s = ""
            continue

        if track:
            s += line.replace("True", "true").replace("False", "false") + "\n"
    
    with open(f"{save_path}.json", "w") as f:
        json.dump(envs, f, indent=2)
