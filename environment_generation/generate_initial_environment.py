

import sys, os
sys.path.append(os.getcwd())

import json
import argparse
import openai as oai
from dotenv import dotenv_values


gpt_prompt = """
You are a environment designer agent for a game called "Crafter".
Your job is to design four environments which can be used to teach an agent how to play.
The environments should not be super easy, but instead focus on putting the agent in a situation where it can learn on its own eventually.

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

Output in the following format and include a brief explanation of the purpose of each environment: 

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
    parser.add_argument("--save_path", type=str, default="./environment_generation/generated_environments/llm_environments.json")
    parser.add_argument("--use_azure", action="store_true")
    parser.add_argument("--engine", type=str, default="gpt-4-1106-preview")
    
    args = parser.parse_args()

    save_dir = '/'.join(args.save_path.split("/")[:-1])
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    chatlog = [ {"role": "user", "content": gpt_prompt} ]

    resp = call_api(log=chatlog, engine=args.engine, use_azure=args.use_azure)

    with open(f"{args.save_path.replace('.json', '_raw.txt')}", "w") as f:
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
            x = json.loads(s)
            envs.append(x)
            s = ""
            continue

        if track:
            s += line.replace("True", "true").replace("False", "false") + "\n"
    
    with open(args.save_path, "w") as f:
        json.dump(envs, f, indent=2)
