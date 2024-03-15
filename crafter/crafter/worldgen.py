import functools

import numpy as np
import opensimplex

from . import constants
from . import objects


def generate_world(world, player, **kwargs):
	simplex = opensimplex.OpenSimplex(seed=world.random.randint(0, 2 ** 31 - 1))
	tunnels = np.zeros(world.area, bool)
	for x in range(world.area[0]):
			for y in range(world.area[1]):
					_set_material(world, (x, y), player, tunnels, simplex, **kwargs)
	for x in range(world.area[0]):
			for y in range(world.area[1]):
					_set_object(world, (x, y), player, tunnels, **kwargs)


def _set_material(world, pos, player, tunnels, simplex, **kwargs):
	x, y = pos
	simplex = functools.partial(_simplex, simplex)
	uniform = world.random.uniform
	start = 4 - np.sqrt((x - player.pos[0]) ** 2 + (y - player.pos[1]) ** 2)
	start += 2 * simplex(x, y, 8, 3)
	start = 1 / (1 + np.exp(-start))
	water = simplex(x, y, 3, {15: 1, 5: 0.15}, False) + 0.1
	water -= 2 * start
	mountain = simplex(x, y, 0, {15: 1, 5: 0.3})
	mountain -= 4 * start + 0.3 * water

	start_thres = 0.5
	start_material = "grass"
	if kwargs["target_biome"] == "mountain":
		mountain = abs(mountain)
		mountain *= 2
		water *= 0.1
		start_thres = 0.6
		start_material = "path"
	elif kwargs["target_biome"] == "beaches":
		mountain *= 0.1
		start_thres = 0.8
		start_material = "grass"
	elif kwargs["target_biome"] == "grassland":
		water *= 0.1
		mountain *= 0.1
		start_thres = 0.6
		start_material = "grass"
		
	if start > start_thres:
		world[x, y] = start_material
	elif mountain > 0.15 and kwargs["enable_mountain"]:
		if (simplex(x, y, 6, 7) > 0.15 and mountain > 0.3) and kwargs["enable_path"]:    # cave
			world[x, y] = 'path'
		elif simplex(2 * x, y / 5, 7, 3) > 0.4 and kwargs["enable_path"]:    # horizonal tunnle
			world[x, y] = 'path'
			tunnels[x, y] = True
		elif simplex(x / 5, 2 * y, 7, 3) > 0.4 and kwargs["enable_path"]:    # vertical tunnle
			world[x, y] = 'path'
			tunnels[x, y] = True
		elif mountain > 0.18 and uniform() > kwargs["coal_spawn_thres"] and kwargs["enable_coal"]:
			world[x, y] = 'coal'
		elif mountain > 0.18 and uniform() > kwargs["iron_spawn_thres"] and kwargs["enable_iron"]:
			world[x, y] = 'iron'
		elif mountain > 0.18 and uniform() > kwargs["diamond_spawn_thres"] and kwargs["enable_diamond"]:
			world[x, y] = 'diamond'
		elif mountain > 0.3 and simplex(x, y, 6, 5) > 0.35 and kwargs["enable_lava"]:
			world[x, y] = 'lava'
		else:
			world[x, y] = 'stone'
	elif 0.25 < water <= 0.35 and simplex(x, y, 4, 9) > -0.2 and kwargs["enable_sand"]:
		world[x, y] = 'sand'
	elif water > 0.3  and kwargs["enable_water"]:
		world[x, y] = 'water'
	else:
		if simplex(x, y, 5, 7) > 0 and uniform() > kwargs["tree_spawn_thres"] and kwargs["enable_trees"]:
			world[x, y] = 'tree'
		else:
			world[x, y] = 'grass'


def _set_object(world, pos, player, tunnels, **kwargs):
	x, y = pos
	
	uniform = world.random.uniform
	dist = np.sqrt((x - player.pos[0]) ** 2 + (y - player.pos[1]) ** 2)
	material, _ = world[x, y]

	if material not in constants.walkable:
			return
	
	if not kwargs["enable_mobs"]:
			return

	num_cows = len([1 for obj in world._objects if isinstance(obj, objects.Cow)])
	num_zombies = len([1 for obj in world._objects if isinstance(obj, objects.Zombie)])
	num_skeletons = len([1 for obj in world._objects if isinstance(obj, objects.Skeleton)])

	if dist < kwargs["max_cow_spawn_dist"] and dist > kwargs["min_cow_spawn_dist"] and material == 'grass' and uniform() > kwargs["cow_spawn_thres"] and num_cows < kwargs["max_num_cows"]:
			world.add(objects.Cow(world, (x, y)))
	elif dist < kwargs["max_zombie_spawn_dist"] and dist > max(kwargs["min_zombie_spawn_dist"], 10) and uniform() > kwargs["zombie_spawn_thres"] and not kwargs["peaceful_mode"] and num_zombies < kwargs["max_num_zombies"]:
			world.add(objects.Zombie(world, (x, y), player))
	elif dist < kwargs["max_skeleton_spawn_dist"] and dist > max(kwargs["min_skeleton_spawn_dist"], 10) and material == 'path' and tunnels[x, y] and uniform() > kwargs["skeleton_spawn_thres"] and not kwargs["peaceful_mode"] and num_skeletons < kwargs["max_num_skeletons"]:
			world.add(objects.Skeleton(world, (x, y), player))


def _simplex(simplex, x, y, z, sizes, normalize=True):
	if not isinstance(sizes, dict):
			sizes = {sizes: 1}
	value = 0
	for size, weight in sizes.items():
			if hasattr(simplex, 'noise3d'):
					noise = simplex.noise3d(x / size, y / size, z)
			else:
					noise = simplex.noise3(x / size, y / size, z)
			value += weight * noise
	if normalize:
			value /= sum(sizes.values())
	return value
