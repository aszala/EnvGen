import collections

import numpy as np

from . import constants
from . import engine
from . import objects
from . import worldgen

import random

# Gym is an optional dependency.
try:
	import gym
	DiscreteSpace = gym.spaces.Discrete
	BoxSpace = gym.spaces.Box
	DictSpace = gym.spaces.Dict
	BaseClass = gym.Env
except ImportError:
	DiscreteSpace = collections.namedtuple('DiscreteSpace', 'n')
	BoxSpace = collections.namedtuple('BoxSpace', 'low, high, shape, dtype')
	DictSpace = collections.namedtuple('DictSpace', 'spaces')
	BaseClass = object


class Env(BaseClass):

	def __init__(self, area=(64, 64), view=(9, 9), size=(64, 64), reward=True, length=10000, seed=None, initial_inventory=None, world_kwargs=None, out_type="dict"):
		# print(seed)
		view = np.array(view if hasattr(view, '__len__') else (view, view))
		size = np.array(size if hasattr(size, '__len__') else (size, size))
		seed = np.random.randint(0, 2**31 - 1) if seed is None else seed
		self._area = area
		self._view = view
		self._size = size
		self._reward = reward
		self._length = length
		self._seed = seed
		self._episode = 0
		self._world = engine.World(area, constants.materials, (12, 12))
		self._textures = engine.Textures(constants.root / 'assets')
		item_rows = int(np.ceil(len(constants.items) / view[0]))
		self._local_view = engine.LocalView(
				self._world, self._textures, [view[0], view[1] - item_rows])
		self._item_view = engine.ItemView(
				self._textures, [view[0], item_rows])
		self._sem_view = engine.SemanticView(self._world, [
				objects.Player, objects.Cow, objects.Zombie,
				objects.Skeleton, objects.Arrow, objects.Plant])
		self._step = None
		self._player = None
		self._last_health = None
		self._unlocked = None
		# Some libraries expect these attributes to be set.
		self.reward_range = None
		self.metadata = None

		self.out_type = out_type

		self.world_kwargs = {
			"enable_mobs": True,
			"enable_trees": True,
			"enable_lava": True,
			"enable_water": True,
			"enable_sand": True,
			"enable_path": True,
			"enable_diamond": True,
			"enable_coal": True,
			"enable_iron": True,
			"enable_mountain": True,
			"peaceful_mode": False,
			"zombie_spawn_thres": 0.993,
			"skeleton_spawn_thres": 0.95,
			"cow_spawn_thres": 0.985,
			"coal_spawn_thres": 0.85,
			"iron_spawn_thres": 0.75,
			"diamond_spawn_thres": 0.994,
			"tree_spawn_thres": 0.8,
			"target_biome": "none",
			"max_num_cows": 10000,
			"max_num_zombies": 10000,
			"max_num_skeletons": 10000,
			"min_cow_spawn_dist": 3,
			"min_zombie_spawn_dist": 10,
			"min_skeleton_spawn_dist": 0,
			"max_cow_spawn_dist": 10000,
			"max_zombie_spawn_dist": 10000,
			"max_skeleton_spawn_dist": 10000,
		}
		
		if world_kwargs is not None:
			# peaceful mode
			if "peaceful_mode" in world_kwargs:
				self.world_kwargs["peaceful_mode"] = world_kwargs["peaceful_mode"]

			# target_biome
			if "target_biome" in world_kwargs:
				if world_kwargs["target_biome"] == "natural":
					world_kwargs["target_biome"] = "none"

				self.world_kwargs["target_biome"] = world_kwargs["target_biome"]

			# coal rarity
			if "coal_rarity" in world_kwargs:
				x = self.world_kwargs["coal_spawn_thres"]

				if world_kwargs["coal_rarity"] == "rare":
					x = 0.95
				elif world_kwargs["coal_rarity"] == "common":
					x = 0.85
				elif world_kwargs["coal_rarity"] == "very_common":
					x = 0.75
				elif world_kwargs["coal_rarity"] == "default":
					x = self.world_kwargs["coal_spawn_thres"]
				
				self.world_kwargs["coal_spawn_thres"] = x

			# iron rarity
			if "iron_rarity" in world_kwargs:
				x = self.world_kwargs["iron_spawn_thres"]

				if world_kwargs["iron_rarity"] == "rare":
					x = 0.85
				elif world_kwargs["iron_rarity"] == "common":
					x = 0.75
				elif world_kwargs["iron_rarity"] == "very common":
					x = 0.65
				elif world_kwargs["iron_rarity"] == "default":
					x = self.world_kwargs["iron_spawn_thres"]
				
				self.world_kwargs["iron_spawn_thres"] = x

			# diamond rarity
			if "diamond_rarity" in world_kwargs:
				x = self.world_kwargs["diamond_spawn_thres"]

				if world_kwargs["diamond_rarity"] == "rare":
					x = 0.994
				elif world_kwargs["diamond_rarity"] == "common":
					x = 0.9
				elif world_kwargs["diamond_rarity"] == "very common":
					x = 0.85
				elif world_kwargs["diamond_rarity"] == "default":
					x = self.world_kwargs["diamond_spawn_thres"]
				
				self.world_kwargs["diamond_spawn_thres"] = x

			# tree rarity
			if "tree_rarity" in world_kwargs:
				x = self.world_kwargs["tree_spawn_thres"]

				if world_kwargs["tree_rarity"] == "rare":
					x = 0.9
				elif world_kwargs["tree_rarity"] == "common":
					x = 0.8
				elif world_kwargs["tree_rarity"] == "very common":
					x = 0.7
				elif world_kwargs["tree_rarity"] == "default":
					x = self.world_kwargs["tree_spawn_thres"]
				
				self.world_kwargs["tree_spawn_thres"] = x

			self.world_kwargs["spawn_mobs"] = []
			# spawn cows
			if "spawn_cows" in world_kwargs:
				self.world_kwargs["spawn_mobs"].append(["cow", world_kwargs["spawn_cows"]])

			# spawn zombies
			if "spawn_zombies" in world_kwargs:
				self.world_kwargs["spawn_mobs"].append(["zombie", world_kwargs["spawn_zombies"]])

			# spawn skeletons
			if "spawn_skeletons" in world_kwargs:
				self.world_kwargs["spawn_mobs"].append(["skeleton", world_kwargs["spawn_skeletons"]])

		self.initial_inventory = initial_inventory


	@property
	def observation_space(self):
		return BoxSpace(0, 255, tuple(self._size) + (3,), np.uint8)


	@property
	def action_space(self):
		return DiscreteSpace(len(constants.actions))


	@property
	def action_names(self):
		return constants.actions


	def reset(self):
		center = (self._world.area[0] // 2, self._world.area[1] // 2)
		self._episode += 1
		self._step = 0
		self._world.reset(seed=hash((self._seed, self._episode)) % (2 ** 31 - 1))
		self._update_time()
		self._player = objects.Player(self._world, center, initial_inventory=self.initial_inventory)
		self._last_health = self._player.health
		self._world.add(self._player)
		self._unlocked = set()
		
		worldgen.generate_world(self._world, self._player, **self.world_kwargs)

		if "spawn_mobs" in self.world_kwargs:
			for mob, num in self.world_kwargs["spawn_mobs"]:
				for _ in range(num):
					x, y = self._world.random.randint(self._player.pos[0] - 5, self._player.pos[0] + 5), self._world.random.randint(self._player.pos[0] - 4, self._player.pos[0] + 4)
					distance = np.sqrt((x - self._player.pos[0]) ** 2 + (y - self._player.pos[1]) ** 2)

					attempts = 0

					while self._world[x, y][0] in [ "water", "stone", "lava", "tree" ] or distance < 3 or self._world[x, y][1] != None:
						x, y = self._world.random.randint(self._player.pos[0] - 5, self._player.pos[0] + 5), self._world.random.randint(self._player.pos[0] - 4, self._player.pos[0] + 4)
						distance = np.sqrt((x - self._player.pos[0]) ** 2 + (y - self._player.pos[1]) ** 2)
						attempts += 1

						if attempts > 100:
							break

					if attempts > 100:
						continue

					if mob == "cow":
						self._world.add(objects.Cow(self._world, (x, y)))
					elif mob == "zombie":
						self._world.add(objects.Zombie(self._world, (x, y), self._player))
					elif mob == "skeleton":
						self._world.add(objects.Skeleton(self._world, (x, y), self._player))

		if self.initial_inventory:
			delta_x = random.randint(-4, 4)

			if "table" in self.initial_inventory:
				self._player.inventory["wood"] += 2
				target = (self._player.pos[0] + self._player.facing[0] + delta_x, self._player.pos[1] + self._player.facing[1])
				material, obj = self._world[target]

				self._player._place("table", target, material)
			
			if "furnace" in self.initial_inventory:
				self._player.inventory["stone"] += 4
				target = (self._player.pos[0] + self._player.facing[0] + delta_x, self._player.pos[1] + self._player.facing[1] - 2)
				material, obj = self._world[target]

				self._player._place("furnace", target, material)
			
			if "plant" in self.initial_inventory:
				delta_y = random.randint(-4, 4)
				if delta_y == -1:
					delta_y = 0

				self._player.inventory["sapling"] += 1
				target = (self._player.pos[0] + self._player.facing[0] + random.randint(-4, 4), self._player.pos[1] + self._player.facing[1] + delta_y)
				material, obj = self._world[target]

				self._player._place("plant", target, material)

				self._world._objects[-1].grown = 295

			for name, amount in self.initial_inventory.items():
				if "|" in name:
					names = name.split("|")
					name = random.choice(names)
				
				if name == "table" or name == "furnace" or name == "plant":
					continue
				
				if amount == -1:
					amount = random.randint(1, 5)

				self._player.inventory[name] = amount

		return self._obs()


	def step(self, action):
		self._step += 1
		self._update_time()
		self._player.action = constants.actions[action]
		for obj in self._world.objects:
			if self._player.distance(obj) < 2 * max(self._view):
				obj.update()
		if self._step % 10 == 0:
			for chunk, objs in self._world.chunks.items():
				self._balance_chunk(chunk, objs)
		obs = self._obs()
		reward = (self._player.health - self._last_health) / 10
		
		unlocked = {
				name for name, count in self._player.achievements.items()
				if count > 0 and name not in self._unlocked
			}
		if unlocked:
			self._unlocked |= unlocked
			reward += 1.0
		
		self._last_health = self._player.health
		
		dead = self._player.health <= 0
		over = self._length and self._step >= self._length
		done = dead or over
		info = {
			'inventory': self._player.inventory.copy(),
			'achievements': self._player.achievements.copy(),
			'discount': 1 - float(dead),
			'semantic': self._sem_view(),
			'player_pos': self._player.pos,
			'reward': reward,
			"episode": {
				"r": reward,
				"l": self._step,
			}
		}
		if not self._reward:
			reward = 0.0
		return obs, reward, done, info


	def render(self, size=None):
		if (type(size) != list or type(size) != tuple) and len(size) != 2:
			size = (512, 512)
			# self._view = np.array([64, 64])
			# item_rows = int(np.ceil(len(constants.items) / self._view[0]))
			# self._local_view = engine.LocalView(
			# 		self._world, self._textures, [self._view[0], self._view[1] - item_rows])
			# self._item_view = engine.ItemView(
			# 		self._textures, [self._view[0], item_rows])

		if size is None:
			size = self._size

		unit = size // self._view
		canvas = np.zeros(tuple(size) + (3,), np.uint8)
		local_view = self._local_view(self._player, unit)
		item_view = self._item_view(self._player.inventory, unit)
		view = np.concatenate([local_view, item_view], 1)
		border = (size - (size // self._view) * self._view) // 2
		(x, y), (w, h) = border, view.shape[:2]
		canvas[x: x + w, y: y + h] = view
		return canvas.transpose((1, 0, 2))


	def _obs(self):
		if self.out_type == "image":
			return self.render(self._size)
		elif self.out_type == "dict":
			out = {
				"image": self.render(self._size),
			}

			return out


	def _update_time(self):
		progress = (self._step / 300) % 1 + 0.3
		daylight = 1 - np.abs(np.cos(np.pi * progress)) ** 3
		self._world.daylight = daylight


	def _balance_chunk(self, chunk, objs):
		light = self._world.daylight
		
		if not self.world_kwargs["enable_mobs"]:
			return

		self._balance_object(
				chunk, objs, objects.Cow, 'grass', 5, 5, 0.01, 0.1,
				lambda pos: objects.Cow(self._world, pos),
				lambda num, space: (0 if space < 30 else 1, 1.5 + light))

		if self.world_kwargs["peaceful_mode"]:
			return
		
		self._balance_object(
				chunk, objs, objects.Zombie, 'grass', 6, 0, 0.3, 0.4,
				lambda pos: objects.Zombie(self._world, pos, self._player),
				lambda num, space: (
						0 if space < 50 else 3.5 - 3 * light, 3.5 - 3 * light))
		self._balance_object(
				chunk, objs, objects.Skeleton, 'path', 7, 7, 0.1, 0.1,
				lambda pos: objects.Skeleton(self._world, pos, self._player),
				lambda num, space: (0 if space < 6 else 1, 2))
		

	def _balance_object(
			self, chunk, objs, cls, material, span_dist, despan_dist,
			spawn_prob, despawn_prob, ctor, target_fn):
		xmin, xmax, ymin, ymax = chunk
		random = self._world.random
		creatures = [obj for obj in objs if isinstance(obj, cls)]
		mask = self._world.mask(*chunk, material)
		target_min, target_max = target_fn(len(creatures), mask.sum())
		if len(creatures) < int(target_min) and random.uniform() < spawn_prob:
			xs = np.tile(np.arange(xmin, xmax)[:, None], [1, ymax - ymin])
			ys = np.tile(np.arange(ymin, ymax)[None, :], [xmax - xmin, 1])
			xs, ys = xs[mask], ys[mask]
			i = random.randint(0, len(xs))
			pos = np.array((xs[i], ys[i]))
			empty = self._world[pos][1] is None
			away = self._player.distance(pos) >= span_dist
			if empty and away:
				self._world.add(ctor(pos))
		elif len(creatures) > int(target_max) and random.uniform() < despawn_prob:
			obj = creatures[random.randint(0, len(creatures))]
			away = self._player.distance(obj.pos) >= despan_dist
			if away:
				self._world.remove(obj)
