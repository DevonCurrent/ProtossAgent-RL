import random
import math
import os.path

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app

DATA_FILE = 'sparse_agent_data'

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_PROBE = 'selectprobe'
ACTION_BUILD_PYLON = 'buildpylon'
ACTION_BUILD_GATEWAY = 'buildgateway'
ACTION_SELECT_GATEWAY = 'selectgateway'
ACTION_BUILD_ZEALOT = 'buildzealot'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'

smart_actions = [
	ACTION_DO_NOTHING,
	ACTION_SELECT_PROBE,
	ACTION_BUILD_PYLON,
	ACTION_BUILD_GATEWAY,
	ACTION_SELECT_GATEWAY,
	ACTION_BUILD_ZEALOT,
	ACTION_SELECT_ARMY,
]

for mm_x in range(0, 64):
	for mm_y in range(0, 64):
		if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
			smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
	def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
		self.actions = actions  # a list
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy
		self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

	def choose_action(self, observation):
		self.check_state_exist(observation)
		
		if np.random.uniform() < self.epsilon:
			# choose best action
			state_action = self.q_table.ix[observation, :]
			
			# some actions have the same value
			state_action = state_action.reindex(np.random.permutation(state_action.index))
			
			action = state_action.idxmax()
		else:
			# choose random action
			action = np.random.choice(self.actions)
			
		return action

	def learn(self, s, a, r, s_):
		if s == s_:
			return
		
		self.check_state_exist(s_)
		self.check_state_exist(s)
		
		q_predict = self.q_table.ix[s, a]
		
		if s_ != 'terminal':
			q_target = r + self.gamma * self.q_table.ix[s_, :].max()
		else:
			q_target = r  # next state is terminal
		
		# update
		self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

	def check_state_exist(self, state):
		if state not in self.q_table.index:
			# append new state to q table
			self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class ProtossAgentQLearning(base_agent.BaseAgent):
	def __init__(self):
		super(ProtossAgentQLearning, self).__init__()
		
		self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
		
		self.previous_action = None
		self.previous_state = None
		
		self.attack_coordinates = None
		
		if os.path.isfile(DATA_FILE + '.gz'):
			self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')
		
	def unit_type_is_selected(self, obs, unit_type):
		if (len(obs.observation.single_select) > 0 and
		obs.observation.single_select[0].unit_type == unit_type):
				return True

		if (len(obs.observation.multi_select) > 0 and
		obs.observation.multi_select[0].unit_type == unit_type):
			return True

		return False

	def get_units_by_type(self, obs, unit_type):
		return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

	def can_do(self, obs, action):
		return action in obs.observation.available_actions

	def step(self, obs):
		super(ProtossAgentQLearning, self).step(obs)
		
		if obs.last():
			reward = obs.reward
			
			self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')
			
			self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
			
			self.previous_action = None
			self.previous_state = None
			
			return actions.FUNCTIONS.no_op()
		
		player_y, player_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
		self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
		
		nexus = self.get_units_by_type(obs, units.Protoss.Nexus)
		gateways = self.get_units_by_type(obs, units.Protoss.Gateway)
		pylons = self.get_units_by_type(obs, units.Protoss.Pylon)
		mineral_count = obs.observation.player.minerals
		worker_count = obs.observation.player.food_workers
		free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
		killed_unit_score = obs.observation.score_cumulative.killed_value_units
		killed_building_score = obs.observation.score_cumulative.killed_value_structures
		army_supply = obs.observation.player.food_army
		
		current_state = np.zeros(20)
		current_state[0] = len(nexus)
		current_state[1] = len(pylons)
		current_state[2] = len(gateways)
		current_state[3] = army_supply
		
		hot_squares = np.zeros(4)        
		enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.ENEMY).nonzero()
		for i in range(0, len(enemy_y)):
			y = int(math.ceil((enemy_y[i] + 1) / 32))
			x = int(math.ceil((enemy_x[i] + 1) / 32))
			
			hot_squares[((y - 1) * 2) + (x - 1)] = 1
		
		if not self.base_top_left:
			hot_squares = hot_squares[::-1]
		
		for i in range(0, 4):
			current_state[i + 4] = hot_squares[i]
		
		green_squares = np.zeros(4)
		friendly_y, friendly_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
		for i in range(0, len(friendly_y)):
			y = int(math.ceil((friendly_y[i] + 1) / 32))
			x = int(math.ceil((friendly_x[i] + 1) / 32))
			
			green_squares[((y - 1) * 2) + (x - 1)] = 1
		
		if not self.base_top_left:
			green_squares = green_squares[::-1]
		
		for i in range(0, 4):
			current_state[i + 8] = green_squares[i]
		
		
		if self.previous_action is not None:
			self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))
		
		
		rl_action = self.qlearn.choose_action(str(current_state))
		smart_action = smart_actions[rl_action]
		
		self.previous_army_supply = army_supply
		self.previous_killed_unit_score = killed_unit_score
		self.previous_killed_building_score = killed_building_score
		self.previous_state = current_state
		self.previous_action = rl_action
		
		x = 0
		y = 0
		if '_' in smart_action:
			smart_action, x, y = smart_action.split('_')
			self.attack_coordinates = (int(x), int(y))
		
		if smart_action == ACTION_DO_NOTHING:
			return actions.FUNCTIONS.no_op()

		elif smart_action == ACTION_SELECT_PROBE:
			probes = self.get_units_by_type(obs, units.Protoss.Probe)
			if len(probes) > 0:
				probe = random.choice(probes)
				if(0 < probe.x < 83):
					if(0 < probe.y < 83):
						return actions.FUNCTIONS.select_point("select", (probe.x, probe.y))

		elif smart_action == ACTION_BUILD_PYLON:
			if self.unit_type_is_selected(obs, units.Protoss.Probe):
				if self.can_do(obs, actions.FUNCTIONS.Build_Pylon_screen.id):
					x = random.randint(0, 83)
					y = random.randint(0, 83)
					return actions.FUNCTIONS.Build_Pylon_screen("now", (x, y))

		elif smart_action == ACTION_BUILD_GATEWAY:
			if self.unit_type_is_selected(obs, units.Protoss.Probe):
				if self.can_do(obs, actions.FUNCTIONS.Build_Gateway_screen.id):
					x = random.randint(0, 83)
					y = random.randint(0, 83)
					return actions.FUNCTIONS.Build_Gateway_screen("now", (x, y))

		elif smart_action == ACTION_SELECT_GATEWAY:
			if len(gateways) > 0 and not self.unit_type_is_selected(obs, units.Protoss.Gateway):
				gateway = random.choice(gateways)
				return actions.FUNCTIONS.select_point("select", (gateway.x, gateway.y))

		elif smart_action == ACTION_BUILD_ZEALOT:
			if self.can_do(obs, actions.FUNCTIONS.Train_Zealot_quick.id):
				return actions.FUNCTIONS.Train_Zealot_quick("now")

		elif smart_action == ACTION_SELECT_ARMY:
			if self.can_do(obs, actions.FUNCTIONS.select_army.id):
				return actions.FUNCTIONS.select_army("select")

		elif smart_action == ACTION_ATTACK:
			if not self.unit_type_is_selected(obs, units.Protoss.Probe) and self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
				return actions.FUNCTIONS.Attack_minimap("now", self.attack_coordinates)

		return actions.FUNCTIONS.no_op()


def main(unused_argv):
	agent = ProtossAgentQLearning()
	try:
		while True:
			with sc2_env.SC2Env(map_name="Catalyst", players=[sc2_env.Agent(sc2_env.Race.protoss),
			sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.easy)],
			agent_interface_format=features.AgentInterfaceFormat(
			feature_dimensions=features.Dimensions(screen=84, minimap=64),
			use_feature_units=True), step_mul=16, game_steps_per_episode=0, visualize=True, save_replay_episodes=0, 
					replay_dir='E:\Program Files (x86)\StarCraft II\Replays') as env:

				agent.setup(env.observation_spec(), env.action_spec())
				timesteps = env.reset()
				agent.reset()

				while True:
					step_actions = [agent.step(timesteps[0])]
					if timesteps[0].last():
						break
					timesteps = env.step(step_actions)

	except KeyboardInterrupt:
		pass



if __name__ == "__main__":
	app.run(main)