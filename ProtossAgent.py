import random
import math
import os.path

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random

DATA_FILE = 'sparse_agent_data'

ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_PYLON = 'buildpylon'
ACTION_BUILD_GATEWAY = 'buildgateway'
ACTION_BUILD_ZEALOT = 'buildzealot'
ACTION_ATTACK = 'attack'

smart_actions = [
	ACTION_DO_NOTHING,
	ACTION_BUILD_PYLON,
	ACTION_BUILD_GATEWAY,
	ACTION_BUILD_ZEALOT,
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
		self.disallowed_actions = {}

	def choose_action(self, observation, excluded_actions=[]):
		self.check_state_exist(observation)
		
		self.disallowed_actions[observation] = excluded_actions
		
		state_action = self.q_table.ix[observation, :]
		
		for excluded_action in excluded_actions:
			del state_action[excluded_action]

		if np.random.uniform() < self.epsilon:
			# some actions have the same value
			state_action = state_action.reindex(np.random.permutation(state_action.index))
			
			action = state_action.idxmax()
		else:
			action = np.random.choice(state_action.index)
		
		return action

	def learn(self, s, a, r, s_):
		if s == s_:
			return
		
		self.check_state_exist(s_)
		self.check_state_exist(s)
		
		q_predict = self.q_table.ix[s, a]
		
		s_rewards = self.q_table.ix[s_, :]
		
		if s_ in self.disallowed_actions:
			for excluded_action in self.disallowed_actions[s_]:
				del s_rewards[excluded_action]
		
		if s_ != 'terminal':
			q_target = r + self.gamma * s_rewards.max()
		else:
			q_target = r  # next state is terminal
			
		# update
		self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

	def check_state_exist(self, state):
		if state not in self.q_table.index:
			# append new state to q table
			self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))



class ProtossAgent(base_agent.BaseAgent):
	def __init__(self):
		super(ProtossAgent, self).__init__()
		
		self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
		
		self.previous_action = None
		self.previous_state = None
		
		self.move_number = 0
		
		if os.path.isfile(DATA_FILE + '.gz'):
			self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')
		
		self.attack_coordinates = None

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

	def select_available_probe(self, obs):
		probes = self.get_units_by_type(obs, units.Protoss.Probe)
		if len(probes) > 0:
			probe = random.choice(probes)
			return actions.FUNCTIONS.select_point("select_all_type", (probe.x, probe.y))

	def build_pylon(self, obs):
		if self.unit_type_is_selected(obs, units.Protoss.Probe):
			if self.can_do(obs, actions.FUNCTIONS.Build_Pylon_screen.id):
				x = random.randint(0, 83)
				y = random.randint(0, 83)
				return actions.FUNCTIONS.Build_Pylon_screen("now", (x, y))
		return self.select_available_probe(obs)

	def build_gateway(self, obs):
		if self.unit_type_is_selected(obs, units.Protoss.Probe):
			if self.can_do(obs, actions.FUNCTIONS.Build_Gateway_screen.id):
					x = random.randint(0, 83)
					y = random.randint(0, 83)
					return actions.FUNCTIONS.Build_Gateway_screen("now", (x, y))
			return self.select_available_probe(obs)

	def select_army(self, obs):
		if self.can_do(obs, actions.FUNCTIONS.select_army.id):
			return actions.FUNCTIONS.select_army("select")

	def attack_enemy(self, obs):
		if self.unit_type_is_selected(obs, units.Protoss.Zealot):
			if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
				return actions.FUNCTIONS.Attack_minimap("now",
				self.attack_coordinates)
		return self.select_army(obs)

	def step(self, obs):
		super(ProtossAgent, self).step(obs)
		
		if obs.last():
			reward = obs.reward
			self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')
			self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
			self.previous_action = None
			self.previous_state = None
			self.move_number = 0
			
			return actions.FunctionCall(_NO_OP, [])
		
		if obs.first():
			agent_y, agent_x = (obs.observation.feature_minimap.player_relative == 
			features.PlayerRelative.SELF).nonzero()

			agent_xmean = agent_x.mean()
			agent_ymean = agent_y.mean()
			
			if agent_xmean <= 31 and agent_ymean <= 31:
				self.attack_coordinates = (49, 49)
			else:
				self.attack_coordinates = (12, 16)

		gateways = self.get_units_by_type(obs, units.Protoss.Gateway)
		zealots = self.get_units_by_type(obs, units.Protoss.Zealot)
		pylon = self.get_units_by_type(obs, units.Protoss.Pylon)
		free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)

		if len(zealots) >= 12:
			return self.attack_enemy(obs)

		if free_supply < 15:
			return self.build_pylon(obs)

		if len(gateways) <= 2:
			return self.build_gateway(obs)

		if self.can_do(obs, actions.FUNCTIONS.Train_Zealot_quick.id):
			return actions.FUNCTIONS.Train_Zealot_quick("now")

		if len(gateways) > 0 and not self.unit_type_is_selected(obs, units.Protoss.Gateway):
			gateway = random.choice(gateways)
			return actions.FUNCTIONS.select_point("select_all_type", (gateway.x, gateway.y))

		return actions.FUNCTIONS.no_op()


def main(unused_argv):
	agent = ProtossAgent()
	try:
		while True:
			with sc2_env.SC2Env(map_name="Catalyst", players=[sc2_env.Agent(sc2_env.Race.protoss),
			sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.easy)],
			agent_interface_format=features.AgentInterfaceFormat(
			feature_dimensions=features.Dimensions(screen=84, minimap=64),
			use_feature_units=True), step_mul=16, game_steps_per_episode=0, visualize=True, save_replay_episodes=1, replay_dir='E:\Program Files (x86)\StarCraft II\Replays') as env:

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