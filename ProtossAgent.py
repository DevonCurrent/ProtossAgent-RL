from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random

class ProtossAgent(base_agent.BaseAgent):
	def __init__(self):
		super(ProtossAgent, self).__init__()
		
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