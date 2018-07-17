from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random

class ProtossAgent(base_agent.BaseAgent):
	def __init__(self):
		super(ProtossAgent, self).__init__()


def step(self, obs):
	super(ProtossAgent, self).step(obs)
	
	
	return actions.FUNCTIONS.no_op()

def main(unused_argv):
	agent = ProtossAgent()
	try:
		while True:
			with sc2_env.SC2Env(map_name="Catalyst", players=[sc2_env.Agent(sc2_env.Race.protoss),
			sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.easy)],
			agent_interface_format=features.AgentInterfaceFormat(
			feature_dimensions=features.Dimensions(screen=84, minimap=64),
			use_feature_units=True), step_mul=16, game_steps_per_episode=0, visualize=True) as env:

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