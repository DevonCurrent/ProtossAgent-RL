# ProtossAgent-RL

This project is designed to help me familiarize myself with machine learning, as this is my first attempt. This agent is created using the [PySC2 API](https://github.com/deepmind/pysc2) created by DeepMind. Based off of [skjb](https://github.com/skjb/pysc2-tutorial)'s tutorials for implementing PySC2 with Q-learning.

### Installation
In order to run this you will need Starcraft 2, the PySC2 package, and [maps](https://github.com/Blizzard/s2client-proto#downloads) for the agent to run on (this agent was designed with Catalyst in mind).

After cloning this file, add it to the agents folder of the PySC2 package.

### Currently in progress
* Implementing all units and buildings into the agents actions.
* Defeating an A.I on easy.
* Using Q-Learning to reward the agent.
* Having the agent expand to a second base.
* Making the agent flexible against any map or enemy race.


### Goals for this project
* Have the agent beat an A.I on hard.
* Use Q-learning to teach the agent to make smart decisions.
* Teach the agent to use different strategies at random to go off of (rushing, expanding, cloak/air units)
