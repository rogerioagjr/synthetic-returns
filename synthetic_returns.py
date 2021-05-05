import numpy as np
import torch
import torch.nn as nn

class Environment(object):
    def __init__(self, n_actions, state_size):
        """
        Creates a simulated environment for the agent.
        :param n_actions: the finite number of possible actions available throughout the entire simulation.
        :param state_size: the shape of the PyTorch Tensor that represents the parameters of the simulation state that
                           are visible to the agent
        """
        raise NotImplementedError("Environment Constructor not implemented!")

    def step(self, action):
        """
        Simulates a time step in the environment
        :param action: action performed by the agent. Integer between 0 and n_actions-1
        :return: (state, reward, done, info).
                 state: new state of the environment after the agent performs action, if simulation is not done
                 reward: immediate reward obtained by after the agent performs action
                 done: True if simulation is over after the agent performs action. False otherwise.
                 info: Additional information or variables that might be useful
        """
        raise NotImplementedError("Environment step() not implemented!")

    def get_state(self):
        """
        :return: The current state of the simulation, a PyTorch Tensor.
        """
        raise NotImplementedError("Environment get_state() not implemented!")


class Agent(object):
    def __init__(self, n_actions):
        """
        Creates a new agent that can perform actions in an Environment
        :param n_actions: number of actions available to the Agent
        """
        raise NotImplementedError("Agent Constructor not implemented!")

    def select_action(self, state):
        """
        :param state: the parameters of the current state of the simulated environment that are visible to the agent
        :return: the action that the agent will perform, an integer between 0 and n_actions-1
        """
        raise NotImplementedError("Agent select_action() not implemented!")


class StandardAgent(Agent):
    def __init__(self, n_actions):

        self.n_actions = n_actions

        #TODO: implement Standard Agent. Must have LSTM as internal state to capture multiple steps

        super(Agent, self).__init__(n_actions)


class SyntheticReturnsAgent(Agent):
    def __init__(self, n_actions, episode_length, future_reward_network, immediate_reward_network, future_reward_gate):

        self.n_actions = n_actions
        self.episode_length = episode_length
        self.future_reward_network = future_reward_network
        self.immediate_reward_network = immediate_reward_network
        self.future_reward_gate = future_reward_gate

        super(Agent, self).__init__(n_actions)
