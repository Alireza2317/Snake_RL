from nn import NeuralNetwork
import numpy as np
import random
from snake import Direction
from collections import deque

random.seed(23)

class ReplayBuffer:
	def __init__(self, capacity: int) -> None:
		self.buffer = deque(maxlen=capacity)


	def add(self, state, action, reward, next_state, done) -> None:
		self.buffer.append(
			(state, action, reward, next_state, done)
		)


	def sample(self, batch_size: int) -> list[tuple]:
		return random.sample(self.buffer, batch_size)


	def __len__(self) -> int:
		return len(self.buffer)


class Agent:
	def __init__(self, train_mode: bool = False) -> None:
		# creating the neural network
		self.network = NeuralNetwork(
			layers_structure=[14, 48, 48, 4],
			activations='tanh'
		)

		self.network.weights = [
			np.random.uniform(
				-np.sqrt(6 / (shape[0]+shape[1])),
				np.sqrt(6 / (shape[0]+shape[1])),
				size=shape
			)
			for shape in self.network._weights_shapes
		]

		# discount factor
		self.gamma: float = 0.91

		# learning rate
		self.alpha: float = 3e-3

		# epsilon-greedy policy for explore-exploit trade-off
		# should decay over training to lower the exploration
		if train_mode:
			self.epsilon: float = 2
		else:
			self.epsilon: float = 0


	def choose_action(self, state: list[float]) -> Direction:
		"""
		chooses and returns a move based on the current state
		the action is in the form of a Direction enum
		"""

		actions: list[Direction] = list(Direction)

		# with probability epsilon, pick a random action
		if random.random() < self.epsilon:
			return random.choice(actions)

		# otherwise pick the action with the highest Q(a, s)
		else:
			# these would be a list with four elements
			# each represents one direction
			q_values = self.network.predict_output(state)
			return actions[np.argmax(q_values)]


	def decay_epsilon(self) -> None:
		"""
			decay epsilon over time to minimize exploration
		"""
		self.epsilon = max(0.1, self.epsilon * 0.998)


	def update(
			self,
			state: list[float],
			action: int,
			reward: float,
			next_state: list[float],
			done: bool
	) -> None:
		"""
			updates the neural network using the Bellman equation.

			@param state: the current state of the game
			@param action: the action that the agent picked for the state
			@param reward: the reward that the agent received for the action
			@param next_state: the state after the agent's action
			@param done: if the game is over or not
		"""
		q_values = self.network.predict_output(state)
		target_q_values = q_values.copy()

		if done:
			# game is over
			target_q_values[action] = reward
		else:
			# using Bellman equation to compute the target
			next_q_values = self.network.predict_output(next_state)
			target_q_values[action] = reward + self.gamma * np.max(next_q_values)

		# train the network
		state = np.array(state).reshape((-1, 1))
		target_q_values = np.array(target_q_values).reshape((-1, 1))
		self.network.train(
			x_train=state,
			y_train=target_q_values,
			learning_rate=self.alpha,
			decay_rate=0.99999,
			constant_lr=False,
			batch_size=1,
			number_of_epochs=2,
			verbose=False
		)
