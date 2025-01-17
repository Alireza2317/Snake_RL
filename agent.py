from nn import NeuralNetwork
import numpy as np
import random
from snake import Direction
from collections import deque
import os

PARAMETERS_FILE = 'nn_params.txt'

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
	def __init__(
		self,
		train_mode: bool = False,
		buffer_capacity = 10000,
		batch_size = 256,
		parameters_filename = PARAMETERS_FILE
	) -> None:
		self.replay_buffer = ReplayBuffer(buffer_capacity)
		self.batch_size = batch_size

		# creating the neural network
		self.network = NeuralNetwork(
			layers_structure=[14, 256, 4],
			activations='relu'
		)

		#self.network.weights = [
		#	np.random.uniform(
		#		-np.sqrt(6 / (shape[0]+shape[1])),
		#		np.sqrt(6 / (shape[0]+shape[1])),
		#		size=shape
		#	)
		#	for shape in self.network._weights_shapes
		#]

		# discount factor
		self.gamma: float = 0.91

		# learning rate
		self.alpha: float = 1e-3

		# epsilon-greedy policy for explore-exploit trade-off
		# should decay over training to lower the exploration
		if train_mode:
			self.epsilon: float = 1.7
		else:
			self.epsilon: float = 0

		self.parameters_filename = parameters_filename
		self.hyperparameters_filename = 'hparams.txt'
		self.params_dir = './params'


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


	def save_params(self):
		if not os.path.exists(self.params_dir):
			os.mkdir(self.params_dir)

		filepath = os.path.join(self.params_dir, self.parameters_filename)
		self.network.save_parameters_to_file(filepath)

		# saving epsilon
		# can save other hyperparameters later, if needed: TODO
		filepath = os.path.join(self.params_dir, self.hyperparameters_filename)
		with open(filepath, 'w') as file:
			file.write(f'{self.epsilon}')


	def load_params(self) -> bool:
		model_params_file = os.path.join(self.params_dir, self.parameters_filename)
		hyperparams_file = os.path.join(self.params_dir, self.hyperparameters_filename)

		if not (os.path.exists(model_params_file) and os.path.exists(hyperparams_file)):
			return False

		# if both files exist, load them
		self.network.load_params_from_file(model_params_file)

		with open(hyperparams_file, 'r') as file:
			eps = float(file.read().strip())
			self.epsilon = eps

		return True


	def update_short(
			self,
			state: list[float],
			action: int,
			reward: float,
			next_state: list[float],
			done: bool
	) -> None:
		q_values = self.network.predict_output(state)
		target_qs = q_values.copy()

		if not done:
			next_q_values = self.network.predict_output(next_state)
			target_qs[action] = reward + self.gamma * np.max(next_q_values)
		else:
			target_qs[action] = reward

		state = np.array(state).reshape((-1, 1))
		target_qs = np.array(target_qs).reshape((-1, 1))

		self.network.train(
			x_train=state,
			y_train=target_qs,
			learning_rate=self.alpha,
			constant_lr=True,
			number_of_epochs=1,
			batch_size=1,
			verbose=False
		)


	def update_with_memory(self) -> None:
		"""
			updates the neural network using the Bellman equation.

			@param state: the current state of the game
			@param action: the action that the agent picked for the state
			@param reward: the reward that the agent received for the action
			@param next_state: the state after the agent's action
			@param done: if the game is over or not
		"""
		STATES_LEN = 14
		ACTIONS_LEN = 4


		# train only if there are at least batch_size experiences
		if len(self.replay_buffer) < self.batch_size:
			return

		batch = self.replay_buffer.sample(self.batch_size)

		# now seperate all the sections in the buffer
		states, actions, rewards, next_states, dones = zip(*batch)

		# convert them all to numpy arrays
		states = np.array(states)
		actions = np.array(actions)
		rewards = np.array(rewards)
		next_states = np.array(next_states)
		dones = np.array(dones)

		states = states.reshape((-1, STATES_LEN))
		next_states = next_states.reshape((-1, STATES_LEN))

		next_q_values = np.array(
			[self.network.predict_output(ns.reshape((-1, 1))) for ns in next_states]
		).reshape((-1, ACTIONS_LEN))

		max_next_q_values = np.max(next_q_values, axis=1).reshape((-1, 1))

		conditions = (1 - dones).reshape((-1, 1))

		targets = (rewards.reshape((-1, 1)) + self.gamma * max_next_q_values * conditions)

		targets = targets.reshape((-1, 1))


		q_values = np.array(
			[self.network.predict_output(s.reshape((-1, 1))) for s in states]
		).reshape((-1, ACTIONS_LEN))

		for i_batch, action in enumerate(actions):
			q_values[i_batch, action] = targets[i_batch]

		self.network.train(
			x_train=states.T,
			y_train=q_values.T,
			learning_rate=self.alpha,
			constant_lr=True,
			#decay_rate=0.99999,
			batch_size=self.batch_size,
			number_of_epochs=1,
			verbose=False
		)
