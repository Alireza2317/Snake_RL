from nn import NeuralNetwork
import numpy as np
import random
from snake import Direction, NUM_STATES, NUM_ACTIONS
from collections import deque
import os

PARAMETERS_FILE = 'nn_params.txt'
MAX_CAPACITY = 10_000

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
		hidden_layers_structure: list[int] = [320],
		activations: str | list[str] = 'tanh',
		learning_rate: float = 2e-3,
		batch_size: int = 1024,
		parameters_filename: str = PARAMETERS_FILE,
		gamma: float = 0.9,
		epsilon_decay_rate: float = 0.999,
		init_xavier: bool = False,
	) -> None:
		self.replay_buffer = ReplayBuffer(MAX_CAPACITY)
		self.batch_size = batch_size

		# creating the neural network
		structure: list[int] = []

		# number of inputs = size of the state of the game
		structure.append(NUM_STATES)

		for nodes in hidden_layers_structure:
			structure.append(nodes)

		# number of outputs = size of the actions of the game
		structure.append(NUM_ACTIONS)

		self.network = NeuralNetwork(layers_structure=structure, activations=activations)

		# Xavier initializer
		if init_xavier:
			self.network.weights = [
				np.random.uniform(
					-np.sqrt(6 / (shape[0]+shape[1])),
					np.sqrt(6 / (shape[0]+shape[1])),
					size=shape
				)
				for shape in self.network._weights_shapes
			]

		# discount factor
		self.gamma: float = gamma

		# learning rate
		self.alpha: float = learning_rate

		# epsilon-greedy policy for explore-exploit trade-off
		# should decay over training to lower the exploration
		if train_mode:
			self.epsilon: float = 1
			self.minimum_epsilon = 0.1
			self.epsilon_decay_rate = epsilon_decay_rate
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
			# this would be a list with four elements
			# each element represents one direction
			q_values = self.network.predict_output(state)
			dir_binary = state[-4:]
			idx = np.argmax(dir_binary)
			snake_direction: Direction = actions[idx]

			max_idx = np.argmax(q_values)

			dir_u = snake_direction == Direction.UP and max_idx == Direction.DOWN.value
			dir_r = snake_direction == Direction.RIGHT and max_idx == Direction.LEFT.value
			dir_d = snake_direction == Direction.DOWN and max_idx == Direction.UP.value
			dir_l = snake_direction == Direction.LEFT and max_idx == Direction.RIGHT.value

			if dir_u or dir_r or dir_d or dir_l:
				# ignoring 180 degree turns
				q_values[max_idx] = -np.inf

			return actions[np.argmax(q_values)]


	def decay_epsilon(self) -> None:
		"""
			decay epsilon over time to minimize exploration
		"""
		self.epsilon = max(self.minimum_epsilon, self.epsilon * self.epsilon_decay_rate)


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


	def _update(self, states, actions, rewards, next_states, dones, short = True) -> None:
		"""
			updates the neural network using the Bellman equation.

			* all parameters can be in a form of a list of multiple data points
			@param states: the current state of the game
			@param actions: the action that the agent picked for the state
			@param rewards: the reward that the agent received for the action
			@param next_states: the state after the agent's action
			@param dones: if the game is over or not
		"""

		# convert all these to np arrays
		states = np.array(states)
		actions = np.array(actions)
		rewards = np.array(rewards)
		next_states = np.array(next_states)
		dones = np.array(dones)

		# if this is for a short train
		# we should adjust the shapes
		if short:
			batch_size = 1
			states = states.reshape((1, NUM_STATES))
			actions = actions.reshape((1, 1))
			rewards = rewards.reshape((1, 1))
			next_states = next_states.reshape((1, NUM_STATES))
			dones = dones.reshape((1, 1))
		else:
			batch_size = min(len(self.replay_buffer), self.batch_size)
			states = states.reshape((-1, NUM_STATES))
			next_states = next_states.reshape((-1, NUM_STATES))
			actions = actions.reshape((-1, 1))
			rewards = rewards.reshape((-1, 1))
			dones = dones.reshape((-1, 1))


		target_qs_all = []

		for i, done in enumerate(dones):
			s = states[i].reshape((-1, 1))
			q_values = self.network.predict_output(s)
			target_qs = q_values.copy()

			q_new = rewards[i].item()
			if not done:
				ns = next_states[i].reshape((-1, 1))
				next_q_values = self.network.predict_output(ns)
				q_new += self.gamma * np.max(next_q_values)

			target_qs[actions[i]] = q_new
			target_qs_all.append(target_qs)

		target_qs_all = np.array(target_qs_all).reshape((-1, NUM_ACTIONS))


		self.network.train(
			x_train=states.T,
			y_train=target_qs_all.T,
			learning_rate=self.alpha,
			constant_lr=True,
			number_of_epochs=1,
			batch_size=batch_size,
			verbose=False
		)


	def update_short(
			self,
			state: list[float],
			action: int,
			reward: float,
			next_state: list[float],
			done: bool
	) -> None:
		self._update(state, action, reward, next_state, done, short=True)


	def update_with_memory(self) -> None:
		buffer_size = len(self.replay_buffer)
		if buffer_size < self.batch_size:
			batch = self.replay_buffer.sample(buffer_size)
		else:
			batch = self.replay_buffer.sample(self.batch_size)

		states, actions, rewards, next_states, dones = zip(*batch)

		self._update(states, actions, rewards, next_states, dones, short=False)


	def q_vals(self) -> str:
		output_layer = self.network.layers[-1].reshape((NUM_ACTIONS, ))
		return str(output_layer)