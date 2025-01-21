import os
import json
import random
import numpy as np
from dataclasses import dataclass, field
from matplotlib import pyplot as plt
from snake import SnakeGame, SnakeGameGUI, Direction
from agent import Agent

random.seed(23)
np.random.seed(23)


@dataclass
class TrainerConfig:
	params_subdir: str = 'default'
	episodes: int = 100
	hidden_layers_structure: list[int] = field(default_factory=lambda: [256, 256])
	activations: str | list[str] = 'relu'
	learning_rate: float = 1e-3
	buffer_max_capacity: int = 10_000
	batch_size: int = 1024
	gamma: float = 0.9
	epsilon_decay_rate: float = 0.98
	init_xavier: bool = True
	constant_lr: bool = True
	lr_decay_rate: float = 0.985

	resume: bool = False
	render: bool = False
	verbose: bool = True

	def __dict__(self):
		return {
			'params_subdir': self.params_subdir,
			'episodes': self.episodes,
			'hidden_layers_structure': self.hidden_layers_structure,
			'activations': self.activations,
			'learning_rate': self.learning_rate,
			'buffer_max_capacity': self.buffer_max_capacity,
			'batch_size': self.batch_size,
			'gamma': self.gamma,
			'epsilon_decay_rate': self.epsilon_decay_rate,
			'init_xavier': self.init_xavier,
			'constant_lr': self.constant_lr,
			'lr_decay_rate': self.lr_decay_rate
		}


class Trainer:
	def __init__(
			self,
			config: TrainerConfig
	) -> None:
		""" Initializes all the necessary configurations from the input config parameter. """

		self.config: TrainerConfig = config

		# put all files inside the params parent directory
		self.params_dir = os.path.join('./params/', config.params_subdir)
		if not os.path.exists(self.params_dir):
			os.makedirs(self.params_dir)

		self.agent = Agent(
			files_savepath=self.params_dir,
			train_mode=True,
			hidden_layers_structure=config.hidden_layers_structure,
			activations=config.activations,
			learning_rate=config.learning_rate,
			buffer_max_capacity=config.buffer_max_capacity,
			batch_size=config.batch_size,
			gamma=config.gamma,
			epsilon_decay_rate=config.epsilon_decay_rate,
			init_xavier=config.init_xavier
		)

		if config.resume:
			# make the agent use the saved parameters instead of random values
			self.agent.load_params()

		self.game = SnakeGameGUI() if config.render else SnakeGame()
		self.num_episodes = config.episodes

		# number of points to plot
		self.log_freqency = 10 if config.episodes >= 100 else max(1, config.episodes // 10)


	def train_step(self) -> tuple[int, float, int]:
		episode_reward: float = 0

		state = self.game.get_state()

		# each loop corresponds to one whole game
		while not self.game.game_over:
			# agent chooses an action
			ai_action: Direction = self.agent.choose_action(state)
			# applying the action to the actual game
			self.game.action = ai_action

			next_state, reward = self.game.step()

			episode_reward += reward

			# call the short train on this current experience and save it to buffer
			self.agent.update_short(
				state, ai_action.value, reward, next_state, self.game.game_over
			)

			self.agent.replay_buffer.add(
				state, ai_action.value, reward, next_state, self.game.game_over
			)

			# transition the states
			state = next_state

		# since the game is over
		survived = self.game.survival_score
		food_score = self.game.food_score

		# reset the game
		self.game.reset()

		return survived, episode_reward, food_score


	def train(self):
		total_reward = 0
		total_food_score = 0
		avg_rewards = []
		surviveds = []
		avg_foods_eaten = []
		episodes = []

		for episode in range(1, self.num_episodes+1):
			if isinstance(self.game, SnakeGameGUI):
				self.game.text = f'{episode=}'

			survived, episode_reward, food_score = self.train_step()
			total_reward += episode_reward
			total_food_score += food_score

			# updating the agent with the replay buffer
			# also decaying epsilon
			# and save parameters every episode
			self.update_agent()

			if self.config.verbose and (episode % self.log_freqency) == 0:
				avg_reward = total_reward / episode
				avg_food_score = total_food_score / episode
				self.log_and_plot(
					episode, total_reward, avg_reward, survived, avg_food_score,
					episodes, avg_rewards, surviveds, avg_foods_eaten
				)

			# reduce learning rate if necessary
			decay_rate = self.config.lr_decay_rate
			if not self.config.constant_lr:
				self.agent.lr = max(1e-4, self.agent.lr * decay_rate)

		if self.config.verbose:
			self.final_plot(episodes, avg_rewards, surviveds, avg_foods_eaten)

		self.save_configs()


	def update_agent(self):
		self.agent.update_with_memory()
		self.agent.decay_epsilon()
		self.agent.save_params()


	def log_and_plot(
			self,
			episode: int,
			total_reward: float,
			avg_reward: float,
			survived: int,
			avg_food_score: float,
			episodes: list[int],
			avg_rewards: list[float],
			surviveds: list[int],
			avg_foods_eaten: list[float]
	):
		print(
			f'Episode {episode:>4}:',
	  	 	f'total_reward={total_reward:>9.2f},',
	 	   	f'avg_reward={avg_reward:>7.2f},',
	 	   	f'avg_food={avg_food_score:>6.1f},',
	  	  	f'survived={survived:>4}, eps={self.agent.epsilon:.2f},',
			f'lr={self.agent.lr:.5f},',
			f'test_acc={self.agent.test_agent()*100:.1f}%',
	   		sep=' '
		)
		episodes.append(episode)
		avg_rewards.append(avg_reward)
		surviveds.append(survived)
		avg_foods_eaten.append(avg_food_score)

		self.live_plot(episodes, avg_rewards)


	def live_plot(self, x, y):
		plt.ion()
		plt.clf()

		plt.xlabel('episode')
		plt.ylabel('average reward')
		plt.title('progress')

		plt.xlim(0, x[-1]+len(x)//5)
		plt.plot(x, y)
		plt.text(x[-1], y[-1], f'{y[-1]:.3f}')

		plt.pause(0.01)
		plt.show(block=False)


	def final_plot(
		self,
		episodes: list[int],
		avg_rewards: list[float],
		surviveds: list[int],
		avg_foods_eaten: list[int]
	):
		plt.show()
		plt.ioff()
		plt.figure(2)
		plt.clf()

		plt.plot(
			episodes, avg_rewards,
			episodes, surviveds,
			episodes, avg_foods_eaten
		)
		plt.legend(['average rewards', 'steps survived', 'average foods eaten'])
		plt.show()


	def save_configs(self):
		config = self.config.__dict__()

		filepath = os.path.join(self.params_dir, 'configs.json')

		with open(filepath, 'w') as json_file:
			json.dump(config, json_file, indent='\t')


def play(agent: Agent):
	game = SnakeGameGUI()
	game.fps = 12
	agent.epsilon = 0
	agent.load_params()

	n_games = 1
	while True:
		state = game.get_state()
		game.action = agent.choose_action(state)

		game.text = f'game number: {n_games}'
		game.step()

		if game.game_over:
			game.reset()
			n_games += 1



if __name__ == '__main__':
	default_configs = TrainerConfig()

	custom_configs = TrainerConfig(
		params_subdir='custom',
		episodes=280,
		constant_lr=True,
		learning_rate=1e-2
	)

	trainer = Trainer(config=custom_configs)

	trainer.train()

	play(trainer.agent)
