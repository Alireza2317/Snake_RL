import os
import json
from matplotlib import pyplot as plt
from snake import SnakeGame, SnakeGameGUI, Direction
from agent import Agent


class Trainer:
	def __init__(
			self,
			*,
			resume: bool,
			params_subdir: str,
			episodes: int,
			hidden_layers_structure: list[int],
			activations: str | list[str],
			learning_rate: float,
			buffer_max_capacity: int,
			batch_size: int,
			gamma: float,
			epsilon_decay_rate: float,
			init_xavier: bool,
			render: bool = False,
	) -> None:
		# put all files inside the params parent directory
		self.params_dir = os.path.join('./params/', params_subdir)
		if not os.path.exists(self.params_dir):
			os.makedirs(self.params_dir)

		self.agent = Agent(
			files_savepath=self.params_dir,
			train_mode=True,
			hidden_layers_structure=hidden_layers_structure,
			activations=activations,
			learning_rate=learning_rate,
			buffer_max_capacity=buffer_max_capacity,
			batch_size=batch_size,
			gamma=gamma,
			epsilon_decay_rate=epsilon_decay_rate,
			init_xavier=init_xavier
		)

		if resume:
			# make the agent use the saved parameters instead of random values
			self.agent.load_params()

		self.game = SnakeGameGUI() if render else SnakeGame()
		self.num_episodes = episodes

		# number of points to plot
		self.log_freqency = 10 if episodes >= 100 else max(1, episodes // 10)


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


	def train(self, constant_alpha: bool, alpha_decay_rate: float, verbose: bool = True):
		total_reward = 0
		avg_rewards = []
		surviveds = []
		foods_eaten = []
		episodes = []

		for episode in range(1, self.num_episodes+1):
			if isinstance(self.game, SnakeGameGUI):
				self.game.text = f'{episode=}'
			#print(f'before: {self.agent.q_vals}')
			survived, episode_reward, food_score = self.train_step()
			total_reward += episode_reward

			# updating the agent with the replay buffer
			# also decaying epsilon
			# and save parameters every episode
			self.update_agent()
			#print(f'after: {self.agent.q_vals}')

			if verbose and (episode % self.log_freqency) == 0:
				avg_reward = total_reward / episode
				self.log_and_plot(
					episode, total_reward, avg_reward, survived, food_score,
					episodes, avg_rewards, surviveds, foods_eaten
				)

			# reduce learning rate if necessary
			if not constant_alpha:
				self.agent.alpha = max(1e-4, self.agent.alpha * alpha_decay_rate)

		if verbose:
			self.final_plot(episodes, avg_rewards, surviveds, foods_eaten)


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
			food_score: int,
			episodes: list[int],
			avg_rewards: list[float],
			surviveds: list[int],
			foods_eaten: list[int]
	):
		print(
			f'Episode {episode: >4}:',
	  	 	f'total_reward={total_reward:.2f},',
	 	   	f'avg_reward={avg_reward:.2f},',
	 	   	f'foods_eaten={food_score},',
	  	  	f'steps_survived={survived: >3}, epsilon={self.agent.epsilon:.2f},',
			f'lr={self.agent.alpha:.4f},',
			f'test_acc={self.agent.test_agent()*100:.1f}%',
	   		sep=' '
		)
		episodes.append(episode)
		avg_rewards.append(avg_reward)
		surviveds.append(survived)
		foods_eaten.append(food_score)

		self.live_plot(episodes, avg_rewards)


	def live_plot(self, x, y):
		plt.ion()
		plt.clf()

		plt.xlabel('episode')
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
		foods_eaten: list[int]
	):
		plt.show()
		plt.ioff()
		plt.figure(2)
		plt.clf()

		plt.plot(
			episodes, avg_rewards,
			episodes, surviveds,
			episodes, foods_eaten
		)
		plt.legend(['average rewards', 'steps survived', 'foods eaten'])
		plt.show()


	def save_configs(self, configs: dict):
		filepath = os.path.join(self.params_dir, 'configs.json')
		with open(filepath, 'w') as json_file:
			json.dump(configs, json_file, indent='\t')


def play(agent: Agent):
	game = SnakeGameGUI()

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


# default configs
configs = {
	'episodes': 300,

	'render': False,
	'verbose': True,
	'resume': False,

	'alpha': 1e-2,
	'hidden_layers_structure': [256, 256],
	'activations': 'relu',

	'batch_size': 1024,
	'buffer_max_capacity': 10_000,

	'epsilon_decay_rate': 0.98,
	'gamma': 0.9,
	'constant_alpha': False,
	'alpha_decay_rate': 0.985,

	'params_subdir': 'default',

	'init_xavier': True
}


if __name__ == '__main__':
	trainer = Trainer(
		resume=configs['resume'],
		params_subdir=configs['params_subdir'],
		episodes=configs['episodes'],
		render=configs['render'],
		learning_rate=configs['alpha'],
		hidden_layers_structure=configs['hidden_layers_structure'],
		activations=configs['activations'],
		batch_size=configs['batch_size'],
		buffer_max_capacity=configs['buffer_max_capacity'],
		gamma=configs['gamma'],
		epsilon_decay_rate=configs['epsilon_decay_rate'],
		init_xavier=configs['init_xavier']
	)

	trainer.train(
		constant_alpha=configs['constant_alpha'], alpha_decay_rate=configs['alpha_decay_rate'],
		verbose=configs['verbose']
	)
	trainer.save_configs(configs=configs)

	play(trainer.agent)