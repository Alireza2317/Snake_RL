from snake import SnakeGame, SnakeGameGUI, Direction, Reward
from agent import Agent
from matplotlib import pyplot as plt

class Trainer:
	def __init__(self, episodes: int, render: bool = False, **kwargs):
		self.agent = Agent(train_mode=True, **kwargs)
		self.game = SnakeGameGUI() if render else SnakeGame()
		self.num_episodes = episodes

		# number of points to plot
		self.log_freqency = 100 if episodes >= 100 else max(1, episodes // 10)



	def train_step(self) -> tuple[int, float]:
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
		self.game.reset()

		return self.game.survival_score, episode_reward


	def train(self):
		total_reward = 0
		avg_rewards = []
		surviveds = []
		episodes = []

		for episode in range(1, self.num_episodes+1):
			survived, episode_reward = self.train_step()
			total_reward += episode_reward

			# updating the agent with the replay buffer
			# also decaying epsilon
			# and save parameters every episode
			self.update_agent()

			if (episode % self.log_freqency) == 0:
				avg_reward = total_reward / episode
				self.log_and_plot(
					episode, total_reward, avg_reward, survived,
					episodes, avg_rewards, surviveds
				)

		self.final_plot(episodes, avg_rewards, surviveds)


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
			episodes: list[int],
			avg_rewards: list[float],
			surviveds: list[int]
	):
		print(
			f'Episode {episode}:',
	  	 	f'total_reward={total_reward:.2f},',
	 	   	f'avg_reward={avg_reward:.2f},',
	  	  	f'steps_survived={survived}, epsilon={self.agent.epsilon:.2f}',
	   		sep=' '
		)

		episodes.append(episode)
		avg_rewards.append(avg_reward)
		surviveds.append(survived)

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
		surviveds: list[int]
	):
		plt.show()
		plt.ioff()
		plt.figure(2)
		plt.clf()

		plt.plot(
			episodes, avg_rewards,
			episodes, surviveds
		)
		plt.legend(['average rewards', 'steps survived'])
		plt.show()



def play():
	game = SnakeGameGUI()
	agent = Agent(train_mode=False)
	agent.load_params()

	n_games = 1
	while True:
		state = game.get_state()
		game.action = agent.choose_action(state)

		game.text = f'game number: {n_games}'
		_, _, done = game.step()

		if done:
			game.reset()
			n_games += 1


if __name__ == '__main__':
	trainer = Trainer(episodes=2, render=False)

	trainer.train()