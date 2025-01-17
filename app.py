from snake import SnakeGame, SnakeGameGUI, Direction, PARAMETERS_FILE
from agent import Agent
from matplotlib import pyplot as plt


def live_plot(x, y):
	plt.clf()

	plt.xlabel('episode')
	plt.title('progress')

	plt.xlim(0, x[-1]+len(x)//5)
	plt.plot(x, y)
	plt.text(x[-1], y[-1], f'{y[-1]:.3f}')

	plt.show(block=False)
	plt.pause(0.0001)


def train_agent(resume: bool = False, episodes: int = 20, render: bool = False):
	agent = Agent(train_mode=True)
	if render:
		game = SnakeGameGUI()
		game.fps = 40
	else:
		game = SnakeGame()

	total_reward: float = 0
	episode_rewards = []
	total_rewards = []
	rewards_average = [0]
	steps_survived_list = [0]
	episodes_plot = [1]


	if resume:
		try:
			with open('params.txt', 'r') as file:
				eps = float(file.read().strip())
				agent.epsilon = eps

			agent.network.load_params_from_file(PARAMETERS_FILE)
		except FileNotFoundError:
			print('No trained file was found, training from scratch!')

	# number of points to plot
	NUM_POINTS = episodes if episodes < 100 else 100

	for episode in range(1, episodes+1):
		steps_survived: int = 0
		episode_reward: float = 0

		game.reset()

		state = game.get_state()

		done = False

		while not done:
			steps_survived += 1
			ai_action: Direction = agent.choose_action(state)
			game.action = ai_action

			next_state, reward, done = game.step()

			action: int = game.action.value

			# here should call the short train
			agent.update_short(state, action, reward, next_state, done)

			# save the experience to the buffer
			agent.replay_buffer.add(state, action, reward, next_state, done)

			episode_reward += reward

			state = next_state

		# here should call the long train
		agent.update_with_memory()

		agent.decay_epsilon()
		total_reward += episode_reward


		if episode % (episodes // NUM_POINTS) == 0:
			episodes_plot.append(episode)
			episode_rewards.append(episode_reward)
			total_rewards.append(total_reward)
			steps_survived_list.append(steps_survived)
			rewards_average.append(total_reward / episode)
			print(
				f'Episode {episode}:',
				f'{total_reward=:.1f}, {episode_reward=:.1f}',
				f'{steps_survived=}, {agent.epsilon=:.3f}',
				sep=' '
			)

		# save every episode
		agent.network.save_parameters_to_file(PARAMETERS_FILE)
		with open('params.txt', 'w') as file:
			file.write(f'{agent.epsilon}')

		live_plot(episodes_plot, rewards_average)

	plt.plot(episodes_plot, rewards_average)
	plt.show()


def play():
	game = SnakeGameGUI()
	agent = Agent(train_mode=False)
	agent.network.load_params_from_file('nn_params.txt')

	while True:
		state = game.get_state()
		game.action = agent.choose_action(state)

		_, _, done = game.step()

		if done:
			game.reset()


if __name__ == '__main__':
	train_agent(resume=False, episodes=50, render=False)