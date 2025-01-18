from snake import SnakeGame, SnakeGameGUI, Direction
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
	rewards_average = [0]
	steps_survived_list = [0]
	episodes_plot = [1]


	if resume:
		agent.load_params()

	# number of points to plot
	NUM_POINTS = episodes if episodes < 100 else 100

	for episode in range(1, episodes+1):
		steps_survived: int = 0

		state = game.get_state()

		# each loop corresponds to one whole game
		done = False
		while not done:
			steps_survived += 1

			# agent chooses an action
			ai_action: Direction = agent.choose_action(state)

			# applying the action to the actual game
			game.action = ai_action
			next_state, reward, done = game.step()

			total_reward += reward

			# convert the action from Direction to int index
			action: int = game.action.value

			# call the short train on this current experience and save it to buffer
			agent.update_short(state, action, reward, next_state, done)
			agent.replay_buffer.add(state, action, reward, next_state, done)

			# transition the states
			state = next_state

		# the game is over so done=True here
		game.reset()

		# updating the agent with the replay buffer
		agent.update_with_memory()

		# lower the exploration probability after each episode
		agent.decay_epsilon()

		# printing and plotting some data
		if episode % (episodes // NUM_POINTS) == 0:
			avg_reward: float = total_reward / episode
			episodes_plot.append(episode)
			steps_survived_list.append(steps_survived)
			rewards_average.append(avg_reward)

			print(
				f'Episode {episode}:',
				f'{total_reward=:.3f},',
				f'avg_reward={avg_reward:.3f},',
				f'{steps_survived=}, epsilon={agent.epsilon:.3f}',
				sep=' '
			)

			live_plot(episodes_plot, rewards_average)

		# save parameters every episode
		agent.save_params()


	# plot the whole data in the end of training
	plt.clf()
	plt.plot(
		episodes_plot, rewards_average,
		episodes_plot, steps_survived_list
	)
	plt.legend(['average rewards', 'steps survived'])
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
	train_agent(resume=False, episodes=2000, render=False)