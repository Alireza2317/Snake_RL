from snake import SnakeGame, SnakeGameGUI, Direction, PARAMETERS_FILE
from agent import Agent
from matplotlib import pyplot as plt

def train_agent(resume: bool = False, episodes: int = 20, render: bool = False):
	agent = Agent(train_mode=True)
	if render:
		game = SnakeGameGUI()
		game.fps = 40
	else:
		game = SnakeGame()

	total_reward: float = 0
	episode_rewards = []
	steps_survived_list = []


	if resume:
		try:
			with open('params.txt', 'r') as file:
				eps = float(file.read().strip())
				agent.epsilon = eps

			agent.network.load_params_from_file(PARAMETERS_FILE)
		except FileNotFoundError:
			print('No trained file was found, training from scratch!')


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
			agent.update(state, action, reward, next_state, done)
			episode_reward += reward

			state = next_state

		agent.decay_epsilon()
		total_reward += episode_reward


		if episode%20 == 0:
			episode_rewards.append(episode_reward)
			steps_survived_list.append(steps_survived)
			print(f'Episode {episode}:\t{total_reward=:.1f}, {episode_reward=:.1f}, {steps_survived=}, {agent.epsilon=:.3f}')

	agent.network.save_parameters_to_file(PARAMETERS_FILE)
	with open('params.txt', 'w') as file:
		file.write(f'{agent.epsilon}')

	plt.plot(list(range(1, 1+episodes//20)), episode_rewards, list(range(1, 1+episodes//20)), steps_survived_list)
	plt.legend(['ep rewards', 'survived'])
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
	#play()