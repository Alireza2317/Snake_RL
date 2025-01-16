import sys
import numpy as np
import pygame as pg
from random import choice, random
from copy import deepcopy
from nn import NeuralNetwork
from collections import namedtuple

# speed. the higher the fps, the faster the game
FPS = 3
# if set to true, will scale up the fps after eating a food
INCREMENT_SPEED = False
SCALE = 1.0065

# dimensions
WN: int = 10
HN: int = 10
BLOCK_SIZE = 50

PD = 50
# some padding outside the walls of the game. lower values might not allow
# the score text to have enough space!
WIDTH = WN * BLOCK_SIZE + 2*PD
HEIGHT = HN * BLOCK_SIZE + 2*PD

# snake
# this will determine the shape of the pixels, either round or square-shaped
# uncomment the one you desire
SHAPE = 'square'
#SHAPE = 'circle'

INITIAL_SIZE = 3

# colors, all in tuple format, (r, g, b)
BG_COLOR = (35, 35, 35)
SNAKE_HEAD_COLOR = (255, 204, 62)
SNAKE_COLOR = (54, 110, 156)
GRID_COLOR = (45, 45, 45)
WALL_COLOR = (220, 220, 240)
FOOD_COLOR = (239, 57, 57)

# fonts for texts
FONT_SIZE = 22
FONT_COLOR = (220, 220, 220)

PARAMETERS_FILE = 'nn_params.txt'

if  FPS <= 0:
	raise ValueError('FPS should be positive numbers!')
if INITIAL_SIZE >= WN-1:
	raise ValueError('Snake\'s INITIAL_SIZE is too high!')

# comment this if your screen is big enough
if WIDTH > 1920 or HEIGHT > 1080:
	raise ValueError('Consider reducing WN and HN or BLOCK_SIZE! Too big for most screens!')


Position = namedtuple('Position', ['x', 'y'])


class SnakeGame:
	"""
	handles all the logic of a snake game
	"""
	def __init__(self) -> None:
		self.reset()


	def reset(self) -> None:
		self.direction: str = 'r'
		self.action: str = 'r'

		# first element will be the head
		self.snake: list[Position] = [
			Position(i, 0)
			for i in range(INITIAL_SIZE, -1, -1)
		]

		# useful when growing the snake
		self._left_over: Position = self.snake[-1]

		# the world consists of 4 elements:
		# 'h': is the position of the snake's head(only one)
		# 's': is the position of the snake's body parts(can be more than one)
		# 'f': is the position of the food in the world
		# '': is the empty cells in the world
		self.world: list[list[str]] = [
			['' for col in range(WN)] for row in range(HN)
		]

		# creates self.food property
		self.generate_food()

		# update the world
		self.update_world()

		self.game_over = False

		self.score: int = 0


	def update_world(self) -> None:
		"""
		updates self.world based on self.snake and self.food positions
		"""
		for r in range(HN):
			for c in range(WN):
				pos = (c, r)
				if pos == self.food:
					self.world[r][c] = 'f'
				elif self.hit_position(pos):
					self.world[r][c] = 's'
				else:
					self.world[r][c] = ''
				if self.head == pos:
					self.world[r][c] = 'h'


	@property
	def size(self):
		return len(self.snake)


	@property
	def head(self) -> Position:
		return self.snake[0]


	def turn(self, dir_to_turn: str) -> None:
		if self.direction in ('u', 'd') and dir_to_turn in ('u', 'd'):
			return

		elif self.direction in ('r', 'l') and dir_to_turn in ('r', 'l'):
			return

		self.direction = dir_to_turn


	def grow(self) -> None:
		self.snake.append(self._left_over)


	def move(self) -> None:
		# remove the last body part and save it
		self._left_over = self.snake.pop()

		match self.direction:
			case 'r':
				new_head = Position(self.head.x+1, self.head.y)
			case 'l':
				new_head = Position(self.head.x-1, self.head.y)
			case 'd':
				new_head = Position(self.head.x, self.head.y+1)
			case 'u':
				new_head = Position(self.head.x, self.head.y-1)

		# all body parts are the same
		# except the first and the last Position (head and tail)
		self.snake.insert(0, new_head)


	def ate_food(self) -> bool:
		return (self.head == self.food)


	def hit_position(self, pos: Position | tuple) -> bool:
		return (pos in self.snake)


	def hit_self(self) -> bool:
		return (self.head in self.snake[1:])


	def hit_wall(self) -> bool:
		out_of_x = (self.head.x < 0 or self.head.x >= WN)
		out_of_y = (self.head.y < 0 or self.head.y >= HN)

		if out_of_x or out_of_y:
			return True

		return False


	def generate_food(self) -> None:
		valid_cells: list[Position] = []
		for r, row in enumerate(self.world):
				for c, cell in enumerate(row):
					pos = Position(x=c, y=r)
					if not hasattr(self, 'food'):
						if not self.hit_position(pos):
							valid_cells.append(pos)
					else:
						if cell == '':
							valid_cells.append(pos)

		self.food = choice(valid_cells)


	def is_world_full(self) -> bool:
		# check to see if there are any empty cells in the world
		for row in self.world:
			for cell in row:
				if cell == '':
					return False
		return True


	def is_world_snaked(self) -> bool:
		# best function name does not exist:
		for row in self.world:
			for cell in row:
				# if the cell is not the snake's head or body parts
				if cell not in ['h', 's']:
					return False

		return True


	def step(self) -> tuple[list[float], float, bool]:
		"""
		takes in an action and applies it to the game
		returns a tuple in this order:
			the game state after the snake's move
			reward
			game_over flag
		"""
		reward: float = 0

		self.turn(self.action)

		self.move()

		# snake grows if ate any food
		if self.ate_food():
			self.grow()
			self.score += 1
			reward = 1

			if not self.is_world_full():
				self.generate_food()


		# check collisions
		self.game_over = self.hit_self() or self.hit_wall()
		if self.game_over:
			#print('You lost!')
			reward = -1

		# update the world
		self.update_world()

		# check if the player won the game?!
		if self.is_world_snaked():
			# this will probably never happen in a real game!
			self.game_over = True # good game over!
			print('You won the snake game!')

		return self.get_state(), reward, self.game_over


	def pretty_print(self) -> None:
		for r, row in enumerate(self.world):
			for c, cell in enumerate(row):
				if cell == '':
					print('📦 ', end='')
				elif cell == 'h':
					print('🐍 ', end='')
				elif cell == 's':
					print('🟢 ', end='')
				elif cell == 'f':
					print('🍎 ', end='')
			print()

		print('___________')


	def get_state(self) -> list[float]:
		"""
		returning the current game state which consists of:
		- current direction of the snake:
			a binary list of 4 numbers as: [up, right, down, left]
		- snake's body parts -> 1
		- snake's head -> 2
		- food -> 5
		- empty cells -> 0
		"""
		state: list[float] = []

		if self.direction == 'u':
			state.extend([1, 0, 0, 0])
		elif self.direction == 'r':
			state.extend([0, 1, 0, 0])
		elif self.direction == 'd':
			state.extend([0, 0, 1, 0])
		elif self.direction == 'l':
			state.extend([0, 0, 0, 1])

		for row in self.world:
			for cell in row:
				if cell == '':
					state.append(0)
				elif cell == 'h':
					state.append(2)
				elif cell == 's':
					state.append(1)
				elif cell == 'f':
					state.append(5)

		return state



class SnakeGameGUI:
	"""
	handles all the graphical features and rendering the snake game
	"""

	class Block:
		"""
		Block class holds the information of each grid in the snake game
		Contains information of the positions as well as styles
		"""
		def __init__(
				self,
				left: int | float,
				top: int | float,
				border: int = 0,
				size: int = BLOCK_SIZE,
				color: tuple[int, int, int] = (255, 255, 255),
				kind: str = '',
				border_radius: tuple[int, int, int, int] = (0, 0, 0, 0) # (tl, tr, br, bl)
		) -> None:


			self.block: pg.Rect = pg.Rect((left, top), (size, size))
			self.color: pg.Color = pg.Color(*color)
			self.border: int = border
			self.border_radius: tuple[int, int, int, int] = border_radius

			# kind could be:
			# 'h': the snake's head
			# 's': the snake's body parts
			# 'f': the food
			# '': empty cells
			self.kind: str = kind


		def __repr__(self) -> str:
			return self.kind


	def __init__(self, render_enabled: bool = True) -> None:
		self.render_enabled = render_enabled
		self.game = SnakeGame()

		if self.render_enabled:
			pg.init()
			self.screen = pg.display.set_mode((WIDTH, HEIGHT))
			pg.display.set_caption('Snake Game')
			self.clock = pg.time.Clock()

			self.font = pg.font.Font(pg.font.get_default_font(), FONT_SIZE)

			self.fps = FPS

			# updates self.world from self.game.world
			self.update_world()


	def update_world(self) -> None:
		"""
		updates self.world from self.game.world out of Block objects instead of simple strings
		"""
		if SHAPE == 'circle':
			radiuses = tuple([BLOCK_SIZE for _ in range(4)])
		else:
			radiuses = tuple([0 for _ in range(4)])

		self.world = deepcopy(self.game.world)

		for r, row in enumerate(self.world):
			for c, cell in enumerate(row):
				# calculating the coordinates of each pixel/block
				left = c * BLOCK_SIZE + PD
				top = r * BLOCK_SIZE + PD

				coordinate = (c, r)
				# this seems backwards but is actually the right way
				# because r, which is rows, goes up and down -> y coordinate
				# and c, which is cols, goes right and left -> x coordinate

				# snake's head block
				if cell == 'h':
					self.world[r][c] = self.Block(
						left=left, top=top,
						color=SNAKE_HEAD_COLOR,
						kind=cell,
						border_radius=radiuses
					)

				elif cell == 's':
					self.world[r][c] = self.Block(
						left=left, top=top,
						color=SNAKE_COLOR,
						kind=cell,
						border_radius=radiuses
					)

				elif cell == 'f':
					self.world[r][c] = self.Block(
						left=left, top=top,
						color=FOOD_COLOR,
						kind=cell,
						border_radius=radiuses
					)

				else: # just the empty world block
					self.world[r][c] = self.Block(
						left=left, top=top,
						color=GRID_COLOR,
						border=1,
						kind=''
					)


	def draw_world(self) -> None:
		self.screen.fill(color=BG_COLOR)

		for r in range(HN):
			for c in range(WN):
				block = self.world[r][c].block
				block_color = self.world[r][c].color
				border = self.world[r][c].border
				radiuses = self.world[r][c].border_radius

				pg.draw.rect(
					self.screen,
					color=block_color,
					rect=block,
					width=border,
					border_top_left_radius=radiuses[0],
					border_top_right_radius=radiuses[1],
					border_bottom_right_radius=radiuses[2],
					border_bottom_left_radius=radiuses[3]
				)

		# to draw the walls
		# to make the blocks near the edge of the wall the correct size
		adj = PD//10
		pg.draw.rect(
			self.screen,
			color=WALL_COLOR,
			width=5,
			rect=(PD-adj, PD-adj, WIDTH - 2*PD + 2*adj, HEIGHT - 2*PD + 2*adj) # very nasty!
		)


	def messg_on_game_over(self, messg: str, color = FONT_COLOR) -> None:
		pg.time.delay(1000)
		while True:
			for event in pg.event.get():
				if event.type == pg.QUIT:
					return
				if event.type == pg.KEYDOWN:
					if event.key in  [pg.K_RETURN, pg.K_KP_ENTER]:
						return


			font = pg.font.Font(pg.font.get_default_font(), int(FONT_SIZE*1.8))
			text = font.render(messg, True, color)

			self.screen.fill(BG_COLOR)
			self.screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//3))
			pg.display.update()


	def step(self) -> bool:
		if not self.render_enabled:
			return self.game.game_over

		# get user input in event loop
		for event in pg.event.get():
			if event.type == pg.QUIT:
				pg.quit()
				sys.exit()

			if event.type == pg.KEYDOWN:
				if event.key == pg.K_KP_PLUS:
					if self.fps + 1 <= 25:
						self.fps += 1
				elif event.key == pg.K_KP_MINUS:
					if self.fps - 1 > 0:
						self.fps -= 1

		# stepping the game with a random move
		self.game.action = choice(['u', 'd', 'r', 'l'])

		self.game.step()

		self.update_world()


		# draw the whole game world and the score
		self.draw_world()

		info = self.font.render(f'Score = {self.game.score} ------- FPS = {self.fps:.0f}', True, FONT_COLOR)
		self.screen.blit(info, (PD, int(PD/5)))

		pg.display.update()
		self.clock.tick(self.fps)

		return self.game.game_over



class Agent:
	def __init__(self, train_mode: bool = False) -> None:
		# creating the neural network
		self.network = NeuralNetwork(
			layers_structure=[WN*HN+4, 16, 16, 4],
			activations='relu'
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
		self.gamma: float = 0.97

		# learning rate
		self.alpha: float = 0.0005

		# epsilon-greedy policy for explore-exploit trade-off
		# should decay over training to lower the exploration
		if train_mode:
			self.epsilon: float = 1
		else:
			self.epsilon: float = 0


	def choose_action(self, state: list[int]) -> str:
		"""
		chooses and returns a move based on the current state
		the action is in the form of a string in a form of:
		'u', 'r', 'd', 'l'
		"""

		actions: list[str] = ['u', 'r', 'd', 'l']

		# with probability epsilon, pick a random action
		if random() < self.epsilon:
			return choice(actions)

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
		self.epsilon = max(0.1, self.epsilon * 0.995)


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
			constant_lr=True,
			batch_size=1,
			number_of_epochs=5,
			verbose=False
		)


def train_agent(resume: bool = False, episodes: int = 20):
	agent = Agent(train_mode=True)
	game = SnakeGame()

	total_reward: float = 0

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
			ai_action: str = agent.choose_action(state)
			game.action = ai_action

			next_state, reward, done = game.step()

			action_map = {
				'u': 0,
				'r': 1,
				'd': 2,
				'l': 3
			}

			action: int = action_map[ai_action]

			agent.update(state, action, reward, next_state, done)
			episode_reward += reward

			state = next_state

		agent.decay_epsilon()
		total_reward += episode_reward

		if episode%20 == 0:
			print(f'Episode {episode}:\t{total_reward=:.1f}, {episode_reward=:.1f}, {steps_survived=}, {agent.epsilon=:.3f}')

	agent.network.save_parameters_to_file(PARAMETERS_FILE)
	with open('params.txt', 'w') as file:
		file.write(f'{agent.epsilon}')


if __name__ == '__main__':
	train_agent(resume=False, episodes=2000)