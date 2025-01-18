import sys
import pygame as pg
from copy import deepcopy
from collections import namedtuple
from enum import Enum
from random import choice, seed

seed(23)

# speed. the higher the fps, the faster the game
FPS = 6
# if set to true, will scale up the fps after eating a food
INCREMENT_SPEED = False
SCALE = 1.0065

# dimensions
WN: int = 20
HN: int = 20
BLOCK_SIZE = 35

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

if  FPS <= 0:
	raise ValueError('FPS should be positive numbers!')
if INITIAL_SIZE >= WN-1:
	raise ValueError('Snake\'s INITIAL_SIZE is too high!')

# comment this if your screen is big enough
if WIDTH > 1920 or HEIGHT > 1080:
	raise ValueError('Consider reducing WN and HN or BLOCK_SIZE! Too big for most screens!')


Position = namedtuple('Position', ['x', 'y'])

class Direction(Enum):
	UP = 0
	RIGHT = 1
	DOWN = 2
	LEFT = 3

class Reward(Enum):
	SURVIVE = 0.1
	GROW = 1
	DIE = -1
	PROXIMITY = 0.2


NUM_STATES = 14
NUM_ACTIONS = 4

class SnakeGame:
	"""
	handles all the logic of a snake game
	"""
	def __init__(self) -> None:
		self.reset()


	def reset(self) -> None:
		self.direction: Direction = Direction.RIGHT
		self.action: Direction = Direction.RIGHT

		# first element will be the head
		self.snake: list[Position] = [
			Position(i, HN//2)
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

		self.survival_score: int = 0
		self.food_score: int = 0


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


	def turn(self, dir_to_turn: Direction) -> None:
		verticals = (Direction.DOWN, Direction.UP)
		horizentals = (Direction.LEFT, Direction.RIGHT)

		if self.direction in verticals and dir_to_turn in verticals:
			return

		elif self.direction in horizentals and dir_to_turn in horizentals:
			return

		self.direction = dir_to_turn


	def grow(self) -> None:
		self.snake.append(self._left_over)
		self.food_score += 1


	def move(self) -> None:
		self.survival_score += 1

		# remove the last body part and save it
		self._left_over = self.snake.pop()

		match self.direction:
			case Direction.RIGHT:
				new_head = Position(self.head.x+1, self.head.y)
			case Direction.LEFT:
				new_head = Position(self.head.x-1, self.head.y)
			case Direction.DOWN:
				new_head = Position(self.head.x, self.head.y+1)
			case Direction.UP:
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

		for r in range(HN):
			for c in range(WN):
				pos = Position(c, r)
				if not self.hit_position(pos):
					valid_cells.append(pos)

		if hasattr(self, 'food'):
			if self.food in valid_cells:
				valid_cells.remove(self.food)

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


	def is_getting_close_to_food(self) -> bool:
		# the current distance from food
		current_head_pos = self.head

		dist_x = self.food.x - current_head_pos.x
		dist_y = self.food.y - current_head_pos.y

		dist = abs(dist_x) + abs(dist_y)

		if self.direction == Direction.UP:
			next_head_pos = Position(current_head_pos.x, current_head_pos.y-1)
		if self.direction == Direction.RIGHT:
			next_head_pos = Position(current_head_pos.x+1, current_head_pos.y)
		if self.direction == Direction.DOWN:
			next_head_pos = Position(current_head_pos.x, current_head_pos.y+1)
		if self.direction == Direction.LEFT:
			next_head_pos = Position(current_head_pos.x-1, current_head_pos.y)

		next_dist_x = self.food.x - next_head_pos.x
		next_dist_y = self.food.y - next_head_pos.y

		next_dist = abs(next_dist_x) + abs(next_dist_y)

		return (next_dist < dist)


	def step(self) -> tuple[list[float], float, bool]:
		"""
		takes in an action and applies it to the game
		returns a tuple in this order:
			the game state after the snake's move
			reward
			game_over flag
		"""

		# reward for staying alive but not eating food
		# avoid encouraging the agent to just stay alive and not eat food
		reward: float = Reward.SURVIVE.value

		self.turn(self.action)

		if self.is_getting_close_to_food():
			reward += Reward.PROXIMITY.value
		else:
			reward -= Reward.PROXIMITY.value

		self.move()

		# snake grows if ate any food
		if self.ate_food():
			self.grow()
			# reward for eating food
			reward += Reward.GROW.value

			if not self.is_world_full():
				self.generate_food()


		# check collisions
		self.game_over = self.hit_self() or self.hit_wall()
		if self.game_over:
			# reward for dying
			reward += Reward.DIE.value

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
		returning the current game state which consists of 14 elements:
		* float values normalized
			- food_x_dist
			- food_y_dist
			- up_wall_dist
			- right_wall_dist
			- down_wall_dist
			- left_wall_dist
		* binary values
			- up_self_danger
			- right_self_danger
			- down_self_danger
			- left_self_danger
			- up_direction
			- right_direction
			- down_direction
			- left_direction
		"""

		food_x_dist = (self.food.x -  self.head.x) / WN
		food_y_dist = (self.food.y -  self.head.y) / HN

		up_wall_dist = (self.head.y) / HN
		right_wall_dist = (WN - self.head.x) / WN
		down_wall_dist = (HN - self.head.y) / HN
		left_wall_dist = (self.head.x) / WN

		up_self_danger = 0
		right_self_danger = 0
		down_self_danger = 0
		left_self_danger = 0

		# these are needed because we want to ignore the snake's body(neck)
		# based on its direction
		dir_down: bool = self.direction == Direction.DOWN
		dir_left: bool = self.direction == Direction.LEFT
		dir_up: bool = self.direction == Direction.UP
		dir_right: bool = self.direction == Direction.RIGHT

		# these happen when the snake turns in that direction
		dies_if_turned_up: bool = self.hit_position((self.head.x, self.head.y-1))
		dies_if_turned_right: bool = self.hit_position((self.head.x+1, self.head.y))
		dies_if_turned_down: bool = self.hit_position((self.head.x, self.head.y+1))
		dies_if_turned_left: bool = self.hit_position((self.head.x-1, self.head.y))

		if dies_if_turned_up and not dir_down:
			up_self_danger = 1

		if dies_if_turned_right and not dir_left:
			right_self_danger = 1

		if dies_if_turned_down and not dir_up:
			down_self_danger = 1

		if dies_if_turned_left and not dir_right:
			left_self_danger = 1

		state: list[float] = [
			food_x_dist,
			food_y_dist,
			up_wall_dist,
			right_wall_dist,
			down_wall_dist,
			left_wall_dist,
			up_self_danger,
			right_self_danger,
			down_self_danger,
			left_self_danger
		]

		if dir_up:
			state.extend([1, 0, 0, 0])
		elif dir_right:
			state.extend([0, 1, 0, 0])
		elif dir_down:
			state.extend([0, 0, 1, 0])
		elif dir_left:
			state.extend([0, 0, 0, 1])

		return state



class SnakeGameGUI(SnakeGame):
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


	def __init__(self) -> None:
		super().__init__()

		pg.init()
		self.screen = pg.display.set_mode((WIDTH, HEIGHT))
		pg.display.set_caption('Snake Game')
		self.clock = pg.time.Clock()

		self.font = pg.font.Font(pg.font.get_default_font(), FONT_SIZE)

		self.fps = FPS

		self.text = ''

		# updates self.gui_world from self.world
		self.update_world()


	def update_gui_world(self) -> None:
		"""
		updates self.world from SnakeGame.world out of Block objects instead of simple strings
		"""

		super().update_world()

		self.gui_world = deepcopy(self.world)

		if SHAPE == 'circle':
			radiuses = tuple([BLOCK_SIZE for _ in range(4)])
		else:
			radiuses = tuple([0 for _ in range(4)])

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
					self.gui_world[r][c] = self.Block(
						left=left, top=top,
						color=SNAKE_HEAD_COLOR,
						kind=cell,
						border_radius=radiuses
					)

				elif cell == 's':
					self.gui_world[r][c] = self.Block(
						left=left, top=top,
						color=SNAKE_COLOR,
						kind=cell,
						border_radius=radiuses
					)

				elif cell == 'f':
					self.gui_world[r][c] = self.Block(
						left=left, top=top,
						color=FOOD_COLOR,
						kind=cell,
						border_radius=radiuses
					)

				else: # just the empty world block
					self.gui_world[r][c] = self.Block(
						left=left, top=top,
						color=GRID_COLOR,
						border=1,
						kind=''
					)


	def draw_world(self) -> None:
		self.screen.fill(color=BG_COLOR)

		for r in range(HN):
			for c in range(WN):
				block = self.gui_world[r][c].block
				block_color = self.gui_world[r][c].color
				border = self.gui_world[r][c].border
				radiuses = self.gui_world[r][c].border_radius

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


	def step(self) -> tuple[list[float], float, bool]:
		# get user input in event loop
		for event in pg.event.get():
			if event.type == pg.QUIT:
				pg.quit()
				sys.exit()

			if event.type == pg.KEYDOWN:
				if event.key == pg.K_KP_PLUS:
					if self.fps + 1 <= 60:
						self.fps += 1
				elif event.key == pg.K_KP_MINUS:
					if self.fps - 1 > 0:
						self.fps -= 1

		# stepping the game with a random move
		self.action = choice(list(Direction))

		state, reward, done = super().step()

		self.update_gui_world()


		# draw the whole game world and the score
		self.draw_world()

		info = self.font.render(
			f'survived={self.survival_score: >4}   ----   foods={self.food_score: >3}   ----   FPS = {self.fps:.0f}',
			True, FONT_COLOR
		)
		self.screen.blit(info, (PD, int(PD/5)))

		additional_text = self.font.render(
			self.text, True, FONT_COLOR
		)

		self.screen.blit(additional_text, (WIDTH//2-WIDTH//5, HEIGHT-PD+15))

		pg.display.update()
		self.clock.tick(self.fps)

		return state, reward, done
