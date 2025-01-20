import sys
import random
import pygame as pg
from copy import deepcopy
from collections import namedtuple
from enum import Enum
from dataclasses import dataclass, field


class Shape(Enum):
	square = 0
	circle = 1


class Direction(Enum):
	UP = 0
	RIGHT = 1
	DOWN = 2
	LEFT = 3


class Reward(Enum):
	SURVIVE = -0.001
	GROW = 2
	DIE = -1
	PROXIMITY = 0.2


@dataclass
class SnakeGameConfig:
	"""
	Stores all the configuration variables for the snake game.
	"""
	# grid dimensions
	grid_n_columns: int = 20
	grid_n_rows: int = 20

	# screen sizes in pixel
	grid_size: int = 40
	# some padding outside the walls of the game. space to render texts
	padding: int = 50
	screen_width: int = grid_size * grid_n_columns + 2 * padding
	screen_height: int = grid_size * grid_n_rows + 2 * padding

	# speed. the higher the fps, the faster the game
	fps: int = 6

	# snake's initial conditions
	initial_size: int = 3
	initial_direction: Direction = Direction.RIGHT
	initial_action: Direction = Direction.RIGHT

	# colors, all in tuple format, (r, g, b)
	bg_color: tuple[int, int, int] = (35, 35, 35)
	snake_head_color: tuple[int, int, int] = (255, 204, 62)
	snake_color: tuple[int, int, int] = (54, 110, 156)
	grid_color: tuple[int, int, int] = (38, 38, 38)
	wall_color: tuple[int, int, int] = (220, 220, 240)
	food_color: tuple[int, int, int] = (239, 57, 57)

	# fonts for texts
	font_size= 22
	font_color = (220, 220, 220)

	# shape of the grid cells
	cell_shape: Shape = Shape.square

	# make these immutable since they should be constant all the time
	NUM_STATES: int = field(init=False, default=14)
	NUM_ACTIONS: int = field(init=False, default=4)


	def __post_init__(self):
		# data validation

		if self.fps <= 0:
			raise ValueError('FPS should be positive numbers!')

		if self.initial_size >= self.grid_n_columns - 1:
			raise ValueError('Snake initial size is too high!')

		if self.grid_n_rows <= 0 or self.grid_n_columns <= 0:
			raise ValueError('Number of rows and columns should be positive!')

		if self.screen_width > 1920 or self.screen_height > 1080:
			raise ValueError('Consider reducing grid_n_rows and grid_n_columns or grid_size!')


Position = namedtuple('Position', ['x', 'y'])


class SnakeGame:
	"""
	handles all the logic of a snake game such as:
	- turning the snake
	- moving the snake
	- growing the snake
	- generating food
	- identify self or wall collision
	- getting the current state of the game
	- stepping the game
	and so on ...
	"""
	def __init__(self) -> None:
		# create a config with the default values
		self.cfg: SnakeGameConfig = SnakeGameConfig()
		self.reset()


	def reset(self) -> None:
		""" Resets the whole game. """

		self.direction: Direction = Direction.RIGHT
		self.action: Direction = Direction.RIGHT

		# first element will be the head
		self.snake: list[Position] = [
			Position(i + self.cfg.grid_n_columns // 2 - self.cfg.initial_size, self.cfg.grid_n_rows // 2)
			for i in range(self.cfg.initial_size, -1, -1)
		]

		# useful when growing the snake
		self._left_over: Position = self.snake[-1]

		# the world consists of 4 elements:
		# 'h': is the position of the snake's head(only one)
		# 's': is the position of the snake's body parts(can be more than one)
		# 'f': is the position of the food in the world
		# '': is the empty cells in the world
		self.world: list[list[str]] = [
			['' for col in range(self.cfg.grid_n_columns)] for row in range(self.cfg.grid_n_rows)
		]

		# creates self.food property
		self.generate_food()

		# update the world
		self.update_world()

		self.game_over = False

		self.survival_score: int = 0
		self.food_score: int = 0


	def update_world(self) -> None:
		""" Updates self.world based on self.snake and self.food positions. """

		for r in range(self.cfg.grid_n_rows):
			for c in range(self.cfg.grid_n_columns):
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
		""" Turn the snake based on the given direction. """

		verticals = (Direction.DOWN, Direction.UP)
		horizentals = (Direction.LEFT, Direction.RIGHT)

		if self.direction in verticals and dir_to_turn in verticals:
			return

		elif self.direction in horizentals and dir_to_turn in horizentals:
			return

		self.direction = dir_to_turn


	def grow(self) -> None:
		""" Grows the snake by one unit. """

		self.snake.append(self._left_over)
		self.food_score += 1


	def move(self) -> None:
		""" Moves the snake and updates the snake's position based on the current direction. """

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
		""" Detects collision with the food. """

		return (self.head == self.food)


	def hit_position(self, pos: Position | tuple) -> bool:
		""" Detects if the given position collides with the snake. """

		return (pos in self.snake)


	def hit_self(self) -> bool:
		""" Detects if the snake hits itself. """

		return (self.head in self.snake[1:])


	def hit_wall(self) -> bool:
		""" Detects if the snake hits the walls. """

		out_of_x = (self.head.x < 0 or self.head.x >= self.cfg.grid_n_columns)
		out_of_y = (self.head.y < 0 or self.head.y >= self.cfg.grid_n_rows)

		return (out_of_x or out_of_y)


	def generate_food(self) -> None:
		""" Generates a food randomly, and sets self.food. """

		valid_cells: list[Position] = []

		for r in range(self.cfg.grid_n_rows):
			for c in range(self.cfg.grid_n_columns):
				pos = Position(c, r)
				if not self.hit_position(pos):
					valid_cells.append(pos)

		if hasattr(self, 'food'):
			if self.food in valid_cells:
				valid_cells.remove(self.food)

		self.food = random.choice(valid_cells)


	def is_world_full(self) -> bool:
		""" Checks to see if there are any empty cells in the world. """

		for row in self.world:
			for cell in row:
				if cell == '':
					return False
		return True


	def is_world_occupied_by_snake(self) -> bool:
		""" Checks to see if the snake occupies all the cells in the world. """

		for row in self.world:
			for cell in row:
				# if the cell is not the snake's head or body parts
				if cell not in ['h', 's']:
					return False

		return True


	def is_getting_close_to_food(self) -> bool:
		""" Predicts food proximity based on the cuurent direction. Assuming the Direction being applied. """

		# the current distance from food
		current_head_pos = self.head

		dist_x = abs(self.food.x - current_head_pos.x)
		dist_y = abs(self.food.y - current_head_pos.y)

		dist = dist_x + dist_y

		if self.direction == Direction.UP:
			next_head_pos = Position(current_head_pos.x, current_head_pos.y-1)
		elif self.direction == Direction.RIGHT:
			next_head_pos = Position(current_head_pos.x+1, current_head_pos.y)
		elif self.direction == Direction.DOWN:
			next_head_pos = Position(current_head_pos.x, current_head_pos.y+1)
		elif self.direction == Direction.LEFT:
			next_head_pos = Position(current_head_pos.x-1, current_head_pos.y)

		next_dist_x = abs(self.food.x - next_head_pos.x)
		next_dist_y = abs(self.food.y - next_head_pos.y)

		next_dist = next_dist_x + next_dist_y

		return (next_dist < dist)


	def step(self) -> tuple[list[float], float]:
		"""
		Applies self.action to the game
		Returns a tuple in this order:
			the game state after the snake's move
			reward
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
		if self.is_world_occupied_by_snake():
			# this will probably never happen in a real game!
			self.game_over = True # good game over!
			print('You won the snake game!')

		return self.get_state(), reward


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
		Returns the current game state which consists of 14 elements:
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

		# normalized distances
		food_x_dist = (self.food.x -  self.head.x) / self.cfg.grid_n_columns
		food_y_dist = (self.food.y -  self.head.y) / self.cfg.grid_n_rows

		up_wall_dist = (self.head.y) / self.cfg.grid_n_rows
		right_wall_dist = (self.cfg.grid_n_columns - self.head.x) / self.cfg.grid_n_columns
		down_wall_dist = (self.cfg.grid_n_rows - self.head.y) / self.cfg.grid_n_rows
		left_wall_dist = (self.head.x) / self.cfg.grid_n_columns

		up_self_danger = 0
		right_self_danger = 0
		down_self_danger = 0
		left_self_danger = 0

		# these are needed because we want to ignore the snake's body(neck)
		# based on its direction
		dir_up: bool = self.direction == Direction.UP
		dir_right: bool = self.direction == Direction.RIGHT
		dir_down: bool = self.direction == Direction.DOWN
		dir_left: bool = self.direction == Direction.LEFT

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
	"""	Handles all the graphical features and rendering the snake game. """

	class Block:
		"""
		Block class holds the information of each grid in the snake game
		Contains information of the positions as well as styles.
		Creates square blocks.
		"""

		def __init__(
				self,
				top: int,
				left: int,
				color: tuple[int, int, int],
				size: int,
				kind: str = '',
				border: int = 0,
				border_radii: tuple[int, int, int, int] = (0, 0, 0, 0) # (tl, tr, br, bl)
		) -> None:

			self.block: pg.Rect = pg.Rect((left, top), (size, size))
			self.color: pg.Color = pg.Color(*color)
			self.border: int = border
			self.border_radii: tuple[int, int, int, int] = border_radii

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
		self.screen = pg.display.set_mode((self.cfg.screen_width, self.cfg.screen_height))
		pg.display.set_caption('Snake Game')
		self.clock = pg.time.Clock()

		self.font = pg.font.Font(pg.font.get_default_font(), self.cfg.font_size)

		self.fps = self.cfg.fps

		self.text = ''

		# updates self.gui_world from self.world
		self.update_world()


	def update_gui_world(self) -> None:
		"""	Updates self.world from SnakeGame.world out of Block objects. """

		super().update_world()

		self.gui_world = deepcopy(self.world)

		if self.cfg.cell_shape == Shape.circle:
			radiuses = tuple([self.cfg.grid_size for _ in range(4)])
		elif self.cfg.cell_shape == Shape.square:
			radiuses = tuple([0 for _ in range(4)])

		for r, row in enumerate(self.world):
			for c, cell in enumerate(row):
				# calculating the coordinates of each pixel/block
				left = c * self.cfg.grid_size + self.cfg.padding
				top = r * self.cfg.grid_size + self.cfg.padding

				# snake's head block
				if cell == 'h':
					self.gui_world[r][c] = self.Block(
						left=left, top=top,
						size=self.cfg.grid_size,
						color=self.cfg.snake_head_color,
						kind=cell,
						border_radii=radiuses
					)
				elif cell == 's':
					self.gui_world[r][c] = self.Block(
						left=left, top=top,
						size=self.cfg.grid_size,
						color=self.cfg.snake_color,
						kind=cell,
						border_radii=radiuses
					)

				elif cell == 'f':
					self.gui_world[r][c] = self.Block(
						left=left, top=top,
						size=self.cfg.grid_size,
						color=self.cfg.food_color,
						kind=cell,
						border_radii=radiuses
					)

				else: # just the empty world block
					self.gui_world[r][c] = self.Block(
						left=left, top=top,
						size=self.cfg.grid_size,
						color=self.cfg.grid_color,
						border=1,
						kind=''
					)


	def draw_world(self) -> None:
		""" Renders self.world on the screen. """

		self.screen.fill(color=self.cfg.bg_color)

		for r in range(self.cfg.grid_n_rows):
			for c in range(self.cfg.grid_n_columns):
				block = self.gui_world[r][c].block
				block_color = self.gui_world[r][c].color
				border = self.gui_world[r][c].border
				radii = self.gui_world[r][c].border_radii

				pg.draw.rect(
					self.screen,
					color=block_color,
					rect=block,
					width=border,
					border_top_left_radius=radii[0],
					border_top_right_radius=radii[1],
					border_bottom_right_radius=radii[2],
					border_bottom_left_radius=radii[3]
				)


		# to draw the walls
		# adjustment to align the pixels
		adj = self.cfg.padding // 10
		pg.draw.rect(
			self.screen,
			color=self.cfg.wall_color,
			width=5,
			rect=(
				self.cfg.padding - adj, self.cfg.padding - adj,
				self.cfg.screen_width - 2 * self.cfg.padding + 2 * adj,
				self.cfg.screen_height - 2 * self.cfg.padding + 2 * adj
			)
		)


	def step(self) -> tuple[list[float], float]:
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

		state, reward = super().step()

		self.update_gui_world()

		# draw the whole game world and the score
		self.draw_world()

		info = self.font.render(
			f'survived={self.survival_score: >4}   ----   foods={self.food_score: >3}   ----   FPS = {self.cfg.fps}',
			True, self.cfg.font_color
		)
		self.screen.blit(info, (self.cfg.padding, self.cfg.padding // 5))

		additional_text = self.font.render(
			self.text, True, self.cfg.font_color
		)

		self.screen.blit(additional_text, (self.cfg.screen_width-self.cfg.padding, 0))

		pg.display.update()
		self.clock.tick(self.fps)

		return state, reward


if __name__ == '__main__':
	game = SnakeGameGUI()
	game.fps = 1

	while True:
		d = random.choice(list(Direction))
		game.action = d
		game.step()
		if game.game_over:
			break