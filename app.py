import sys
import time
import pygame as pg
from random import choice, randint
from copy import deepcopy

# speed. the higher the fps, the faster the game
FPS = 8
# if set to true, will scale up the fps after eating a food
INCREMENT_SPEED = False
SCALE = 1.0065

# dimensions
WN: int = 12
HN: int = 12
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


if  FPS <= 0:
	raise ValueError('FPS should be positive numbers!')
if INITIAL_SIZE >= WN-1:
	raise ValueError('Snake\'s INITIAL_SIZE is too high!')

# comment this if your screen is big enough
if WIDTH > 1920 or HEIGHT > 1080:
	raise ValueError('Consider reducing WN and HN or BLOCK_SIZE! Too big for most screens!')


class Position:
	"""
	Position class that acts as coordinates in a 2d plane
	"""
	def __init__(self, x: int, y: int) -> None:
		self.x = x
		self.y = y


	def __eq__(self, other: tuple | object) -> bool:
		if isinstance(other, tuple):
			return self.astuple == other

		elif isinstance(other, Position):
			return ((self.x == other.x) and (self.y == other.y))

		return False

	def __ne__(self, other: tuple | object) -> bool:
		if isinstance(other, tuple):
			return self.astuple != other

		elif isinstance(other, Position):
			return ((self.x != other.x) or (self.y != other.y))

		return False

	@property
	def astuple(self):
		return (self.x, self.y)


	def __repr__(self) -> str:
		return str(self.astuple)


class SnakeGame:
	"""
	handles all the logic of a snake game
	"""
	def __init__(self, init_size: int = INITIAL_SIZE) -> None:
		self.direction: str = 'r'

		# first element will be the head
		self.snake: list[Position] = [
			Position(i, 0)
			for i in range(init_size, -1, -1)
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
		# find out the new position of the head
		new_head = Position(*self.head.astuple)

		# remove the last body part and save it
		self._left_over = self.snake.pop()


		match self.direction:
			case 'r':
				new_head.x += 1
			case 'l':
				new_head.x -= 1
			case 'd':
				new_head.y += 1
			case 'u':
				new_head.y -= 1


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


	def step(self, action: str) -> bool:
		"""
		takes in an action and applies it to the game
		returns wether the game is over or not
		"""
		self.turn(action)

		self.move()

		# snake grows if ate any food
		if self.ate_food():
			self.grow()
			self.score += 1

			if not self.is_world_full():
				self.generate_food()


		# check collisions
		self.game_over = self.hit_self() or self.hit_wall()
		if self.game_over:
			print('You lost!')

		# update the world
		self.update_world()

		# check if the player won the game?!
		if self.is_world_snaked():
			# this will probably never happen in a real game!
			self.game_over = True # good game over!
			print('You won the snake game!')

		return self.game_over


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
		action = choice(['u', 'd', 'r', 'l'])

		self.game.step(action)
		
		self.update_world()


		# draw the whole game world and the score
		self.draw_world()

		info = self.font.render(f'Score = {self.game.score} ---- FPS = {self.fps:.1f}', True, FONT_COLOR)
		self.screen.blit(info, (PD, int(PD/5)))

		pg.display.update()
		self.clock.tick(self.fps)

		return self.game.game_over


if __name__ == '__main__':
	game: SnakeGame = SnakeGameGUI(render_enabled=True)

	while True:
		game_over = game.step()

		if game_over:
			break

	pg.quit()
	sys.exit()