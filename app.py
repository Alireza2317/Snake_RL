import sys
import pygame as pg
from random import randint

# speed. the higher the fps, the faster the game
FPS = 8
# if set to true, will scale up the fps after eating a food
INCREMENT_SPEED = False
SCALE = 1.0065

# dimensions
WN: int = 20
HN: int = 16
BLOCK_SIZE = 40

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

INITIAL_SIZE = 4
# the snake's head roundness, 0 to disable
# only when SHAPE = 'square'
ROUNDNESS = BLOCK_SIZE


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

# foods
# this will be the number of foods that will always be in the game map
NUM_FOODS = 1

# checking some parameters, to make the game more well-behaved
if NUM_FOODS >= WN * HN:
	raise ValueError(f'Number of foods should be less than {WN*HN}')
if  FPS <= 0:
	raise ValueError('FPS should be positive numbers!')
if INITIAL_SIZE >= WN-1:
	raise ValueError('Snake\'s INITIAL_SIZE is too high!')

# comment this if your screen is big enough
if WIDTH > 1920 or HEIGHT > 1080:
	raise ValueError('Consider reducing WN and HN or BLOCK_SIZE! Too big for most screens!')

# comment this if you wanna get dizzy :)
if FPS > 25:
	raise ValueError('Consider using a lower value for FPS.')


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


class Snake:
	"""
	Snake class that handles all the logic of a snake
	"""
	def __init__(self, init_size: int = 3) -> None:
		self.direction: str = 'r'

		# first element will be the head
		self.body: list[Position] = [
			Position(i-1, 0)
			for i in range(init_size-1, -1, -1)
		]

		# useful when growing the snake
		self._left_over: Position = self.body[-1]


	@property
	def size(self):
		return len(self.body)


	@property
	def head(self) -> Position:
		return self.body[0]


	def turn(self, dir_to_turn: str) -> None:
		if self.direction == 'u':
			if dir_to_turn == 'd':
				return

		elif self.direction == 'd':
			if dir_to_turn == 'u':
				return

		elif self.direction == 'r':
			if dir_to_turn == 'l':
				return

		elif self.direction == 'l':
			if dir_to_turn == 'r':
				return

		self.direction = dir_to_turn


	def grow(self) -> None:
		self.body.append(self._left_over)


	def move(self) -> None:
		# find out the new position of the head
		new_head = Position(*self.head.astuple)

		# remove the last body part and save it
		self._left_over = self.body.pop()


		match self.direction:
			case 'r':
				new_head.x += 1
			case 'l':
				new_head.x -= 1
			case 'd':
				new_head.y += 1
			case 'u':
				new_head.y -= 1


		# all except the first and the last Position (head and tail)
		self.body.insert(0, new_head)


	def ate_food(self, food_pos: Position | tuple) -> bool:
		return (self.head == food_pos)


	def hit_position(self, pos: Position | tuple) -> bool:
		return (pos in self.body)


	def hit_self(self) -> bool:
		return (self.head in self.body[1:])


	def __repr__(self) -> str:
		return str(self.body)


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
			kind: str = 'blank',
			border_radius: tuple[int, int, int, int] = (0, 0, 0, 0) # (tl, tr, br, bl)
	) -> None:


		self.block: pg.Rect = pg.Rect((left, top), (size, size))
		self.color: pg.Color = pg.Color(*color)
		self.border: int = border
		self.border_radius: tuple[int, int, int, int] = border_radius

		# kind could be: 'blank', 'snake', 'head', 'food'
		self.kind: str = kind


	def __repr__(self) -> str:
		return self.kind[0]


class SnakeGame:
	def __init__(self) -> None:
		pg.init()
		self.screen = pg.display.set_mode((WIDTH, HEIGHT))
		pg.display.set_caption('Snake Game')
		self.clock = pg.time.Clock()

		self.font = pg.font.Font(pg.font.get_default_font(), FONT_SIZE)

		self.score: int = 0
		self.fps = FPS
		self.game_over = False
		self.num_foods = NUM_FOODS

		# will initialize self.world, self.snake, self.foods
		self.reset()


	def reset(self) -> None:
		# reset
		self.game_over = False
		self.score = 0

		self.world: list[list[Block]] = [
			[None for col in range(WN)] for row in range(HN)
		]

		self.snake = Snake(init_size=INITIAL_SIZE)

		self.foods: list[Position] = []
		self.generate_foods()

		self.update_world()


	def calc_border_radiuses(self) -> tuple[int, int, int, int]:
		direction = self.snake.direction

		tr = tl = br = bl = 0
		if direction == 'u':
			tr = tl = ROUNDNESS
		elif direction == 'd':
			br = bl = ROUNDNESS
		elif direction == 'r':
			tr = br = ROUNDNESS
		elif direction == 'l':
			tl = bl = ROUNDNESS

		return (tl, tr, br, bl)


	def update_world(self) -> None:
		for r in range(HN):
			for c in range(WN):
				# calculating the coordinates of each pixel/block
				left = c * BLOCK_SIZE + PD
				top = r * BLOCK_SIZE + PD

				coordinate = (c, r)
				# this seems backwards but is actually the right way
				# because r, which is rows, goes up and down -> y coordinate
				# and c, which is cols, goes right and left -> x coordinate

				if SHAPE == 'circle':
					radiuses = tuple([BLOCK_SIZE for _ in range(4)])
				elif SHAPE == 'square':
					radiuses = tuple([0 for _ in range(4)])

				if self.snake.hit_position(pos=coordinate):
					# neat trick to use ate_food method to check collision with head
					if self.snake.ate_food(coordinate):
						# the snake's head, only if want different color for the head
						# rounding the head based on the direction of snake
						if SHAPE == 'square':
							radiuses = self.calc_border_radiuses()
						self.world[r][c] = Block(left=left, top=top, color=SNAKE_HEAD_COLOR, kind='head', border_radius=radiuses)
					else: # snake body parts except the head

						self.world[r][c] = Block(left=left, top=top, color=SNAKE_COLOR, kind='snake', border_radius=radiuses)

				elif coordinate in self.foods:
					self.world[r][c] = Block(left=left, top=top, color=FOOD_COLOR, kind='food', border_radius=radiuses)

				else: # just the empty world block
					self.world[r][c] = Block(left=left, top=top, color=GRID_COLOR, border=1, kind='blank')


	def draw_world(self) -> None:
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


	def hit_wall(self) -> bool:
		out_of_x = (self.snake.head.x < 0 or self.snake.head.x >= WN)
		out_of_y = (self.snake.head.y < 0 or self.snake.head.y >= HN)

		return (out_of_x or out_of_y)


	def generate_foods(self) -> None:
		while True:
			if len(self.foods) >= self.num_foods: break

			x, y = randint(0, WN-1), randint(0, HN-1)
			p = Position(x, y)

			# this coordinate should not collide with other foods or the snake
			if self.snake.hit_position(pos=p) or (p in self.foods): continue

			self.foods.append(p)


	def is_world_full(self) -> bool:
		# check to see if there are any blank blocks in the world
		for row in self.world:
			for blk in row:
				if blk.kind == 'blank':
					return False
		return True


	def is_world_snaked(self) -> bool:
		# best function name does not exist:
		for row in self.world:
			for blk in row:
				if blk.kind not in ['head', 'snake']:
					return False

		return True


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
		# 1. get user input
		# event loop
		for event in pg.event.get():
			if event.type == pg.QUIT:
				pg.quit()
				sys.exit()

			elif event.type == pg.KEYDOWN:
				if event.key == pg.K_UP:
					self.snake.turn('u')
				elif event.key == pg.K_DOWN:
					self.snake.turn('d')
				elif event.key == pg.K_LEFT:
					self.snake.turn('l')
				elif event.key == pg.K_RIGHT:
					self.snake.turn('r')

				if event.key == pg.K_KP_PLUS:
					if self.fps + 1 <= 25:
						self.fps += 1
				elif event.key == pg.K_KP_MINUS:
					if self.fps - 1 > 0:
						self.fps -= 1
				break

		# 2. snake moves
		self.snake.move()

		# 3. snake grows if ate any food
		for i, food in enumerate(self.foods):
			if self.snake.ate_food(food_pos=food):
				if INCREMENT_SPEED: self.fps *= SCALE

				self.foods.pop(i)
				self.snake.grow()

				self.score += 1

				if not self.is_world_full():
					self.generate_foods()

				# if snake ate a food, no need to continue this loop
				break


		# 4. check collisions
		if self.snake.hit_self() or self.hit_wall():
			self.game_over = True
			self.messg_on_game_over(messg='Game Over!')

		# 5. update the world
		self.update_world()

		# 6. check if the player won the game?!
		if self.is_world_snaked():
			# this will probably never happen in a real game!
			self.game_over = True # good game over!
			self.messg_on_game_over(messg='Well congrats! You won the snake game!')

		# 7. draw the whole game world and the score
		self.screen.fill(color=BG_COLOR)
		self.draw_world()

		info = self.font.render(f'Score = {self.score} ---- FPS = {self.fps:.1f}', True, FONT_COLOR)
		self.screen.blit(info, (PD, int(PD/5)))


		pg.display.update()
		self.clock.tick(self.fps)

		return self.game_over


if __name__ == '__main__':
	game: SnakeGame = SnakeGame()

	while True:
		game_over = game.step()

		if game_over:
			break

	pg.quit()
	sys.exit()
