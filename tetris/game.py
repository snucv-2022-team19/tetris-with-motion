from copy import deepcopy
from random import choice, randrange

import pygame


class Tetris:
    def __init__(self):
        self.running = False
        self.sc = None
        self.game_sc = None
        self.clock = None
        self.fps = 30

        # score rule
        self.scores = {0: 0, 1: 100, 2: 300, 3: 700, 4: 1500}

        # sizes
        self.sc_size = self.sc_w, self.sc_h = 750, 940
        self.tile_size = 45
        self.n_tile_w, self.n_tile_h = 10, 20
        self.game_size = self.game_w, self.game_h = (
            self.n_tile_w * self.tile_size,
            self.n_tile_h * self.tile_size,
        )

        # grid
        self.grid = [
            pygame.Rect(
                x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size
            )
            for x in range(self.n_tile_w)
            for y in range(self.n_tile_h)
        ]

        # Tetris figures
        # The first element is the center of the rotation for the each figure.
        self.figures_pos = [
            [(-1, 0), (-2, 0), (0, 0), (1, 0)],
            [(0, -1), (-1, -1), (-1, 0), (0, 0)],
            [(-1, 0), (-1, 1), (0, 0), (0, -1)],
            [(0, 0), (-1, 0), (0, 1), (-1, -1)],
            [(0, 0), (0, -1), (0, 1), (-1, -1)],
            [(0, 0), (0, -1), (0, 1), (1, -1)],
            [(0, 0), (0, -1), (0, 1), (-1, 0)],
        ]
        self.figures = [
            [pygame.Rect(x + self.n_tile_w // 2, y + 1, 1, 1) for x, y in fig_pos]
            for fig_pos in self.figures_pos
        ]

    def setup(self):
        """
        Run before a new game starts.
        """
        pygame.init()
        self.running = True
        self.sc = pygame.display.set_mode(
            self.sc_size, pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self.game_sc = pygame.Surface(self.game_size)
        self.clock = pygame.time.Clock()

        # resources
        self.sc_bg = pygame.image.load("tetris/img/bg.jpg").convert()
        self.game_bg = pygame.image.load("tetris/img/bg2.jpg").convert()
        self.font_bold = pygame.font.Font("tetris/font/font.ttf", 65)
        self.font_norm = pygame.font.Font("tetris/font/font.ttf", 45)
        self.title_tetris = self.font_bold.render(
            "TETRIS", True, pygame.Color("darkorange")
        )
        self.title_score = self.font_norm.render("score:", True, pygame.Color("green"))
        self.title_record = self.font_norm.render(
            "record:", True, pygame.Color("purple")
        )

        # read record
        self.get_record()

        # initialize game state
        self.init_game_state()

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.dx = -1
            elif event.key == pygame.K_RIGHT:
                self.dx = 1
            elif event.key == pygame.K_UP:
                self.stack = True
            elif event.key == pygame.K_DOWN:
                self.anim_limit = 0
            elif event.key == pygame.K_x:
                self.clockwise = True
            elif event.key == pygame.K_z:
                self.counter_clockwise = True

    def loop(self):
        """
        Handles game logic.
        """
        # move x
        figure_old = deepcopy(self.figure)
        for rect in self.figure:
            rect.x += self.dx
            # figure hits the wall
            if not self.check_borders():
                self.figure = deepcopy(figure_old)
                break

        # move y
        self.anim_count += self.anim_speed
        if self.anim_count > self.anim_limit:
            self.anim_count = 0
            figure_old = deepcopy(self.figure)
            for rect in self.figure:
                rect.y += 1
                # figure hits the bottom
                if not self.check_borders():
                    # update field
                    for rect_old in figure_old:
                        self.field[rect_old.y][rect_old.x] = self.color

                    # shift figure and color
                    self.figure, self.next_figure = (
                        self.next_figure,
                        deepcopy(choice(self.figures)),
                    )
                    self.color, self.next_color = self.next_color, self.get_color()
                    break

        # stack
        if self.stack:
            self.anim_count = 0
            while True:
                figure_old = deepcopy(self.figure)
                for rect in self.figure:
                    rect.y += 1
                if not self.check_borders():
                    # update field
                    for rect_old in figure_old:
                        self.field[rect_old.y][rect_old.x] = self.color

                    # shift figure and color
                    self.figure, self.next_figure = (
                        self.next_figure,
                        deepcopy(choice(self.figures)),
                    )
                    self.color, self.next_color = self.next_color, self.get_color()
                    break

        # rotate clockwhise
        if self.clockwise:
            center = self.figure[0]
            figure_old = deepcopy(self.figure)
            for rect in self.figure:
                x = rect.y - center.y
                y = rect.x - center.x
                rect.x = center.x - x
                rect.y = center.y + y
                if not self.check_borders():
                    self.figure = deepcopy(figure_old)
                    break

        # rotate counter clockwise
        elif self.counter_clockwise:
            center = self.figure[0]
            figure_old = deepcopy(self.figure)
            for rect in self.figure:
                x = rect.y - center.y
                y = rect.x - center.x
                rect.x = center.x + x
                rect.y = center.y - y
                if not self.check_borders():
                    self.figure = deepcopy(figure_old)
                    break

        # delay for full lines on previous iteration
        for _ in range(self.lines):
            pygame.time.wait(200)

        # check lines
        line, self.lines = self.n_tile_h - 1, 0
        for y in range(self.n_tile_h - 1, -1, -1):
            cnt = 0
            for x in range(self.n_tile_w):
                if self.field[y][x]:
                    cnt += 1
                self.field[line][x] = self.field[y][x]
            if cnt < self.n_tile_w:
                line -= 1
            else:
                self.anim_speed += 3
                self.lines += 1

        # compute score
        self.score += self.scores[self.lines]

        self.init_ctrl_state()

    def render(self):
        # draw background
        self.sc.blit(self.sc_bg, (0, 0))
        self.sc.blit(self.game_sc, (20, 20))
        self.game_sc.blit(self.game_bg, (0, 0))

        # draw titles
        self.sc.blit(self.title_tetris, (485, -10))
        self.sc.blit(self.title_score, (535, 780))
        self.sc.blit(
            self.font_norm.render(str(self.score), True, pygame.Color("white")),
            (550, 840),
        )
        self.sc.blit(self.title_record, (525, 650))
        self.sc.blit(
            self.font_norm.render(str(self.record), True, pygame.Color("gold")),
            (550, 710),
        )

        # draw grid
        for rect in self.grid:
            pygame.draw.rect(self.game_sc, (40, 40, 40), rect, 1)

        figure_rect = pygame.Rect(0, 0, self.tile_size - 2, self.tile_size - 2)

        # draw figure
        for rect in self.figure:
            figure_rect.x = rect.x * self.tile_size
            figure_rect.y = rect.y * self.tile_size
            pygame.draw.rect(self.game_sc, self.color, figure_rect)

        # draw field
        for y, row in enumerate(self.field):
            for x, col in enumerate(row):
                if col:
                    figure_rect.x = x * self.tile_size
                    figure_rect.y = y * self.tile_size
                    pygame.draw.rect(self.game_sc, col, figure_rect)

        # draw next figure
        for rect in self.next_figure:
            figure_rect.x = rect.x * self.tile_size + 380
            figure_rect.y = rect.y * self.tile_size + 185
            pygame.draw.rect(self.sc, self.next_color, figure_rect)

        pygame.display.flip()

    def check_game_over(self):
        for x in range(self.n_tile_w):
            if self.field[0][x]:
                self.set_record(self.score)
                self.field = [
                    [0 for _ in range(self.n_tile_w)] for _ in range(self.n_tile_h)
                ]
                self.init_game_state()
                for rect in self.grid:
                    pygame.draw.rect(self.game_sc, self.get_color(), rect)
                    self.sc.blit(self.game_sc, (20, 20))
                    pygame.display.flip()
                    self.clock.tick(200)

    def cleanup(self):
        pygame.quit()
        self.init_game_state()

    def run(self):
        self.setup()

        while self.running:
            for event in pygame.event.get():
                self.handle_event(event)
            self.loop()
            self.render()
            self.check_game_over()
            self.clock.tick(self.fps)

        self.cleanup()

    def init_game_state(self):
        """
        Initialize the game states, e.g. on game start or on game over.
        """
        self.anim_count, self.anim_speed, self.anim_limit = 0, 60, 2000
        self.figure, self.next_figure = (
            deepcopy(choice(self.figures)),
            deepcopy(choice(self.figures)),
        )
        self.field = [[0 for _ in range(self.n_tile_w)] for _ in range(self.n_tile_h)]
        self.color, self.next_color = self.get_color(), self.get_color()
        self.score, self.lines = 0, 0
        self.record = 0
        self.init_ctrl_state()

    def init_ctrl_state(self):
        """
        Clears control state set by the event handler. Run before/after each iteration.
        """
        self.dx = 0
        self.clockwise = False
        self.counter_clockwise = False
        self.anim_limit = 2000
        self.stack = False

    def check_borders(self):
        for rect in self.figure:
            if not (0 <= rect.x < self.n_tile_w):
                return False
            if not (0 <= rect.y < self.n_tile_h):
                return False
            if self.field[rect.y][rect.x]:
                return False
        return True

    def get_record(self):
        try:
            with open("./tetris/record") as f:
                self.record = int(f.readline())
        except FileNotFoundError:
            with open("./tetris/record", "r") as f:
                f.write("0")
                self.record = 0

    def set_record(self, score):
        rec = max(self.record, score)
        with open("./tetris/record", "w") as f:
            f.write(str(rec))

    @staticmethod
    def get_color():
        return (randrange(50, 256), randrange(50, 256), randrange(50, 256))


if __name__ == "__main__":
    tetris = Tetris()
    tetris.run()

    # # override 'handle_event()' and...
    # SOME_EVENT = pygame.USEREVENT + 1
    # pygame.event.post(pygame.event.Event(SOME_EVENT))
