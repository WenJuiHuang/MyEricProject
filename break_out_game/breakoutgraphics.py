"""
stanCode Breakout Project
Adapted from Eric Roberts's Breakout by
Sonja Johnson-Yu, Kylie Jue, Nick Bowman, 
and Jerry Liao

First, create the BreakoutGraphics class and put whatever function and method I need.
Second, let the ball move in breakout.py
Third, to detect different situations when the ball touches something
"""
from campy.graphics.gwindow import GWindow
from campy.graphics.gobjects import GOval, GRect, GLabel
from campy.gui.events.mouse import onmouseclicked, onmousemoved
import random

BRICK_SPACING = 5  # Space between bricks (in pixels). This space is used for horizontal and vertical spacing.
BRICK_WIDTH = 40  # Height of a brick (in pixels).
BRICK_HEIGHT = 15  # Height of a brick (in pixels).
BRICK_ROWS = 10  # Number of rows of bricks.
BRICK_COLS = 10  # Number of columns of bricks.
BRICK_OFFSET = 50  # Vertical offset of the topmost brick from the window top (in pixels).
BALL_RADIUS = 10  # Radius of the ball (in pixels).
PADDLE_WIDTH = 75  # Width of the paddle (in pixels).
PADDLE_HEIGHT = 15  # Height of the paddle (in pixels).
PADDLE_OFFSET = 50  # Vertical offset of the paddle from the window bottom (in pixels).

INITIAL_Y_SPEED = 7  # Initial vertical speed for the ball.
MAX_X_SPEED = 5  # Maximum initial horizontal speed for the ball.
NUM_LIVES = 3


class BreakoutGraphics:

    def __init__(self, ball_radius=BALL_RADIUS, paddle_width=PADDLE_WIDTH,
                 paddle_height=PADDLE_HEIGHT, paddle_offset=PADDLE_OFFSET,
                 brick_rows=BRICK_ROWS, brick_cols=BRICK_COLS,
                 brick_width=BRICK_WIDTH, brick_height=BRICK_HEIGHT,
                 brick_offset=BRICK_OFFSET, brick_spacing=BRICK_SPACING,
                 title='Breakout'):

        # Create a graphical window, with some extra space
        self.window_width = brick_cols * (brick_width + brick_spacing) - brick_spacing
        self.window_height = brick_offset + 3 * (brick_rows * (brick_height + brick_spacing) - brick_spacing)
        self.window = GWindow(width=self.window_width, height=self.window_height, title=title)

        # Create a paddle
        self.paddle_offset = paddle_offset
        self.paddle = GRect(paddle_width, paddle_height, x=(self.window.width - paddle_width) / 2,
                            y=self.window.height - self.paddle_offset)
        self.paddle.filled = True
        self.paddle.fill_color = 'Black'
        self.window.add(self.paddle)

        # Center a filled ball in the graphical window
        self.ball_r = ball_radius
        self.ball = GOval(self.ball_r, self.ball_r, x=(self.window_width - self.ball_r) / 2,
                          y=(self.window_height - self.ball_r) / 2)
        self.ball.filled = True
        self.ball.fill_color = 'Black'
        self.window.add(self.ball)

        # Default initial velocity for the ball
        self.__dx = random.randint(0, MAX_X_SPEED)
        if random.random() > 0.5:
            self.__dx = - self.__dx
        self.__dy = INITIAL_Y_SPEED

        # Initialize our mouse listeners
        # To create a switch
        self.click_to_start = False
        onmouseclicked(self.start_the_ball)
        onmousemoved(self.event_paddle)

        # Draw bricks
        self.brick_row = brick_rows
        self.brick_col = brick_cols
        self.brick_width = brick_width
        self.brick_height = brick_height
        self.brick_spacing = brick_spacing
        self.brick_offset = brick_offset
        self.brick_s = brick_rows * brick_cols
        for i in range(self.brick_row):
            for j in range(self.brick_col):
                self.brick = GRect(self.brick_width, self.brick_height)
                self.brick.filled = True
                if i < self.brick_row / 5:
                    self.brick.fill_color = 'red'
                elif i < self.brick_row / 5 * 2:
                    self.brick.fill_color = 'Orange'
                elif i < self.brick_row / 5 * 3:
                    self.brick.fill_color = 'Yellow'
                elif i < self.brick_row / 5 * 4:
                    self.brick.fill_color = 'Green'
                elif i <= self.brick_row:
                    self.brick.fill_color = 'Blue'
                self.window.add(self.brick, j * (self.brick_width + self.brick_spacing), self.brick_offset +
                                i * (self.brick_height + self.brick_spacing))

        # Create Label
        self.score = 0
        self.score_label = GLabel('Score : ' + str(self.score))
        self.window.add(self.score_label, 0, self.score_label.height + 2)

        self.life = NUM_LIVES
        self.life_label = GLabel('Life : ' + str(self.life))
        self.window.add(self.life_label, self.window_width - self.life_label.width - 10, self.life_label.height + 2)

    def start_the_ball(self, click):
        # once the mouse being clicked, False--> True
        self.click_to_start = True

    def event_paddle(self, mouse):
        self.paddle.x = mouse.x - self.paddle.width / 2
        self.paddle.y = self.window_height - PADDLE_OFFSET

        # make whole of the paddle stay in the window
        if self.paddle.x + self.paddle.width >= self.window.width:
            self.paddle.x = self.window_width - self.paddle.width
        if self.paddle.x < 0:
            self.paddle.x = 0
    # getters
    def get_dx(self):
        return self.__dx

    def get_dy(self):
        return self.__dy

    def get_click_to_start(self):
        return self.click_to_start

    # Situations when the ball touches the brick or paddle.
    def check_for_collisions(self):
        self.obj = self.window.get_object_at(self.ball.x, self.ball.y)
        if self.obj is not None and self.obj is not self.score_label and self.obj is not self.life_label:
            return self.obj
        else:
            self.obj = self.window.get_object_at(self.ball.x + self.ball.width, self.ball.y)
            if self.obj is not None and self.obj is not self.score_label and self.obj is not self.life_label:
                return self.obj
            else:
                self.obj = self.window.get_object_at(self.ball.x + self.ball.width, self.ball.y + self.ball.height)
                if self.obj is not None and self.obj is not self.score_label and self.obj is not self.life_label:
                    return self.obj
                else:
                    self.obj = self.window.get_object_at(self.ball.x, self.ball.y + self.ball.height)
                    if self.obj is not None and self.obj is not self.score_label and self.obj is not self.life_label:
                        return self.obj

    def reset_ball(self):
        self.window.add(self.ball, x=(self.window_width - self.ball_r) / 2, y=(self.window_height - self.ball_r) / 2)
        self.click_to_start = False
