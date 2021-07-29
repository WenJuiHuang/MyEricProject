"""
File: bouncing_ball.py
Name: bouncing ball
-------------------------
To create a program that makes a ball bouncing like it does in the reality.
Try to define the function of onmouseclicked, so that every click starts the ball again.
"""

from campy.graphics.gobjects import GOval
from campy.graphics.gwindow import GWindow
from campy.gui.events.timer import pause
from campy.gui.events.mouse import onmouseclicked

# Constant
VX = 3
DELAY = 10
GRAVITY = 1
SIZE = 20
REDUCE = 0.9
START_X = 30
START_Y = 40

# Global variables
window = GWindow(800, 500, title='bouncing_ball.py')
click_to_bounce = True
count = 0
ball = GOval(SIZE, SIZE, x=START_X, y=START_Y)
ball.filled = True
ball.fill_color = 'Black'
window.add(ball)


def main():
    """
    This program simulates a bouncing ball at (START_X, START_Y)
    that has VX as x velocity and 0 as y velocity. Each bounce reduces
    y velocity to REDUCE of itself.
    """

    onmouseclicked(bouncing_ball)


def bouncing_ball(mouse):
    global click_to_bounce, count
    vy = 0
    if click_to_bounce and count < 3:
        # when the program is processing, the mouseclicked function won't work
        click_to_bounce = False
        while True:
            ball.move(VX, vy)
            vy += GRAVITY
            if ball.y + SIZE >= window.height:
                vy *= -REDUCE

            if ball.x + SIZE > window.width:
                count += 1
                window.add(ball, x=START_X, y=START_Y)
                click_to_bounce = True
                break
            pause(DELAY)


if __name__ == "__main__":
    main()
