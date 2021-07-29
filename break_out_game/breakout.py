"""
stanCode Breakout Project
Adapted from Eric Roberts's Breakout by
Sonja Johnson-Yu, Kylie Jue, Nick Bowman,
and Jerry Liao.


"""

from campy.gui.events.timer import pause
from breakoutgraphics import BreakoutGraphics

FRAME_RATE = 1000 / 120  # 120 frames per second
NUM_LIVES = 3  # Number of attempts


def main():
    graphics = BreakoutGraphics()

    # Add animation loop here!

    start_x = graphics.get_dx()
    start_y = graphics.get_dy()
    bricks = graphics.brick_s
    score = graphics.score
    score_label = graphics.score_label
    life_label = graphics.life_label
    lives = NUM_LIVES

    while True:
        start = graphics.get_click_to_start()

        if start:

            # two ways to end the game
            if bricks == 0:
                break
            if lives == 0:
                break

            graphics.ball.move(start_x, start_y)
            # bounce back when the ball touches walls
            if graphics.ball.x < 0 or graphics.ball.x + graphics.ball.width >= graphics.window.width:
                start_x = -start_x
            if graphics.ball.y < 0:
                start_y = -start_y

            # the situation when user miss catching the ball with paddle
            if graphics.ball.y + graphics.ball.height >= graphics.window_height:
                graphics.reset_ball()
                lives -= 1
                life_label.text = 'Life: ' + str(lives)

            # Situations when the ball touches the brick or paddle.
            paddle_or_brick = graphics.check_for_collisions()

            if paddle_or_brick is not None and paddle_or_brick is not score_label and paddle_or_brick is not life_label:
                if paddle_or_brick is graphics.paddle:
                    if start_y > 0:
                        start_y = -start_y
                else:
                    start_y = -start_y
                    graphics.window.remove(graphics.obj)
                    bricks -= 1
                    score += 1
                    # to change the text of Glabel
                    score_label.text = 'Score: ' + str(score)


        pause(FRAME_RATE)


if __name__ == '__main__':
    main()
