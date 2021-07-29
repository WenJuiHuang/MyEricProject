"""
File: draw_line py
Name: draw_line
-------------------------
First, distinguish the movements of the first click and the second.
Second, try to save the first (x,y).
"""

from campy.graphics.gobjects import GOval, GLine
from campy.graphics.gwindow import GWindow
from campy.gui.events.mouse import onmouseclicked

# Constant
SIZE = 10

# variable
window = GWindow(800, 500, title='draw_line py')
click = 0

#critical point, used as a tool to save the (x,y) of the first click
start_point = GOval(SIZE, SIZE)


def main():
    """
    This program creates lines on an instance of GWindow class.
    There is a circle indicating the user’s first click. A line appears
    at the condition where the circle disappears as the user clicks
    on the canvas for the second time.
    """

    onmouseclicked(event)


def event(mouse):
    global click # Since 'start_point' is an object, it doesn't need global to summon it.
    click += 1

    if click % 2 != 0:
        window.add(start_point, x=mouse.x - (SIZE / 2), y=mouse.y - (SIZE / 2))

    else:
        window.remove(start_point)
        line = GLine(start_point.x, start_point.y, mouse.x, mouse.y)
        window.add(line)



if __name__ == "__main__":
    main()
