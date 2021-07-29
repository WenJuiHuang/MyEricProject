from campy.graphics.gimage import GImage
from campy.gui.events.mouse import onmouseclicked
from campy.graphics.gwindow import GWindow

window = GWindow(width=640, height=480)


def main():
    onmouseclicked(event)
    img = GImage('Goku.jpg')
    window.add(img)



def event(mouse):
    print(mouse.x, mouse.y)
if __name__ == '__main__':
    main()