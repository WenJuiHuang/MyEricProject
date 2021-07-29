"""
File: 
Name:
----------------------
TODO:
"""

from campy.graphics.gobjects import GOval, GRect, GPolygon, GLine
from campy.graphics.gwindow import GWindow

window = GWindow(width=300, height=300)


def main():
    """
    TODO:
    """

    face1 = GPolygon()
    face1.add_vertex((122, 52))
    face1.add_vertex((114, 58))
    face1.add_vertex((104, 53))
    face1.add_vertex((96, 68))
    face1.add_vertex((100, 87))
    face1.add_vertex((114, 112))
    face1.add_vertex((144, 138))
    face1.add_vertex((156, 134))
    face1.add_vertex((182, 110))
    face1.add_vertex((184, 96))
    face1.add_vertex((184, 58))
    face1.add_vertex((170, 78))
    face1.add_vertex((174, 59))
    face1.add_vertex((156, 37))
    face1.add_vertex((115, 55))
    face1.filled = True
    face1.fill_color = 'bisque'
    window.add(face1)


    face2 = GLine(132, 100, 169, 80)
    window.add(face2)

    eye1 = GPolygon()
    eye1.add_vertex((128, 94))
    eye1.add_vertex((156, 75))
    eye1.add_vertex((154, 85))
    eye1.add_vertex((133, 97))
    eye1.filled = True
    eye1.fill_color = 'white'
    window.add(eye1)

    eye2 = GPolygon()
    eye2.add_vertex((107, 90))
    eye2.add_vertex((120, 96))
    eye2.add_vertex((119, 103))
    eye2.add_vertex((113, 102))
    eye2.filled = True
    eye2.fill_color = 'white'
    window.add(eye2)

    eye3 = GOval(5, 5, x=137, y=89)
    eye3.filled = True
    eye3.fill_color = 'Black'
    window.add(eye3)

    eye4 = GOval(4, 4, x=114, y=94)
    eye4.filled = True
    eye4.fill_color = 'Black'
    window.add(eye4)

    eyebrow1 = GPolygon()
    eyebrow1.add_vertex((121, 94))
    eyebrow1.add_vertex((160, 64))
    eyebrow1.add_vertex((170, 66))
    eyebrow1.add_vertex((130, 94))
    window.add(eyebrow1)

    eyebrow2 = GPolygon()
    eyebrow2.add_vertex((117, 90))
    eyebrow2.add_vertex((116, 93))
    eyebrow2.add_vertex((103, 92))
    eyebrow2.add_vertex((100, 83))
    window.add(eyebrow2)

    ear1 = GPolygon()
    ear1.add_vertex((183, 61))
    ear1.add_vertex((193, 50))
    ear1.add_vertex((198, 56))
    ear1.add_vertex((200, 73))
    ear1.add_vertex((193, 86))
    ear1.add_vertex((183, 95))
    ear1.filled = True
    ear1.fill_color = 'moccasin'
    window.add(ear1)

    ear2 = GPolygon()
    ear2.add_vertex((189, 62))
    ear2.add_vertex((193, 55))
    ear2.add_vertex((195, 61))
    ear2.add_vertex((196, 70))
    window.add(ear2)

    ear3 = GPolygon()
    ear3.add_vertex((187, 72))
    ear3.add_vertex((192, 74))
    ear3.add_vertex((191, 82))
    ear3.add_vertex((184, 86))
    window.add(ear3)

    hair0 = GPolygon()
    hair0.add_vertex((120, 47))
    hair0.add_vertex((106, 36))
    hair0.add_vertex((81, 50))
    hair0.add_vertex((74, 89))
    hair0.add_vertex((95, 118))
    hair0.add_vertex((86, 84))
    hair0.add_vertex((101, 55))
    hair0.add_vertex((113, 55))
    hair0.filled = True
    hair0.fill_color = 'yellow'
    window.add(hair0)

    nose1 = GLine(122, 102, 119,118)
    nose2 = GLine(119,118, 128, 115)
    window.add(nose1)
    window.add(nose2)

    mouth = GLine(128, 125, 136, 120)
    window.add(mouth)

    hair1 = GPolygon()
    hair1.add_vertex((78, 52))
    hair1.add_vertex((66, 41))
    hair1.add_vertex((54, 24))
    hair1.add_vertex((67, 31))
    hair1.add_vertex((82, 46))
    hair1.filled = True
    hair1.fill_color = 'yellow'
    window.add(hair1)

    hair2 = GPolygon()
    hair2.add_vertex((82, 48))
    hair2.add_vertex((65, 27))
    hair2.add_vertex((66, 0))
    hair2.add_vertex((82, 27))
    hair2.add_vertex((90, 43))
    hair2.filled = True
    hair2.fill_color = 'yellow'
    window.add(hair2)

    hair3 = GPolygon()
    hair3.add_vertex((90, 43))
    hair3.add_vertex((82, 22))
    hair3.add_vertex((83, 1))
    hair3.add_vertex((98, 0))
    hair3.add_vertex((102, 19))
    hair3.add_vertex((108, 36))
    hair3.add_vertex((97, 36))
    hair3.filled = True
    hair3.fill_color = 'yellow'
    window.add(hair3)

    hair4 = GPolygon()
    hair4.add_vertex((110, 36))
    hair4.add_vertex((115, 12))
    hair4.add_vertex((120, 0))
    hair4.add_vertex((141, 23))
    hair4.add_vertex((129, 45))
    hair4.add_vertex((122, 48))
    hair4.add_vertex((110, 35))
    hair4.filled = True
    hair4.fill_color = 'yellow'
    window.add(hair4)

    hair5 = GPolygon()
    hair5.add_vertex((109, 33))
    hair5.add_vertex((98, 4))
    hair5.add_vertex((121, 3))
    hair5.add_vertex((115, 19))
    hair5.add_vertex((109, 37))
    hair5.filled = True
    hair5.fill_color = 'yellow'
    window.add(hair5)

    hair6 = GPolygon()
    hair6.add_vertex((137, 41))
    hair6.add_vertex((152, 17))
    hair6.add_vertex((158, 3))
    hair6.add_vertex((173, 15))
    hair6.add_vertex((152, 36))
    hair6.filled = True
    hair6.fill_color = 'yellow'
    window.add(hair6)

    hair7 = GPolygon()
    hair7.add_vertex((132, 13))
    hair7.add_vertex((140, 0))
    hair7.add_vertex((154, 17))
    hair7.add_vertex((142, 39))
    hair7.filled = True
    hair7.fill_color = 'yellow'
    window.add(hair7)

    hair8 = GPolygon()
    hair8.add_vertex((180, 44))
    hair8.add_vertex((200, 30))
    hair8.add_vertex((210, 8))
    hair8.add_vertex((230, 0))
    hair8.add_vertex((245, 1))
    hair8.add_vertex((235, 18))
    hair8.add_vertex((194, 50))
    hair8.filled = True
    hair8.fill_color = 'yellow'
    window.add(hair8)

    hair9 = GPolygon()
    hair9.add_vertex((160, 30))
    hair9.add_vertex((172, 13))
    hair9.add_vertex((209, 1))
    hair9.add_vertex((183, 34))
    hair9.add_vertex((174, 28))
    hair9.filled = True
    hair9.fill_color = 'yellow'
    window.add(hair9)

    hair10 = GPolygon()
    hair10.add_vertex((196, 54))
    hair10.add_vertex((227, 26))
    hair10.add_vertex((250, 1))
    hair10.add_vertex((257, 1))
    hair10.add_vertex((258, 6))
    hair10.add_vertex((242, 44))
    hair10.add_vertex((201, 62))
    hair10.filled = True
    hair10.fill_color = 'yellow'
    window.add(hair10)

    hair11 = GPolygon()
    hair11.add_vertex((200, 66))
    hair11.add_vertex((213, 59))
    hair11.add_vertex((237, 57))
    hair11.add_vertex((202, 79))
    hair11.filled = True
    hair11.fill_color = 'yellow'
    window.add(hair11)

    body1 = GPolygon()
    body1.add_vertex((186, 98))
    body1.add_vertex((182, 120))
    body1.add_vertex((159, 153))
    body1.add_vertex((182, 155))
    body1.add_vertex((206, 127))
    body1.add_vertex((221, 118))
    body1.add_vertex((199, 76))
    body1.filled = True
    body1.fill_color = 'moccasin'
    window.add(body1)

    body2 = GPolygon()
    body2.add_vertex((216, 99))
    body2.add_vertex((254, 118))
    body2.add_vertex((235, 138))
    body2.add_vertex((192, 153))
    body2.add_vertex((183, 152))
    body2.add_vertex((217, 99))
    body2.filled = True
    body2.fill_color = 'bisque'
    window.add(body2)

    body3 = GPolygon()
    body3.add_vertex((193, 152))
    body3.add_vertex((235, 137))
    body3.add_vertex((226, 151))
    body3.add_vertex((197, 150))
    body3.filled = True
    body3.fill_color = 'navajowhite'
    window.add(body3)

    body4 = GPolygon()
    body4.add_vertex((220, 151))
    body4.add_vertex((196, 150))
    body4.add_vertex((170, 154))
    body4.add_vertex((149, 160))
    body4.add_vertex((131, 167))
    body4.add_vertex((124, 178))
    body4.add_vertex((121, 190))
    body4.add_vertex((141, 185))
    body4.add_vertex((170, 180))
    body4.add_vertex((200, 166))
    body4.add_vertex((220, 155))
    body4.filled = True
    body4.fill_color = 'bisque'
    window.add(body4)

    body5 = GPolygon()
    body5.add_vertex((129, 170))
    body5.add_vertex((114, 164))
    body5.add_vertex((89, 169))
    body5.add_vertex((86, 172))
    body5.add_vertex((93, 187))
    body5.add_vertex((120, 189))
    body5.filled = True
    body5.fill_color = 'moccasin'
    window.add(body5)

    body6 = GPolygon()
    body6.add_vertex((92, 167))
    body6.add_vertex((105, 158))
    body6.add_vertex((136, 139))
    body6.add_vertex((140, 149))
    body6.add_vertex((138, 163))
    body6.add_vertex((129, 169))
    body6.add_vertex((116, 165))
    body6.filled = True
    body6.fill_color = 'navajowhite'
    window.add(body6)

    body7 = GPolygon()
    body7.add_vertex((138, 134))
    body7.add_vertex((138, 162))
    body7.add_vertex((158, 154))
    body7.add_vertex((166, 131))
    body7.filled = True
    body7.fill_color = 'navajowhite'
    window.add(body7)

    body8 = GPolygon()
    body8.add_vertex((87, 170))
    body8.add_vertex((94, 184))
    body8.add_vertex((108, 188))
    body8.add_vertex((127, 188))
    body8.add_vertex((158, 182))
    body8.add_vertex((183, 175))
    body8.add_vertex((204, 165))
    body8.add_vertex((220, 153))
    body8.add_vertex((199, 192))
    body8.add_vertex((141, 195))
    body8.add_vertex((91, 194))
    body8.add_vertex((68, 192))
    body8.add_vertex((80, 175))
    body8.filled = True
    body8.fill_color = 'navy'
    window.add(body8)

    body9 = GPolygon()
    body9.add_vertex((257, 114))
    body9.add_vertex((247, 121))
    body9.add_vertex((236, 133))
    body9.add_vertex((227, 146))
    body9.add_vertex((219, 158))
    body9.add_vertex((207, 175))
    body9.add_vertex((197, 192))
    body9.add_vertex((202, 192))
    body9.add_vertex((209, 181))
    body9.add_vertex((218, 167))
    body9.add_vertex((229, 153))
    body9.add_vertex((242, 136))
    body9.add_vertex((254, 123))
    body9.add_vertex((256, 123))
    body9.filled = True
    body9.fill_color = 'orange'
    window.add(body9)

    body10 = GPolygon()
    body10.add_vertex((257, 122))
    body10.add_vertex((204, 192))
    body10.add_vertex((257, 192))
    body10.add_vertex((257, 125))
    body10.add_vertex((257, 121))
    body10.filled = True
    body10.fill_color = 'black'
    window.add(body10)

    body11 = GPolygon()
    body11.add_vertex((138, 135))
    body11.add_vertex((132, 135))
    body11.add_vertex((93, 157))
    body11.add_vertex((44, 176))
    body11.add_vertex((41, 191))
    body11.add_vertex((69, 191))
    body11.add_vertex((82, 170))
    body11.add_vertex((103, 158))
    body11.add_vertex((119, 149))
    body11.add_vertex((139, 140))
    body11.filled = True
    body11.fill_color = 'orange'
    window.add(body11)


if __name__ == '__main__':
    main()
