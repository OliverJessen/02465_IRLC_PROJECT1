# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.utils.graphics_util_pygame import UpgradedGraphicsUtil, rotate_around
import numpy as np

""" This file contains code you can either use (or not) to render the R2D2 robot. class is already called correctly by your R2D2 class, 
and you don't really have to think too carefully about what the code does unless you want to R2D2 to look better.
"""


class R2D2Viewer(UpgradedGraphicsUtil):
    def __init__(self, x_target = (0,0)):
        self.x_target = x_target
        width = 800
        self.scale = width / 1000
        xlim = 3
        self.dw = self.scale * 0.1
        super().__init__(screen_width=width, xmin=-xlim, xmax=xlim, ymin=xlim, ymax=-xlim, title='R2D2')
        self.xlim = xlim
    def render(self):
        # self.
        self.draw_background(background_color=(255, 255, 255))
        dw = self.dw
        self.line("t1", (-self.xlim, 0), (self.xlim, 0), width=1, color=(0,) * 3)
        self.line("t1", (0, -self.xlim), (0, self.xlim), width=1, color=(0,) * 3)


        self.circle("r2d2", pos=(self.x[0], self.x[1]), r=24, outlineColor=(100, 100, 200), fillColor=(100, 100, 200))
        self.circle("r2d2", pos=(self.x[0], self.x[1]), r=20, outlineColor=(100, 100, 200), fillColor=(150, 150, 255))
        self.circle("r2d2", pos=(self.x[0], self.x[1]), r=2, outlineColor=(100, 100, 200), fillColor=(0,)*3)

        dx = 0.13
        dy = dx/2.5
        wheel = [(-dx, dy), (dx, dy), (dx, -dy), (-dx, -dy) ]
        ddy = 0.20
        w1 = [ (x, y + ddy) for x, y in wheel]
        w1 = rotate_around(w1, (0,0), angle=self.x[2] / np.pi * 180)

        w2 = [(x, y - ddy) for x, y in wheel]
        w2 = rotate_around(w2, (0, 0), angle=self.x[2] / np.pi * 180)


        self.polygon("wheel1", coords=[ (x +  self.x[0], self.x[1] + y) for x, y in w1], filled=True, fillColor=(200,)*3, outlineColor=(100,)*3, closed=True)
        self.polygon("wheel2", coords=[ (x +  self.x[0], self.x[1] + y) for x, y in w2], filled=True, fillColor=(200,)*3, outlineColor=(100,)*3, closed=True)

        dc = 0.1
        xx = self.x_target[0]
        yy = self.x_target[1]
        self.line("t1", (xx-dc, yy+dc), (xx+dc, yy-dc), width=4, color=(200, 100, 100))
        self.line("t1", (xx-dc, yy-dc), (xx+dc, yy+dc), width=4, color=(200, 100, 100))


    def update(self, x):
        self.x = x
