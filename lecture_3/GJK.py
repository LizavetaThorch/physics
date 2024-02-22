import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.pyplot import figure

plt.rcParams["figure.figsize"] = (20,20)

import skgeom as sg
from skgeom.draw import draw
p0 = sg.Polygon([sg.Point2(-0.3, 0), sg.Point2(0, 0.3), sg.Point2(0.3, 0)])

p1 = sg.Polygon([sg.Point2(0, 2), sg.Point2(6, 2), sg.Point2(4, 6)])
p2 = sg.Polygon([sg.Point2(12, 2), sg.Point2(5, 4), sg.Point2(6, 7), sg.Point2(12, 7)])
draw(p1, facecolor='red')
draw(p2, facecolor='blue')
#%%

p2_m = sg.Polygon([sg.Point2(-10, -2), sg.Point2(-5, -4), sg.Point2(-6, -7), sg.Point2(-12, -7)])
p1.orientation()
p2.reverse_orientation()
p2_m.reverse_orientation()
from skgeom import minkowski
result = minkowski.minkowski_sum(p1, p2_m)

draw(p1, facecolor='red')
draw(p2, facecolor='blue')
draw(result)
draw(p0, facecolor='red')