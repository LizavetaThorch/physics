import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import imageio

originX = 0
originY = 0
radius = 1

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

circle = patches.Circle((originX, originY), radius, edgecolor='black', facecolor='none')
ax.add_patch(circle)

point, = ax.plot([], [], 'bo')

def update(frame):
    angle = np.radians(frame)
    x = originX + np.cos(angle) * radius
    y = originY + np.sin(angle) * radius
    point.set_data(x, y)
    return point,

ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)

ax.set_xlabel('X')
ax.set_ylabel('Y')

ani.save('circle_animation.gif', writer='imagemagick')

plt.show()
