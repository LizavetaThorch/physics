import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import matplotlib.animation
# %%
def lorenz(x, y, z, s=10, r=28, b=2.667):
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

dt = 0.01

# Set initial values
xs, ys, zs = (-1., 1, 0.)
num_points = 1000

data = np.ndarray([3,num_points], dtype = np.float)
data[0,:] = xs
data[1,:] = xs
data[2,:] = xs
data += np.random.rand(3,num_points)/1000
# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
def anim(data):
    for i in range(num_points):
        xs, ys, zs = data[0, i], data[1, i], data[2, i]
        x_dot, y_dot, z_dot = lorenz(xs, ys, zs)
        data[0, i] = xs + (x_dot * dt)
        data[1, i] = ys + (y_dot * dt)
        data[2, i] = zs + (z_dot * dt)

#%%
# Plot
def update(val):
    global wframe_1, data
    anim(data)
    wframe_1._offsets3d = (data[0,:], data[1,:], data[2,:])
    return wframe_1# %%
fig = plt.figure(figsize=(8,6))
bx = fig.add_subplot(111, projection='3d')

max_radius = 30
bx.set_xlim(-max_radius, max_radius)
bx.set_ylim(-max_radius, max_radius)
bx.set_zlim(0, 2*max_radius)

wframe_1 = None
bx.set_xlabel('x')
bx.set_ylabel('y')
bx.set_zlabel('z')
# bx.view_init(azim=-90, elev=90)

wframe_1 = bx.scatter(data[0,:], data[1,:], data[2,:])
# update (10)

ani = matplotlib.animation.FuncAnimation(fig, update, frames=range(100000), interval = 1, blit=False )

fig.show()
