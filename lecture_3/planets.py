import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import matplotlib.animation
# %%
center_of_gravity = np.array([0., 2., 0.]);
G = 1.
dt = 0.01
mass_planet = 1.
mass_sun = 10.

def gravity (position):

	to_center = (center_of_gravity - position)
	r = np.linalg.norm(to_center)
	result = to_center * ((G * mass_sun) / (r**3))
	return result


class Satelites:
	def __init__(self, position, velocity, euler, color):
		self.position = position
		self.velocity = velocity
		self.euler = euler
		self.color = color

s1 = Satelites(np.array([1., 2., 1.]), np.array([1., 0., -3.]), 'semi', 'r')
s2 = Satelites(np.array([1., 2., 1.]), np.array([1., 0., -3.]), 'expl', 'b')

planets = [s1, s2]
iters = 10000
coords_s1 = np.ndarray([3,iters])
coords_s2 = np.ndarray([3,iters])
def v(x, dt, vn):
	return vn + gravity(x) * dt

def a (v, h, xn):
	return gravity(xn) + gravity(xn + v*dt) * dt

def anim (planets, i = None):
	for s in planets:
		xn = s.position.copy()
		vn = s.velocity.copy()

		if s.euler == 'expl':
			s.position = xn + dt * v(xn, 0, vn)
			s.velocity = vn + dt * a(vn, 0, xn)
			if i != None :
				coords_s2[:, i] = s.position

		if s.euler == 'semi':
			s.position = xn + dt * v(xn, dt, vn)
			s.velocity = vn + dt * a(vn, 0, xn)
			if i != None :
				coords_s1[:, i] = s.position

# %%

for i in range (iters):
	anim(planets, i)
fig = plt.figure(figsize=(8,6))
bx = fig.add_subplot(111, projection='3d')

bx.plot(coords_s1[0,:], coords_s1[1,:], coords_s1[2,:], color = s1.color)
bx.plot(coords_s2[0,:], coords_s2[1,:], coords_s2[2,:], color = s2.color)

fig.show()

# %%

r = 0.5
u_sphere, v_sphere = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
x = r*np.cos(u_sphere) * np.sin(v_sphere)
y = r*np.sin(u_sphere) * np.sin(v_sphere)
z = r*np.cos(v_sphere)

def update(val):
	global wframe_1
	global wframe_2
	anim(planets)
	idx = int(val)
	s = planets[0]
	wframe_1._offsets3d = ([s.position[0]], [s.position[1]], [s.position[2]])
	s = planets[1]
	wframe_2._offsets3d = ([s.position[0]], [s.position[1]], [s.position[2]])
	return wframe_1, wframe_2

# %%
fig = plt.figure(figsize=(8,6))
bx = fig.add_subplot(111, projection='3d')

wframe_1 = None
wframe_2 = None
max_radius = 20
bx.set_xlim(-max_radius, max_radius)
bx.set_ylim(-max_radius, max_radius)
bx.set_zlim(-max_radius, max_radius)

bx.set_xlabel('x')
bx.set_ylabel('y')
bx.set_zlabel('z')
bx.view_init(azim=-90, elev=90)

X,Y,Z = x + center_of_gravity[0],y + center_of_gravity[1] , z + center_of_gravity[2]
wframe_1 = bx.scatter(s1.position[0], s1.position[1], s1.position[2], color = s1.color)
wframe_2 = bx.scatter(s2.position[0], s2.position[1], s2.position[2], color = s2.color)
wframe_3 = bx.plot_surface(X, Y, Z, rstride=4, cstride=4, color = 'y')

# %%
ani = matplotlib.animation.FuncAnimation(fig, update, frames=range(1000), interval = 0, blit=False )

fig.show()