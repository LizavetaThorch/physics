import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
# %%
def get_new_system(n_points, lin_size, mass):
    x = y = np.linspace(-lin_size, lin_size, n_points)
    x_1, y_1 = np.meshgrid(x, y)
    sz = n_points*n_points
    data = np.ndarray([sz * 2, 4], dtype = 'float')
    data[:, 0] = mass
    data[:sz, 1] = np.reshape(x_1, (sz, 1))[:,0]
    data[:sz, 2] = np.reshape(y_1, (sz, 1))[:,0]
    data[:sz, 3] = -lin_size

    data[sz:, 1] = np.reshape(x_1, (sz, 1))[:,0]
    data[sz:, 2] = np.reshape(y_1, (sz, 1))[:,0]
    data[sz:, 3] = lin_size
    #  m  x  y  z
    mass_center = np.array([0, 0, 0, 0], dtype ='float')
    for row in data :
        mass_center[0] += row[0]
        mass_center[1] += row[0]*row[1]
        mass_center[2] += row[0]*row[2]
        mass_center[3] += row[0]*row[3]
    # print(mass_center)
    mass_center[1] /= mass_center[0]
    mass_center[2] /= mass_center[0]
    mass_center[3] /= mass_center[0]

    system = np.add(data, [0, -mass_center[1], -mass_center[2],-mass_center[3]])
    return system
# %%
#calculating the inertia tenzor
def inert_tenzor_calc (coord) :
    inert_tenzor = np.zeros([3,3])
    for row in coord:
        inert_tenzor[0,0] += row[0]*(row[2]**2+row[3]**2)
        inert_tenzor[1,0] += -row[0]*(row[2]*row[1])
        inert_tenzor[2,0] += -row[0]*(row[3]*row[1])

        inert_tenzor[0,1] += -row[0]*(row[1]*row[2])
        inert_tenzor[1,1] += row[0]*(row[1]**2+row[3]**2)
        inert_tenzor[2,1] += -row[0]*(row[3]*row[2])

        inert_tenzor[0,2] += -row[0]*(row[1]*row[3])
        inert_tenzor[1,2] += -row[0]*(row[2]*row[3])
        inert_tenzor[2,2] += row[0]*(row[1]**2+row[2]**2)
    return inert_tenzor

#recaluculete r of every point
def sys_evol (coord, w, dt) :
    u = np.zeros(coord.shape, dtype = 'float')
    u[:,1:] = np.cross(coord[:,1:], w)
    evol_coord = coord + u*dt
    return evol_coord

def step (dt, system, L, inert_tenzor):
    omega = np.dot(np.linalg.inv(inert_tenzor), L.transpose())
    system = sys_evol(system, omega, dt)
    return system

# %%
def rotate_each_vector_by_matrix(array_of_vector, rot_matrix):
	result = np.zeros_like(array_of_vector)
	i = 0
	for vector in array_of_vector:
		result[i] = np.dot(rot_matrix, vector)
		i+=1
	return result

def get_xyz_ellipsoid(A, center):
    U, s, rotation = np.linalg.svd(A)
    radii = 100.0/np.sqrt(s)
    # now carry on with EOL's answer
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center
    return x, y, z
# %%
import matplotlib.animation
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
bx = fig.add_subplot(1, 2, 2, projection='3d')

t = 10
dt = 0.01
L = np.array([5000.0, 5010.0, 100.0], dtype = 'float')
mass = 1
points = 11
length = 5
sys = get_new_system(points, length, mass)
inert_tenzor = inert_tenzor_calc(sys)
sys = step(dt, sys, L, inert_tenzor)

sc = ax.scatter(sys[: , 1], sys[: , 2], sys[: , 3], color = 'b')

X,Y,Z = get_xyz_ellipsoid(inert_tenzor, np.array([0.,0.,0.]))
wframe_1 = bx.plot_surface(X, Y, Z, rstride=4, cstride=4, cmap=matplotlib.cm.coolwarm, label = 'inertia tenzor')

ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z");
bx.set_xlabel("x"), bx.set_ylabel("y"), bx.set_zlabel("z");

bx.set_xlim(-10, 10), bx.set_ylim(-10, 10), bx.set_zlim(-10, 10)
ax.set_xlim(-10, 10), ax.set_ylim(-10, 10), ax.set_zlim(-10, 10)

def update(i):
    global sys
    global sc
    global wframe_1
    if wframe_1:
        bx.collections.remove(wframe_1)
    inert_tenzor = inert_tenzor_calc(sys)
    sys = step(dt, sys, L, inert_tenzor)
    sc.remove()
    sc = ax.scatter(sys[: , 1], sys[: , 2], sys[: , 3], color = 'b')
    X,Y,Z = get_xyz_ellipsoid(inert_tenzor, np.array([0.,0.,0.]))
    wframe_1 = bx.plot_surface(X, Y, Z, rstride=4, cstride=4, cmap=matplotlib.cm.coolwarm)
    return sc, wframe_1

def change_mass(val):
    global sys
    global points, length, mass
    mass = val
    sys = get_new_system(points, length, mass)

def change_length(val):
    global sys
    global points, length, mass
    length = val
    sys = get_new_system(points, length, mass)

def change_points(val):
    global sys
    global points, length, mass
    points = int(val)
    sys = get_new_system(points, length, mass)

# plt.show()
import matplotlib.animation

anim = matplotlib.animation.FuncAnimation(fig, update, frames=1, interval=0)
axmass = plt.axes([0.25, 0.1, 0.65, 0.03])
slmass = Slider(axmass, 'mass', 0.01, 10, valinit=1, valstep=0.01)
slmass.on_changed(change_mass)

axpoints = plt.axes([0.25, 0.15, 0.65, 0.03])
slpoints = Slider(axpoints, 'points', 1, 15, valinit=points, valstep=1)
slpoints.on_changed(change_points)

axlength = plt.axes([0.25, 0.2, 0.65, 0.03])
sllength = Slider(axlength, 'length', 0.01, 10, valinit=length, valstep=0.01)
sllength.on_changed(change_length)

plt.show()
# %%