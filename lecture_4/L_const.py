import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import matplotlib.animation
n_points = 11
lin_size = 5
x = y = np.linspace(-lin_size, lin_size, n_points)
x_1, y_1 = np.meshgrid(x, y)
sz = n_points*n_points
data = np.ndarray([sz * 2, 4], dtype = 'float')
data[:, 0] = 1
data[:sz, 1] = np.reshape(x_1, (sz, 1))[:,0]
data[:sz, 2] = np.reshape(y_1, (sz, 1))[:,0]
data[:sz, 3] = -lin_size

data[sz:, 1] = np.reshape(x_1, (sz, 1))[:,0]
data[sz:, 2] = np.reshape(y_1, (sz, 1))[:,0]
data[sz:, 3] = lin_size
L = np.array([5000.0, 5010.0, 100.0], dtype = 'float')

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
print(mass_center)

new_coord = np.add(data, [0, -mass_center[1], -mass_center[2],-mass_center[3]])

from mpl_toolkits.mplot3d import Axes3D

fig0 = plt.figure()
ax = fig0.add_subplot(111, projection='3d')
ax.scatter(new_coord[:,1], new_coord[:,2], new_coord[:,3], c='r', marker='o', label = 'system')

ax.scatter([0], [0], [0], label = 'mass center')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
plt.show()
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

def simulate_system (t, dt, coord, L):
    steps = int(t/dt)
    system_observ = np.zeros([steps, coord.shape[0], coord.shape[1]], dtype = 'float') # the coordinate of system on every step
    inert_tenzor_observ = np.zeros([steps, 3, 3], dtype = 'float')
    omega_observ = np.zeros([steps, 1, 3], dtype = 'float') # the angle velocity on each step
    K_observ = np.zeros(steps - 1, dtype = 'float')

    system_observ[0] = coord

    #the simulation begin
    for i in range(steps - 1):
        inert_tenzor_observ[i] = inert_tenzor_calc(system_observ[i])
        omega_observ[i] = np.dot(np.linalg.inv(inert_tenzor_observ[i]), L.transpose())
        K_observ[i] = 1/2* (np.dot(omega_observ[i], np.dot(inert_tenzor_observ[i], omega_observ[i].transpose())))
        system_observ[i + 1] = sys_evol(system_observ[i], omega_observ[i], dt)
    return omega_observ, K_observ, system_observ

t = 10
dt = 0.01
w1, K1, sys1 = simulate_system(t, dt, new_coord, L)

# %%
import matplotlib.animation

fig1 = plt.figure()

dx = fig1.add_subplot(111, projection='3d')
sc = dx.scatter(sys1[:, : , 1], sys1[:, : , 2], sys1[:, : , 3], label = 'system evolution')
dx.set_xlabel("x")
dx.set_ylabel("y")
dx.set_zlabel("z")
dx.legend()
def update(i):
    sc._offsets3d = (sys1[i, : , 1], sys1[i, : , 2], sys1[i, : , 3])


anim = matplotlib.animation.FuncAnimation(fig1, update, frames=int(t/dt), interval=70)

plt.tight_layout()
plt.show()