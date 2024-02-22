import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.pyplot import figure

plt.rcParams["figure.figsize"] = (10,10)
# %%

w = 2* np.pi
d = 0.5

A = np.array([[0., 1],[-w**2,  - 2*d*w]])

dt = 0.01
T = 100
iters = int(T/dt)
x0 = np.array([2.,0.])

# %%  Forward Euler

xF = np.zeros([2, iters], dtype = np.float)
tF = np.zeros([iters], dtype = np.float)
xF[:,0] = x0
tF[0] = 0
for i in range (iters - 1):
    tF[i + 1] = i * dt
    xF[:, i + 1] = (np.eye(2) + A * dt).dot(xF[:, i])

plt.plot(tF, xF[0,:], c = 'r')
plt.show()

# %% Backward Euler

xB = np.zeros([2, iters], dtype = np.float)
tB = np.zeros([iters], dtype = np.float)
xB[:,0] = x0
tB[0] = 0
for i in range (iters - 1):
    tB[i + 1] = i * dt
    xB[:, i + 1] = np.linalg.inv(np.eye(2) - A * dt).dot(xB[:, i])



plt.plot(tB, xB[0,:], c = 'b')
plt.show()

#%%
plt.plot(tF, xF[0,:], c = 'r', label = "Forward Euler")
plt.plot(tB, xB[0,:], c = 'b', label = "Backward Euler")
plt.legend(loc="upper right")
plt.show()


# %%

from scipy.integrate import solve_ivp
def spring(t, x): return A.dot(x)
sol = solve_ivp(spring, [0, T], x0)
sol.t.shape
plt.plot(sol.t, sol.y[0,:], c = 'g', label = "Solve IVP")
plt.plot(tF, xF[0,:], c = 'r', label = "Forward Euler")
plt.plot(tB, xB[0,:], c = 'b', label = "Backward Euler")
plt.legend(loc="upper right")
plt.show()