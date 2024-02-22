import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from matplotlib.transforms import Affine2D

g = 9.81
k = 20
m = 1
b = 0.25

# Начальные условия для двух точек и их скоростей (x1, x2, x1_dot, x2_dot)
x1_0 = -2
x2_0 = 2
x1_dot0 = 0
x2_dot0 = 0

def two_spring_masses_ODE(t, y):
    return [
        y[1],  # x1_dot
        (k/m) * (y[2] - y[0]) - b * y[1],
        y[3],  # x2_dot
        -k/m * (y[2] - y[0]) - b * y[3]
    ]


sol = solve_ivp(two_spring_masses_ODE, [0, 10], [x1_0, x1_dot0, x2_0, x2_dot0], 
                t_eval=np.linspace(0, 10, 10*30))

positions = sol.y
times = sol.t

fig = plt.figure()
ax = fig.add_subplot(aspect='equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-1, 1)

spring = Line2D([], [], color='g')
ax.add_line(spring)

circle1 = ax.add_patch(plt.Circle((positions[0, 0], 0), 0.2, fc='orange', ec='none', zorder=3))
circle2 = ax.add_patch(plt.Circle((positions[2, 0], 0), 0.2, fc='blue', ec='none', zorder=3))

def animate(i):
    x1, x1_dot, x2, x2_dot = positions[:, i]
    
    data = np.array([[x1, x2], [0, 0]])
    spring.set_data(data[0, :], data[1, :])

    circle1.set_center((x1, 0))
    circle2.set_center((x2, 0))

ani = animation.FuncAnimation(fig, animate, frames=len(times))
ffmpeg_writer = animation.FFMpegWriter(fps=30)
ani.save('simulation_in_1D_with_springs.gif', writer=ffmpeg_writer)