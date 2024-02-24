import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

g = 9.81
k = 20
m = 1
b = 0.25

x1_0 = -2
y1_0 = -2
x2_0 = 2
y2_0 = 2
x1_dot0 = 0
y1_dot0 = 0
x2_dot0 = 0
y2_dot0 = 0

def two_spring_masses_ODE(t, y):
    x1, y1, x1_dot, y1_dot, x2, y2, x2_dot, y2_dot = y
    dx1_dt = x1_dot
    dy1_dt = y1_dot
    dx1_dot_dt = (k/m) * (x2 - x1) - b * x1_dot
    dy1_dot_dt = (k/m) * (y2 - y1) - b * y1_dot
    dx2_dt = x2_dot
    dy2_dt = y2_dot
    dx2_dot_dt = -k/m * (x2 - x1) - b * x2_dot
    dy2_dot_dt = -k/m * (y2 - y1) - b * y2_dot
    return [dx1_dt, dy1_dt, dx1_dot_dt, dy1_dot_dt, dx2_dt, dy2_dt, dx2_dot_dt, dy2_dot_dt]

sol = solve_ivp(two_spring_masses_ODE, [0, 10], [x1_0, y1_0, x1_dot0, y1_dot0, x2_0, y2_0, x2_dot0, y2_dot0], 
                t_eval=np.linspace(0, 10, 10*30))

positions = sol.y
times = sol.t

fig = plt.figure()
ax = fig.add_subplot(aspect='equal')
ax.set_xlim(-3, 3) 
ax.set_ylim(-3, 3)

spring = Line2D([], [], color='g')
ax.add_line(spring)

circle1 = ax.add_patch(plt.Circle((positions[0, 0], positions[1, 0]), 0.2, fc='orange', ec='none', zorder=3))
circle2 = ax.add_patch(plt.Circle((positions[4, 0], positions[5, 0]), 0.2, fc='blue', ec='none', zorder=3))

def animate(i):
    x1, y1, x1_dot, y1_dot, x2, y2, x2_dot, y2_dot = positions[:, i]
    
    x_data = np.array([[x1, x2], [y1, y2]])
    spring.set_data(x_data[0, :], x_data[1, :])

    circle1.set_center((x1, y1))
    circle2.set_center((x2, y2))

ani = animation.FuncAnimation(fig, animate, frames=len(times))
ffmpeg_writer = animation.FFMpegWriter(fps=30)
ani.save('simulation_in_2D_diagonal.gif', writer=ffmpeg_writer)

plt.show()