import numpy as np
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

##################### Simulate the Spring Mass System #####################

### assign constants
g = 9.81
k = 20
m = 1
b = 0.2
d=0.2

x0 = -1
x_dot0 = 0
y0=0
y_dot0=0

t_final = 10
fps = 30


def spring_mass_ODE(t, y): 
    return (y[1], g - k*(y[0]-d)/m - b*y[1]/m, y[3], -g - k*(y[2]-d)/m - b*y[3]/m)

sol = solve_ivp(spring_mass_ODE, [0, t_final], (x0, x_dot0,y0,y_dot0), t_eval=np.linspace(0,t_final,t_final*fps+1))

x, x_dot,y,y_dot = sol.y
t = sol.t





##################### Animate the Spring Mass #####################
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.transforms import Affine2D


fig = plt.figure()
ax = fig.add_subplot(aspect='equal')
ax.set_xlim(-1, 1)
ax.set_ylim(-5, 0)

def generate_spring(n):
    data = np.zeros((2,n+2))
    data[:,-1] = [0,-1]
    for i in range(1,n+1):
        data[0,i] = -1/(2*n) if i % 2 else 1/(2*n)
        data[1,i] = -(2*i-1)/(2*n)
    return data

data = np.append(generate_spring(30), np.ones((1,30+2)), axis=0)

ell = 2          
y0 = -(ell + y[0])
y1=y[0]+d
spring = Line2D(data[0,:], data[1,:], color='g')
circle = ax.add_patch(plt.Circle( (0,y0), 0.25, fc='y', zorder=3))
ax.add_line(spring)

def animate(i):
    y = -(ell + x[i])
    circle.set_center((0, y))
    y1=x[i]+d
    stretch_factor = -y
    
    A = Affine2D().scale(8/stretch_factor, stretch_factor).get_matrix()
    data_new = np.matmul(A, data)

    xn = data_new[0,:]
    yn = data_new[1,:]

    spring.set_data(xn, yn)

ani = animation.FuncAnimation(fig, animate, frames=len(t))
ffmpeg_writer = animation.FFMpegWriter(fps=fps)
ani.save('spring_mass_decentered_2d.gif', writer=ffmpeg_writer)
