import numpy as np
class Circle:
    def __init__(self, radius, mass, position, velocity, color):
        self.radius = radius
        self.mass = mass
        self.velocity = velocity
        self.color = color
        self.position = position


c1 = Circle(30.,1.,np.array([-50.,0.]), np.array([50.,0.]), 'r')
c2 = Circle(20.,1.,np.array([50.,30.]), np.array([-50.,0.]), 'b')
def collide(c1,c2):
    dp = c1.position - c2.position

    y_l = dp/np.linalg.norm(dp)
    x_l = np.array([0.,0.])
    if y_l[1] == 0:
        x_l = np.array([0.,1.])
    else:
        x_l = np.array([1., - y_l[0]/y_l[1]])
        x_l = x_l/np.linalg.norm(x_l)
    v1 = np.array([0.,0.])
    v2 = np.array([0.,0.])
    v1[0] = c1.velocity.dot(x_l)
    v1[1] = c1.velocity.dot(y_l)
    v2[0] = c2.velocity.dot(x_l)
    v2[1] = c2.velocity.dot(y_l)
    v1_n = np.array([v1[0], 0.])
    v2_n = np.array([v2[0], 0.])
    m1 = c1.mass
    m2 = c2.mass
    v1_n[1] = (2*m2*v2[1] + v1[1]*(m1-m2))/(m1+m2)
    v2_n[1] = (2*m1*v1[1] + v2[1]*(m2-m1))/(m1+m2)
    x = np.array([1.,0.])
    y = np.array([0.,1.])
    x_p = np.array([0.,0.])
    y_p = np.array([0.,0.])
    x_p[0] = x.dot(x_l)
    x_p[1] = x.dot(y_l)
    y_p[0] = y.dot(x_l)
    y_p[1] = y.dot(y_l)

    v1_g = np.array([0.,0.])
    v2_g = np.array([0.,0.])

    v1_g[0] = v1_n.dot(x_p)
    v1_g[1] = v1_n.dot(y_p)
    v2_g[0] = v2_n.dot(x_p)
    v2_g[1] = v2_n.dot(y_p)
    c1.velocity = v1_g
    c2.velocity = v2_g
    return c1, c2

def simulate(c1, c2):
    dt = 0.001
    c1.position = c1.position + c1.velocity * dt
    c2.position = c2.position + c2.velocity * dt
    return c1, c2

x_min, x_max, y_min, y_max = -100, 100, -100, 100

def bound(c):
    if (c.position[0] <= x_min):
        c.velocity[0] = -c.velocity[0]
    if (c.position[0] >= x_max):
        c.velocity[0] = -c.velocity[0]

    if (c.position[1] <= y_min):
        c.velocity[1] = -c.velocity[1]
    if (c.position[1] >= y_max):
        c.velocity[1] = -c.velocity[1]

for i in range(10):
        bound(c1)
        bound(c2)
        dp = c1.position - c2.position
        # print(dp, np.linalg.norm(dp))
        # есть колизия
        if (np.linalg.norm(dp) <= (c1.radius + c2.radius)):
            c1, c2 = collide(c1, c2)
        c1, c2 = simulate(c1, c2)

#%%
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
x_min, x_max, y_min, y_max = -100, 100, -100, 100
fig = plt.figure()
ax = plt.axes(xlim=(-100, 100), ylim=(-100, 100))

c_a1 = plt.Circle(c1.position, c1.radius, color=c1.color)
ax.add_artist(c_a1)
c_a2 = plt.Circle(c2.position, c2.radius, color=c2.color)
ax.add_artist(c_a2)
# %%
def animate(i):
    global c1, c2, c_a1, c_a2
    bound(c1)
    bound(c2)
    dp = c1.position - c2.position
    if (np.linalg.norm(dp) <= (c1.radius + c2.radius)):
        c1, c2 = collide(c1, c2)
    c1, c2 = simulate(c1, c2)
        
    ax.artists.remove(c_a1)
    ax.artists.remove(c_a2)
    c_a1 = plt.Circle(c1.position, c1.radius, color=c1.color)
    c_a2 = plt.Circle(c2.position, c2.radius, color=c2.color)
    ax.add_artist(c_a1)
    ax.add_artist(c_a2)

    return c_a1, c_a2


anim = FuncAnimation(fig, animate,
                               frames=100000, interval=1, blit=True)
plt.show()