#%%
import sympy as sp
# Проверка наших уравнений
x = sp.Function('x')
t,k,m,d = sp.symbols('t, k, m, d')
eq = sp.Eq(-k*x(t), m*x(t).diff(t,t))
sp.classify_ode(eq)

# решает уравнение относительно t
# с начальными условиями
sol = sp.dsolve(eq, ics={x(0): 10, sp.diff(x(t), t).subs(t,0): 5})
sol_t = sol.subs({k: 10, m:1, d:0.1})


A = sp.Matrix([[0,1],[-1, 0]])
V, Y = A.diagonalize()
V*Y*V.inv()
exp_A = (A*t).exp().applyfunc(sp.simplify)
#%%
sp.plot(sol_t.rhs, (t, 0, 50))

#%%
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from lecture_1.hw1.spring import spring

Ox = (0, 0)
x_t = (0, 0)

fig = plt.figure()
ax = plt.axes(xlim=(-10, 10), ylim=(-2, 2))
line, = ax.plot([], [], lw=1)
circle = plt.Circle((0, 0), 0.2, color='g')
ax.add_artist(circle)
def init():
    line.set_data([], [])
    return line,

def animate(i):
    line.set_data(*spring(Ox, (float(sol_t.subs(t,i/10).evalf().rhs), 0), 20, 0.5))
    global circle
    ax.artists.remove(circle)
    circle = plt.Circle((float(sol_t.subs(t,i/10).evalf().rhs), 0), 0.2, color='r')
    ax.add_artist(circle)

    return line, circle


anim = FuncAnimation(fig, animate, init_func=init,
                               frames=10000, interval=1, blit=True)
plt.show()