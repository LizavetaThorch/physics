import numpy as np
import sympy as smp
from scipy.integrate import odeint
from sympy.solvers.solveset import linsolve, nonlinsolve
import matplotlib.pyplot as plt

t = smp.symbols('t')
m1, m2, g = smp.symbols('m1 m2 g')
L1, L2 = smp.symbols('L1, L2')

the1, the2, phi1, phi2 = smp.symbols(r'\theta_1, \theta_2 \phi_1 \phi_2', cls=smp.Function)
the1 = the1(t)
the2 = the2(t)
phi1 = phi1(t)
phi2 = phi2(t)

the1_d = smp.diff(the1, t)
the2_d = smp.diff(the2, t)
phi1_d = smp.diff(phi1, t)
phi2_d = smp.diff(phi2, t)
the1_dd = smp.diff(the1_d, t)
the2_dd = smp.diff(the2_d, t)
phi1_dd = smp.diff(phi1_d, t)
phi2_dd = smp.diff(phi2_d, t)

x1 = L1*smp.sin(the1)*smp.cos(phi1)
y1 = L1*smp.sin(the1)*smp.sin(phi1)
z1 = -L1*smp.cos(the1)
x2 = x1 + L2*smp.sin(the2)*smp.cos(phi2)
y2 = y1 + L2*smp.sin(the2)*smp.sin(phi2)
z2 = z1 -L2*smp.cos(the2)

x1_f = smp.lambdify((the1, the2, phi1, phi2, L1, L2), x1)
y1_f = smp.lambdify((the1, the2, phi1, phi2, L1, L2), y1)
z1_f = smp.lambdify((the1, the2, phi1, phi2, L1, L2), z1)
x2_f = smp.lambdify((the1, the2, phi1, phi2, L1, L2), x2)
y2_f = smp.lambdify((the1, the2, phi1, phi2, L1, L2), y2)
z2_f = smp.lambdify((the1, the2, phi1, phi2, L1, L2), z2)

# Kinetic
T1 = smp.Rational(1,2) * m1 * (smp.diff(x1, t)**2 + smp.diff(y1, t)**2 + smp.diff(z1, t)**2)
T2 = smp.Rational(1,2) * m2 * (smp.diff(x2, t)**2 + smp.diff(y2, t)**2 + smp.diff(z2, t)**2)
T = T1+T2
# Potential
V1 = m1*g*z1
V2 = m2*g*z2
V = V1 + V2
# Lagrangian
L = T-V

LE1 = smp.diff(L, the1) - smp.diff(smp.diff(L, the1_d), t).simplify()
LE2 = smp.diff(L, the2) - smp.diff(smp.diff(L, the2_d), t).simplify()
LE3 = smp.diff(L, phi1) - smp.diff(smp.diff(L, phi1_d), t).simplify()
LE4 = smp.diff(L, phi2) - smp.diff(smp.diff(L, phi2_d), t).simplify()

sols = smp.solve([LE1, LE2, LE3, LE4], (the1_dd, the2_dd, phi1_dd, phi2_dd),
                simplify=False, rational=False)

do1dt_f = smp.lambdify((t,g,m1,m2,L1,L2,the1,the2,the1_d,the2_d,phi1,phi2,phi1_d,phi2_d), sols[the1_dd])
do2dt_f = smp.lambdify((t,g,m1,m2,L1,L2,the1,the2,the1_d,the2_d,phi1,phi2,phi1_d,phi2_d), sols[the2_dd])
dthe1dt_f = smp.lambdify(the1_d, the1_d)
dthe2dt_f = smp.lambdify(the2_d, the2_d)

dw1dt_f = smp.lambdify((t,g,m1,m2,L1,L2,the1,the2,the1_d,the2_d,phi1,phi2,phi1_d,phi2_d), sols[phi1_dd])
dw2dt_f = smp.lambdify((t,g,m1,m2,L1,L2,the1,the2,the1_d,the2_d,phi1,phi2,phi1_d,phi2_d), sols[phi2_dd])
dphi1dt_f = smp.lambdify(phi1_d, phi1_d)
dphi2dt_f = smp.lambdify(phi2_d, phi2_d)

def dSdt(S, t, g, m1, m2, L1, L2):
    the1, the2, phi1, phi2, o1, o2, w1, w2 = S
    return [
        dthe1dt_f(o1),
        dthe2dt_f(o2),
        dphi1dt_f(w1),
        dphi2dt_f(w2),
        do1dt_f(t, g, m1, m2, L1, L2, the1, the2, o1, o2, phi1, phi2, w1, w2),
        do2dt_f(t, g, m1, m2, L1, L2, the1, the2, o1, o2, phi1, phi2, w1, w2),
        dw1dt_f(t, g, m1, m2, L1, L2, the1, the2, o1, o2, phi1, phi2, w1, w2),
        dw2dt_f(t, g, m1, m2, L1, L2, the1, the2, o1, o2, phi1, phi2, w1, w2),
    ]

t = np.linspace(0, 40, 1001)
g = 9.81
m1=2
m2=1
L1 = 2
L2 = 1
ans = odeint(dSdt, y0=[np.pi/3, np.pi/3, 0, -np.pi, 3, 10, -2, -10], t=t, args=(g,m1,m2,L1,L2))

the1 = ans.T[0]
the2 = ans.T[1]
phi1 = ans.T[2]

def get_pos(the1, the2, phi1, phi2, L1, L2):
    return (x1_f(the1, the2, phi1, phi2, L1, L2),
            y1_f(the1, the2, phi1, phi2, L1, L2),
            z1_f(the1, the2, phi1, phi2, L1, L2),
            x2_f(the1, the2, phi1, phi2, L1, L2),
            y2_f(the1, the2, phi1, phi2, L1, L2),
            z2_f(the1, the2, phi1, phi2, L1, L2))

x1, y1, z1, x2, y2, z2 = get_pos(ans.T[0], ans.T[1], ans.T[2], ans.T[3], L1, L2)
np.save('../data/3Dpen', np.array([x1,y1,z1,x2,y2,z2]))



################################################################################################# D R A W ###########################################################################################################################
import vpython
from vpython import *
import numpy as np

x1, y1, z1, x2, y2, z2 = np.load('..\\data\\3Dpen.npy')
ball1 = vpython.sphere(color = color.green, radius = 0.3, make_trail=True, retain=20)
ball2 = vpython.sphere(color = color.blue, radius = 0.3, make_trail=True, retain=20)
rod1 = cylinder(pos=vector(0,0,0),axis=vector(0,0,0), radius=0.05)
rod2 = cylinder(pos=vector(0,0,0),axis=vector(0,0,0), radius=0.05)
base  = box(pos=vector(0,-4.25,0),axis=vector(1,0,0),
            size=vector(10,0.5,10) )
s1 = cylinder(pos=vector(0,-3.99,0),axis=vector(0,-0.1,0), radius=0.8, color=color.gray(luminance=0.7))
s2 = cylinder(pos=vector(0,-3.99,0),axis=vector(0,-0.1,0), radius=0.8, color=color.gray(luminance=0.7))

print('Start')
i = 0
while True:
    rate(30)
    i = i + 1
    i = i % len(x1)
    ball1.pos = vector(x1[i], z1[i], y1[i])
    ball2.pos = vector(x2[i], z2[i], y2[i])
    rod1.axis = vector(x1[i], z1[i], y1[i])
    rod2.pos = vector(x1[i], z1[i], y1[i])
    rod2.axis = vector(x2[i]-x1[i], z2[i]-z1[i], y2[i]-y1[i])
    s1.pos = vector(x1[i], -3.99, y1[i])
    s2.pos = vector(x2[i], -3.99, y2[i])