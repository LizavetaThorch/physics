import numpy as np

import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import matplotlib.animation

def get_rot_matrix(th):
    return np.array([[np.cos(th), -np.sin(th), 0.],[np.sin(th), np.cos(th), 0.],[0.,0.,1.]])

def get_global_coord(l_ps, cm, R):
    res = l_ps.copy()
    for i in range(len(l_ps)):
        res[i] = cm + R.dot(l_ps[i])
    return res

#get_global_coord(l_ps, cm, R)

def get_contact_mask(l_ps, cm, R):
    n_p = len(l_ps)
    gl_ps = get_global_coord(l_ps, cm, R)
    res = [0]*n_p
    for i in range(n_p):
        if gl_ps[i][1] <= 0:
            res[i] = 1
    return res
#get_contact_mask(l_ps, cm, R)
# %%      

lambdas = [0.]*n_c

def seq_impulse(v, mass, I, mask, cm, g_ps, normal, g, max_iter = 1):
    zero_accel = np.array([0., g, 0.,0.,0.,0.])
    n_c = len(l_ps)
    global lambdas
    for k in range(max_iter):
        for i in range(n_c):
            if (mask[i] == 0):
                continue
            old_lambda = lambdas[i]        
            J = np.hstack((normal, np.cross(g_ps[i] - cm, normal)))
            M_inv = np.diag([1./mass,1./mass,1./mass, 0. ,0. ,1./I])
            M_eff = J.dot(M_inv).dot(J.transpose())
            print(J.dot(v))
            lambdas[i] = lambdas[i] - 1/M_eff*(J.dot(v) + dt * g)
            lambdas[i] = np.clip(lambdas[i], 0., None)
            v = v + M_inv.dot(J.transpose())*(lambdas[i] - old_lambda) 
    return v
# %%
th = np.pi/6
R = get_rot_matrix(th)
cm = np.array([0., 10., 0.])
normal = np.array([0., 1., 0.]);
g = -1.1
accel = np.array([0., g, 0.,0.,0.,0.])
a = 1.
l_ps = [np.array([-a,-a, 0.]), np.array([-a,a, 0.]), np.array([a,a, 0.]), np.array([a,-a, 0.])]
v = np.array([0., 0., 0., 0., 0., 0.])
dt = 1/5.
mass = 0.1
I = 2/3 * mass * a * a

# %%
fig1 = plt.figure()
ar = get_global_coord(l_ps, cm, R)
ar.append(ar[0])
gl_quad = np.array(ar)
dx = fig1.add_subplot(111)

sc = dx.plot(gl_quad[:,0], gl_quad[:,1], color = 'b', marker='o')
dx.plot([-10,10], [0,0])
dx.set_xlim(-10, 10), dx.set_ylim(-10, 10)
dx.set_xlabel("x"),dx.set_ylabel("y") 
 
def update(i):
    global v, accel, l_ps, cm, R, th,sc
    v = v + accel * dt
    mask = get_contact_mask(l_ps, cm, R);
    
    ar = get_global_coord(l_ps, cm, R)
    v = seq_impulse(v, mass, I, mask, cm, ar, normal, g)
    cm = cm + v[:3]*dt
    print(i)
    th = th + v[5]*dt
    R = get_rot_matrix(th)
    
    ar = get_global_coord(l_ps, cm, R)
    ar.append(ar[0])
    gl_quad = np.array(ar)
    dx.lines.remove(sc[0])
    sc = dx.plot(gl_quad[:,0], gl_quad[:,1], color = 'b', marker='o')
    return sc

anim = matplotlib.animation.FuncAnimation(fig1, update, frames=100, interval=0)
plt.tight_layout()
plt.show()