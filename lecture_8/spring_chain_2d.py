import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations

name = '2d_spring_run1'
dt = 0.03
t = 10

# Тела
body_x = [
    np.array([0.0, 0.0]),
    np.array([3.0, 1.0]),
    np.array([6.0, 2.0]),
    np.array([9.0, 3.0])
]

body_v = [
    np.array([0.0, 0.0]),
    np.array([3.0, 1.0]) / 3,
    np.array([0.0, 0.0]),
    np.array([0.0, 0.0])
]

# Пружины
spring_k_m = [
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0]
]

spring_l = [
    [0.0, 3.1, 0.0, 0.0],
    [3.1, 0.0, 3.1, 0.0],
    [0.0, 3.1, 0.0, 3.1],
    [0.0, 0.0, 3.1, 0.0]
]

def dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def direction(x1, x2):
    a = x2 - x1
    a_len = dist(np.zeros(a.shape), a)
    return a / a_len

def spring(start, end, nodes, width):
    nodes = max(int(nodes), 1)
    spring_coords = np.zeros((2, nodes + 2))
    spring_coords[:,0], spring_coords[:,-1] = start, end

    length = dist(start, end)
    u_t = direction(start, end)
    u_n = np.array([u_t[0], -u_t[1]])
    normal_dist = np.sqrt(max(0, width**2 - (length**2 / nodes**2))) / 2
    for i in range(1, nodes + 1):
        spring_coords[:,i] = (
            start
            + ((length * (2 * i - 1) * u_t) / (2 * nodes))
            + (normal_dist * (-1)**i * u_n))

    return spring_coords[0,:], spring_coords[1,:]

def draw_body(ax, pos):
    r = 0.4
    circle = plt.Circle((pos[0], pos[1]), r, color='r', zorder=10)
    ax.add_patch(circle)

def draw_spring(ax, x, y):
    ax.plot(*spring(x, y, 6, 1), color="g")

def draw_all(i):
    fig, ax = plt.subplots()
    ax.set_xlim([-1, 11])
    ax.set_ylim([-1, 11])
    ax.set_aspect("equal")

    for body_i in range(len(body_x)):
        for body_j in range(len(body_x)):
            if spring_k_m[body_i][body_j] == 0:
                continue
            draw_spring(ax, body_x[body_i], body_x[body_j])

    for body_i in range(len(body_x)):
        draw_body(ax, body_x[body_i])

    plt.savefig(f'{name}/{i:03d}.png', dpi=300)
    plt.close()

# Симуляция
for time_i in range(int(t / dt)):
    # Обновляем ускорения
    body_a = [np.zeros(body_x[0].shape, float) for i in range(len(body_x))]
    for body_i in range(len(body_x)):
        for body_j in range(len(body_x)):
            if spring_k_m[body_i][body_j] == 0:
                continue
            body_dist = dist(body_x[body_i], body_x[body_j])
            if body_dist < 1e-4:
                continue
            delta = body_dist - spring_l[body_i][body_j]
            body_a[body_i] += spring_k_m[body_i][body_j] * delta * direction(body_x[body_i], body_x[body_j])

    # Обновляем скорости и координаты
    for body_i in range(len(body_x)):
        body_v[body_i] += body_a[body_i] * dt
        # Предполагаем равноускоренное движение, чтобы повысить точность
        body_x[body_i] += body_v[body_i] * dt - body_a[body_i] * dt * dt / 2

    draw_all(time_i)

# Сохраняем гифку
import glob
import contextlib
from PIL import Image

# filepaths
fp_in = f"{name}/*.png"
fp_out = f"{name}/{name}.gif"

# use exit stack to automatically close opened images
with contextlib.ExitStack() as stack:

    # lazily load images
    imgs = (stack.enter_context(Image.open(f))
            for f in sorted(glob.glob(fp_in)))

    # extract  first image from iterator
    img = next(imgs)

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=30, loop=0)
