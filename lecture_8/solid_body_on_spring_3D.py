import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Определить физические параметры
m = 1 # Масса (кг)
k = 100 # Жесткость пружины (Н/м)
d = 0.1 # Смещение крепления пружины от центра масс (м)
g = 9.81 # Ускорение свободного падения (м/с^2)

# Определить начальные условия
x0 = 0 # Начальное положение по оси x (м)
y0 = 0 # Начальное положение по оси y (м)
z0 = 0 # Начальное положение по оси z (м)
v0x = 0 # Начальная скорость по оси x (м/с)
v0y = 0 # Начальная скорость по оси y (м/с)
v0z = 0 # Начальная скорость по оси z (м/с)

# Определить силы, действующие на твердое тело
def forces(state):
    x, y, z, v_x, v_y, v_z = state
    F_spring = -k * (x - d, y, z)
    F_gravity = 0, -m * g, 0
    return F_spring, F_gravity

# Шаг Рунге-Кутты 4-го порядка
def rk4_step(state, dt, forces):
    x, y, z, v_x, v_y, v_z = state
    F1_spring, F1_gravity = forces(state)
    k1_x = dt * v_x
    k1_y = dt * v_y
    k1_z = dt * v_z
    k1_v_x = dt * F1_spring[0] / m
    k1_v_y = dt * (F1_spring[1] + F1_gravity[1]) / m
    k1_v_z = dt * F1_spring[2] / m

    F2_spring, F2_gravity = forces((x + 0.5 * k1_x, y + 0.5 * k1_y, z + 0.5 * k1_z,
                                      v_x + 0.5 * k1_v_x, v_y + 0.5 * k1_v_y, v_z + 0.5 * k1_v_z))
    k2_x = dt * (v_x + 0.5 * k1_v_x)
    k2_y = dt * (v_y + 0.5 * k1_v_y)
    k2_z = dt * (v_z + 0.5 * k1_v_z)
    k2_v_x = dt * F2_spring[0] / m
    k2_v_y = dt * (F2_spring[1] + F2_gravity[1]) / m
    k2_v_z = dt * F2_spring[2] / m

    F3_spring, F3_gravity = forces((x + 0.5 * k2_x, y + 0.5 * k2_y, z + 0.5 * k2_z,
                                      v_x + 0.5 * k2_v_x, v_y + 0.5 * k2_v_y, v_z + 0.5 * k2_v_z))
    k3_x = dt * (v_x + 0.5 * k2_v_x)
    k3_y = dt * (v_y + 0.5 * k2_v_y)
    k3_z = dt * (v_z + 0.5 * k2_v_z)
    k3_v_x = dt * F3_spring[0] / m
    k3_v_y = dt * (F3_spring[1] + F3_gravity[1]) / m
    k3_v_z = dt * F3_spring[2] / m

    F4_spring, F4_gravity = forces((x + k3_x, y + k3_y, z + k3_z,
                                      v_x + k3_v_x, v_y + k3_v_y, v_z + k3_v_z))
    k4_x = dt * (v_x + k3_v_x)
    k4_y = dt * (v_y + k3_v_y)
    k4_z = dt * (v_z + k3_v_z)
    k4_v_x = dt * F4_spring[0] / m
    k4_v_y = dt * (F4_spring[1] + F4_gravity[1]) / m
    k4_v_z = dt * F4_spring[2] / m

    return np.array([x + 1 / 6 * (k1_x + 2 * k2_x + 2 * k3_x + k4_x),
                      y + 1 / 6 * (k1_y + 2 * k2_y + 2 * k3_y + k4_y),
                      z + 1 / 6 * (k1_z + 2 * k2_z + 2 * k3_z + k4_z),
                      v_x + 1 / 6 * (k1_v_x + 2 * k2_v_x + 2 * k3_v_x + k4_v_x),
                      v_y + 1 / 6 * (k1_v_y + 2 * k2_v_y + 2 * k3_v_y + k4_v_y),
                      v_z + 1 / 6 * (k1_v_z + 2 * k2_v_z + 2 * k3_v_z + k4_v_z)])

# Функция для обновления симуляции и отрисовки кадра
def update_sim(i, xyzs, line):
    global x0, y0, z0
    
    # Интегрировать уравнения движения методом РК4
    state = rk4_step((x0, y0, z0, 0, 0, 0), dt, forces)
    x0, y0, z0, _, _, _ = state

    # Обновить координаты точек
    xyzs[:, 0] = x0
    xyzs[:, 1] = y0
    xyzs[:, 2] = z0

    # Обновить линию
    line.set_data(xyzs[:, 0], xyzs[:, 2])
    line.set_3d_properties(xyzs[:, 1])

    return line,

# Определить время и шаг интегрирования
t_end = 10 # Конечное время симуляции (c)
dt = 0.01 # Шаг интегрирования (c)

# Составить массив координат точек твердого тела
num_points = 100
xyzs = np.zeros((num_points, 3))
xyzs[:, 0] = np.linspace(-0.5, 0.5, num_points)
xyzs[:, 1] = 0.05 * np.sin(2 * np.pi * xyzs[:, 0])

# Создать фигуру и ось для отображения симуляции
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-0.6, 0.6])
ax.set_ylim([-0.1, 0.6])
ax.set_zlim([-0.1, 0.6])

# Создать линию для отображения твердого тела
line, = ax.plot(xyzs[:, 0], xyzs[:, 2], xyzs[:, 1], 'o-', markersize=2, linewidth=2)

# Создать анимацию
ani = animation.FuncAnimation(fig, update_sim, frames=int(t_end / dt), fargs=(xyzs, line), interval=1)

# Показать анимацию
plt.show()
