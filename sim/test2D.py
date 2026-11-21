import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

g = 9.81          # m/s²
thrust = 40.0     # N
burn_rate = 0.2   # kg/s

dry_mass = 1.0
fuel_mass = 2.0

x = 0.0
y = 0.0
vx = 0.0
vy = 0.0
angle = np.radians(85)

dt = 0.02

xs = []
ys = []

fig, ax = plt.subplots(figsize=(6, 8))
ax.set_xlim(-20, 20)
ax.set_ylim(0, 150)
ax.set_title("Simple Rocket Simulation")
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Altitude (m)")

trajectory, = ax.plot([], [], lw=1)
rocket, = ax.plot([], [], marker="^", markersize=12)

def update(frame):
    global x, y, vx, vy, fuel_mass

    mass = dry_mass + fuel_mass

    if fuel_mass > 0:
        ax_thrust = thrust * np.cos(angle) / mass
        ay_thrust = thrust * np.sin(angle) / mass

        fuel_mass -= burn_rate * dt
        fuel_mass = max(0, fuel_mass)
    else:
        ax_thrust = 0
        ay_thrust = 0

    ax_total = ax_thrust
    ay_total = ay_thrust - g

    vx += ax_total * dt
    vy += ay_total * dt

    x += vx * dt
    y += vy * dt

    if y < 0:
        y = 0
        vy = 0

    xs.append(x)
    ys.append(y)

    trajectory.set_data(xs, ys)
    rocket.set_data([x], [y])

    return trajectory, rocket

ani = FuncAnimation(
    fig,
    update,
    frames=1000,
    interval=20,
    blit=True
)

plt.show()