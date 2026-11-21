"""3D rocket sim. Run with --help to see options."""

import argparse
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 9.81
RHO0 = 1.225        # kg/m^3
SCALE_HEIGHT = 8500  # m


@dataclass
class Rocket:
    dry_mass: float = 1.5        # kg
    fuel_mass: float = 0.6       # kg
    thrust: float = 80.0         # N
    burn_time: float = 4.0       # s
    diameter: float = 0.08       # m
    length: float = 1.2          # m
    cd: float = 0.5
    chute_cd_area: float = 0.6   # m^2
    chute_delay: float = 1.0     # s

    @property
    def area(self):
        return np.pi * (self.diameter / 2) ** 2

    @property
    def burn_rate(self):
        return self.fuel_mass / self.burn_time


@dataclass
class Launch:
    elevation_deg: float = 85.0  # deg
    azimuth_deg: float = 45.0    # deg, 0 = north
    rail_length: float = 2.0     # m
    wind: tuple = (4.0, 0.0, 0.0)  # m/s

    @property
    def direction(self):
        """Unit vector the rail points along (x east, y north, z up)."""
        el = np.radians(self.elevation_deg)
        az = np.radians(self.azimuth_deg)
        return np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])


def air_density(z):
    """Rough exponential atmosphere, good enough below a few km."""
    return RHO0 * np.exp(-max(z, 0.0) / SCALE_HEIGHT)


def simulate(rocket: Rocket, launch: Launch, dt=0.01, t_max=600.0):
    """Fly the rocket until it hits the ground.

    State is [x, y, z, vx, vy, vz, fuel]. Returns a dict of arrays
    (t, pos, vel, axis, ...) with one row per time step.
    """
    wind = np.array(launch.wind, dtype=float)
    rail_dir = launch.direction

    s = np.array([0, 0, 0, 0, 0, 0, rocket.fuel_mass], dtype=float)
    t = 0.0
    apogee_t = None
    chute_open = False

    def on_rail(state):
        return np.linalg.norm(state[:3]) < launch.rail_length

    def body_axis(state):
        """Which way the nose is pointing."""
        if on_rail(state):
            return rail_dir
        # off rail it weathercocks into airflow
        v_rel = state[3:6] - wind
        speed = np.linalg.norm(v_rel)
        return v_rel / speed if speed > 1e-3 else rail_dir

    def deriv(state, chute):
        pos, vel, fuel = state[:3], state[3:6], state[6]
        mass = rocket.dry_mass + max(fuel, 0.0)
        burning = fuel > 0

        axis = body_axis(state)
        f = np.zeros(3)

        if burning:
            f += rocket.thrust * axis

        f[2] -= mass * G

        v_rel = vel - wind
        speed = np.linalg.norm(v_rel)
        if speed > 0:
            cda = rocket.chute_cd_area if chute else rocket.cd * rocket.area
            f -= 0.5 * air_density(pos[2]) * cda * speed * v_rel

        acc = f / mass
        dfuel = -rocket.burn_rate if burning else 0.0

        if on_rail(state):  # can only slide up rail not down thru pad
            a_along = max(np.dot(acc, rail_dir), 0.0) if np.dot(vel, rail_dir) <= 0 else np.dot(acc, rail_dir)
            acc = a_along * rail_dir

        return np.concatenate([vel, acc, [dfuel]])

    log = {k: [] for k in ("t", "pos", "vel", "axis", "fuel", "burning", "chute")}

    def record():
        log["t"].append(t)
        log["pos"].append(s[:3].copy())
        log["vel"].append(s[3:6].copy())
        log["axis"].append(body_axis(s))
        log["fuel"].append(max(s[6], 0.0))
        log["burning"].append(s[6] > 0)
        log["chute"].append(chute_open)

    record()
    while t < t_max:
        k1 = deriv(s, chute_open)
        k2 = deriv(s + 0.5 * dt * k1, chute_open)
        k3 = deriv(s + 0.5 * dt * k2, chute_open)
        k4 = deriv(s + dt * k3, chute_open)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        s[6] = max(s[6], 0.0)
        t += dt

        if apogee_t is None and t > 0.5 and s[5] < 0:
            apogee_t = t
        if apogee_t is not None and not chute_open and t >= apogee_t + rocket.chute_delay:
            chute_open = True

        # never left the pad
        if t > 1.0 and s[2] <= 0 and np.linalg.norm(s[3:6]) < 1e-6:
            record()
            break

        if s[2] < 0 and t > 0.1:
            s[2] = 0.0
            record()
            break
        record()

    out = {k: np.array(v) for k, v in log.items()}
    out["apogee_t"] = apogee_t
    return out


def summarize(flight):
    """Print apogee, max speed, landing spot, etc."""
    pos, vel, t = flight["pos"], flight["vel"], flight["t"]
    speed = np.linalg.norm(vel, axis=1)
    i_apo = int(np.argmax(pos[:, 2]))
    burnout = np.where(~flight["burning"])[0]
    i_bo = int(burnout[0]) if len(burnout) else len(t) - 1
    drift = np.linalg.norm(pos[-1, :2])
    print("=" * 44)
    print(" FLIGHT SUMMARY")
    print("=" * 44)
    print(f" Burnout:        t={t[i_bo]:6.2f}s  alt={pos[i_bo, 2]:8.1f} m")
    print(f" Max speed:      {speed.max():8.1f} m/s  (Mach {speed.max() / 343:.2f})")
    print(f" Apogee:         {pos[i_apo, 2]:8.1f} m  at t={t[i_apo]:.2f}s")
    print(f" Landing speed:  {speed[-1]:8.1f} m/s")
    print(f" Flight time:    {t[-1]:8.1f} s")
    print(f" Landing point:  ({pos[-1, 0]:.1f}, {pos[-1, 1]:.1f}) m  — {drift:.1f} m from pad")
    print("=" * 44)


def animate(flight, rocket: Rocket, launch: Launch, speedup=1.0, save=None):
    """Play back a finished flight. Pass save="out.gif" to write a file instead."""
    t =flight["t"]
    pos = flight["pos"]
    vel = flight["vel"]
    axes_dir = flight["axis"]
    dt = t[1] - t[0]

    fps = 30
    step = max(1, int(round(speedup / (fps * dt))))
    frames = list(range(0, len(t), step))
    if frames[-1] != len(t) - 1:
        frames.append(len(t) - 1)

    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor("#0b1020")
    ax = fig.add_axes([0.0, 0.0, 0.68, 1.0], projection="3d")
    ax.set_facecolor("#0b1020")

    ax_alt = fig.add_axes([0.73, 0.56, 0.24, 0.36])
    ax_spd = fig.add_axes([0.73, 0.10, 0.24, 0.36])
    for a in (ax_alt, ax_spd):
        a.set_facecolor("#121933")
        a.tick_params(colors="#c8d0e8", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#3a4570")
        a.grid(color="#2a3360", lw=0.5)
    speed = np.linalg.norm(vel, axis=1)
    ax_alt.plot(t, pos[:, 2], color="#3a4570", lw=1)
    ax_spd.plot(t, speed, color="#3a4570", lw=1)
    alt_line, = ax_alt.plot([], [], color="#5ad1ff", lw=1.8)
    spd_line, = ax_spd.plot([], [], color="#ffb04a", lw=1.8)
    ax_alt.set_title("Altitude (m)", color="#e6ebff", fontsize=10)
    ax_spd.set_title("Speed (m/s)", color="#e6ebff", fontsize=10)
    ax_spd.set_xlabel("time (s)", color="#c8d0e8", fontsize=9)

    pad = 10
    lo = pos.min(axis=0)
    hi = pos.max(axis=0)
    span = max(hi[0] - lo[0], hi[1] - lo[1], hi[2] * 0.6, 40) / 2 + pad
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_zlim(0, hi[2] * 1.1 + pad)
    ax.set_box_aspect((1, 1, 1.2))
    for lab, fn in (("East x (m)", ax.set_xlabel), ("North y (m)", ax.set_ylabel), ("Altitude (m)", ax.set_zlabel)):
        fn(lab, color="#c8d0e8")
    ax.tick_params(colors="#8f9ac0", labelsize=8)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((0.07, 0.09, 0.18, 1.0))
        pane.set_edgecolor("#2a3360")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update(color="#26305a", linewidth=0.5)

    gx = np.linspace(cx - span, cx + span, 11)
    gy = np.linspace(cy - span, cy + span, 11)
    GX, GY = np.meshgrid(gx, gy)
    ax.plot_surface(GX, GY, np.zeros_like(GX), color="#1b4d2e", alpha=0.35, shade=False)
    ax.plot_wireframe(GX, GY, np.zeros_like(GX), color="#2f7a4a", lw=0.4)

    ax.scatter([0], [0], [0], color="#ff5a5a", s=40, marker="s")
    w = np.array(launch.wind, dtype=float)
    if np.linalg.norm(w) > 0:
        wd = w / np.linalg.norm(w) * span * 0.25
        ax.quiver(cx - span * 0.8, cy - span * 0.8, 0, wd[0], wd[1], 0, color="#9fd8ff", lw=2, arrow_length_ratio=0.3)
        ax.text(cx - span * 0.8, cy - span * 0.8, 2, f"wind {np.linalg.norm(w):.1f} m/s", color="#9fd8ff", fontsize=8)

    i_apo = int(np.argmax(pos[:, 2]))
    ax.scatter(*pos[-1], color="#ffd84a", s=30, marker="x")

    trail, = ax.plot([], [], [], color="#5ad1ff", lw=1.5)
    shadow, = ax.plot([], [], [], color="#000000", lw=1, alpha=0.5)
    drop, = ax.plot([], [], [], color="#5ad1ff", lw=0.6, ls=":", alpha=0.6)
    body, = ax.plot([], [], [], color="#f2f4ff", lw=5, solid_capstyle="round")
    nose, = ax.plot([], [], [], color="#ff5a5a", lw=5, solid_capstyle="round")
    flame, = ax.plot([], [], [], color="#ffb04a", lw=4, alpha=0.9)
    chute, = ax.plot([], [], [], color="#ff7ad9", lw=1.2)
    apo_marker = ax.scatter([], [], [], color="#ffd84a", s=35, marker="^")

    hud = fig.text(0.015, 0.97, "", color="#e6ebff", family="monospace", fontsize=10, va="top")
    fig.text(0.015, 0.02, "space: pause   r: restart   drag: rotate", color="#6f7aa0", fontsize=8)

    vis_len = span * 0.08  # bigger than real rocket otherwise invisible

    state = {"paused": False, "k": 0}

    def draw(idx):
        p = pos[idx]
        a = axes_dir[idx]
        burning = flight["burning"][idx]
        chute_on = flight["chute"][idx]

        trail.set_data_3d(pos[: idx + 1, 0], pos[: idx + 1, 1], pos[: idx + 1, 2])
        shadow.set_data_3d(pos[: idx + 1, 0], pos[: idx + 1, 1], np.zeros(idx + 1))
        drop.set_data_3d([p[0], p[0]], [p[1], p[1]], [0, p[2]])

        a_draw = np.array([0, 0, 1.0]) if chute_on else a
        tail = p - a_draw * vis_len * 0.5
        tip = p + a_draw * vis_len * 0.5
        nose_base = p + a_draw * vis_len * 0.3
        body.set_data_3d([tail[0], nose_base[0]], [tail[1], nose_base[1]], [tail[2], nose_base[2]])
        nose.set_data_3d([nose_base[0], tip[0]], [nose_base[1], tip[1]], [nose_base[2], tip[2]])

        if burning:
            flicker = 0.6 + 0.4 * np.random.rand()
            fl_end = tail - a_draw * vis_len * 0.6 * flicker
            flame.set_data_3d([tail[0], fl_end[0]], [tail[1], fl_end[1]], [tail[2], fl_end[2]])
        else:
            flame.set_data_3d([], [], [])

        if chute_on:
            r = vis_len * 0.6
            top = tip + np.array([0, 0, vis_len * 0.9])
            ang = np.linspace(0, 2 * np.pi, 9)
            xs, ys, zs = [], [], []
            for th in ang:
                rim = top + np.array([r * np.cos(th), r * np.sin(th), -vis_len * 0.3])
                xs += [tip[0], rim[0], top[0], np.nan]
                ys += [tip[1], rim[1], top[1], np.nan]
                zs += [tip[2], rim[2], top[2], np.nan]
            chute.set_data_3d(xs, ys, zs)
        else:
            chute.set_data_3d([], [], [])

        if idx >= i_apo:
            apo_marker._offsets3d = ([pos[i_apo, 0]], [pos[i_apo, 1]], [pos[i_apo, 2]])

        alt_line.set_data(t[: idx + 1], pos[: idx + 1, 2])
        spd_line.set_data(t[: idx + 1], speed[: idx + 1])

        if flight["chute"][idx]:
            phase = "DESCENT (chute)"
        elif burning:
            phase = "POWERED ASCENT"
        elif idx < i_apo:
            phase = "COASTING"
        elif idx == len(t) - 1:
            phase = "LANDED"
        else:
            phase = "FALLING"
        mass = rocket.dry_mass + flight["fuel"][idx]
        hud.set_text(
            f"T+ {t[idx]:6.2f} s   {phase}\n"
            f"Altitude  {p[2]:8.1f} m\n"
            f"Speed     {speed[idx]:8.1f} m/s\n"
            f"Vert vel  {vel[idx, 2]:8.1f} m/s\n"
            f"Downrange {np.linalg.norm(p[:2]):8.1f} m\n"
            f"Mass      {mass:8.2f} kg\n"
            f"Fuel      {flight['fuel'][idx] / rocket.fuel_mass * 100:7.0f} %\n"
            f"Max alt   {pos[: idx + 1, 2].max():8.1f} m"
        )

    def update(_):
        if not state["paused"]:
            draw(frames[state["k"]])
            if state["k"] < len(frames) - 1:
                state["k"] += 1
        return []

    def on_key(event):
        if event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key == "r":
            state["k"] = 0
            apo_marker._offsets3d = ([], [], [])

    fig.canvas.mpl_connect("key_press_event", on_key)
    ax.view_init(elev=18, azim=-60)

    if save:
        state["k"] = 0
        anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False, repeat=False)
        anim.save(save, fps=fps)
        print(f"saved {save}")
    else:
        anim = FuncAnimation(fig, update, interval=1000 / fps, blit=False, cache_frame_data=False)
        plt.show()
    return anim


def main():
    p = argparse.ArgumentParser(description="3D rocket flight simulator")
    p.add_argument("--thrust", type=float, default=80.0, help="engine thrust, N")
    p.add_argument("--burn-time", type=float, default=4.0, help="motor burn time, s")
    p.add_argument("--dry-mass", type=float, default=1.5, help="kg")
    p.add_argument("--fuel-mass", type=float, default=0.6, help="kg")
    p.add_argument("--cd", type=float, default=0.5, help="drag coefficient")
    p.add_argument("--diameter", type=float, default=0.08, help="body diameter, m")
    p.add_argument("--angle", type=float, default=85.0, help="launch elevation, degrees above horizontal")
    p.add_argument("--heading", type=float, default=45.0, help="launch azimuth, degrees (0=north, 90=east)")
    p.add_argument("--wind", type=float, default=4.0, help="wind speed, m/s")
    p.add_argument("--wind-dir", type=float, default=90.0, help="direction wind blows TOWARD, degrees (0=north)")
    p.add_argument("--speed", type=float, default=2.0, help="playback speed multiplier")
    p.add_argument("--save", type=str, default=None, help="save animation to file (.gif or .mp4)")
    p.add_argument("--no-anim", action="store_true", help="only print summary")
    args = p.parse_args()

    wd = np.radians(args.wind_dir)
    rocket = Rocket(dry_mass=args.dry_mass, fuel_mass=args.fuel_mass, thrust=args.thrust,
                    burn_time=args.burn_time, cd=args.cd, diameter=args.diameter)
    launch = Launch(elevation_deg=args.angle, azimuth_deg=args.heading,
                    wind=(args.wind * np.sin(wd), args.wind * np.cos(wd), 0.0))

    twr = rocket.thrust / ((rocket.dry_mass + rocket.fuel_mass) * G)
    if twr <= 1.0:
        print(f"Warning: thrust-to-weight ratio is {twr:.2f} — the rocket can't lift off.")

    flight = simulate(rocket, launch)
    summarize(flight)
    if not args.no_anim:
        animate(flight, rocket, launch, speedup=args.speed, save=args.save)


if __name__ == "__main__":
    main()
