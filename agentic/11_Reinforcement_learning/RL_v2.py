"""
Tabular Q-learning for 2D circular -> circular orbit transfer (numpy only, no neural nets).

Same normalised units as RL.py: mu = 1, start on a circular orbit of radius 1
(circular speed 1, period 2*pi). The agent learns ONE fixed target radius.

State  : (r, v_radial, v_tangential) discretised into a grid. This is rotation-invariant
         and fully describes the situation for a circular-to-circular problem.
Actions: 9 discrete thrusts in the local (radial, tangential) frame:
         coast, and +/- radial, +/- tangential at two magnitudes (20% and 100% of A_MAX).
Fuel   : total delta-v = sum(|thrust accel| * dt), the quantity Hohmann minimises.

Usage:
    python RL_v2.py hohmann [--target 1.5]
    python RL_v2.py train   [--target 1.5] [--episodes 10000] [--eval-every 250] [--out q_orbit.npz]
    python RL_v2.py plot    [--out q_orbit.npz]
"""

import argparse
import math

import numpy as np

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
MU = 1.0
R0 = 1.0
A_MAX = 0.5  # max thrust acceleration
DT = 0.1  # decision / integration step
MAX_STEPS = 200
FUEL_W = 10.0  # reward = -FUEL_W * dv + shaping + bonus/penalty
SHAPE_W = 10.0
BONUS = 5.0
FAIL = 5.0
R_CRASH, R_ESCAPE = 0.3, 4.0
TIME_W = 0.01  # small per-step cost so "coast forever" is not a free local optimum
ERR_CAP = 3.0  # cap on the shaping error, so crashes/escapes don't dominate every Q-value
TOL_MIN = 0.03  # success: |a - a_t|/a_t < tol and e < tol; tol is raised if the grid is too coarse
GAMMA = 0.995

LEVELS = (0.2, 1.0)
ACTIONS = [(0.0, 0.0)]
for _l in LEVELS:
    ACTIONS += [(_l, 0.0), (-_l, 0.0), (0.0, _l), (0.0, -_l)]  # (radial, tangential) fractions
NA = len(ACTIONS)

NR, NVR, NVT = 60, 17, 60
VR_RANGE = (-0.4, 0.4)
NS = NR * NVR * NVT  # radius / tangential-speed ranges are set per target in OrbitEnv


# ----------------------------------------------------------------------------
# Physics (pure Python floats: much faster than numpy for 4-element states)
# ----------------------------------------------------------------------------
def _f(s, ax, ay):
    x, y, vx, vy = s
    g = -MU / (x * x + y * y) ** 1.5
    return (vx, vy, g * x + ax, g * y + ay)


def _add(s, k, h):
    return (s[0] + h * k[0], s[1] + h * k[1], s[2] + h * k[2], s[3] + h * k[3])


def rk4(s, ax, ay, dt):
    k1 = _f(s, ax, ay)
    k2 = _f(_add(s, k1, 0.5 * dt), ax, ay)
    k3 = _f(_add(s, k2, 0.5 * dt), ax, ay)
    k4 = _f(_add(s, k3, dt), ax, ay)
    return tuple(
        s[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(4)
    )


def elements(s):
    """(a, e) or None if the orbit is unbound."""
    x, y, vx, vy = s
    r = math.hypot(x, y)
    v2 = vx * vx + vy * vy
    inv_a = 2.0 / r - v2 / MU
    if inv_a <= 1e-12:
        return None
    k = v2 - MU / r
    rv = x * vx + y * vy
    ex, ey = (k * x - rv * vx) / MU, (k * y - rv * vy) / MU
    return 1.0 / inv_a, math.hypot(ex, ey)


# ----------------------------------------------------------------------------
# Hohmann baseline
# ----------------------------------------------------------------------------
def hohmann_dv(r1, r2):
    a_t = 0.5 * (r1 + r2)
    v_c1, v_c2 = math.sqrt(MU / r1), math.sqrt(MU / r2)
    v_t1 = math.sqrt(MU * (2.0 / r1 - 1.0 / a_t))
    v_t2 = math.sqrt(MU * (2.0 / r2 - 1.0 / a_t))
    dv1, dv2 = v_t1 - v_c1, v_c2 - v_t2
    return dv1, dv2, abs(dv1) + abs(dv2), math.pi * math.sqrt(a_t**3 / MU)


def hohmann_trajectory(r1, r2, dt=0.01):
    dv1, dv2, _, t_tr = hohmann_dv(r1, r2)
    s = (r1, 0.0, 0.0, math.sqrt(MU / r1) + dv1)
    pts = [(s[0], s[1])]
    n = int(round(t_tr / dt))
    for _ in range(n):
        s = rk4(s, 0.0, 0.0, t_tr / n)
        pts.append((s[0], s[1]))
    v = math.hypot(s[2], s[3])
    s = (s[0], s[1], s[2] + dv2 * s[2] / v, s[3] + dv2 * s[3] / v)
    for _ in range(int(2 * math.pi * math.sqrt(r2**3 / MU) / dt)):
        s = rk4(s, 0.0, 0.0, dt)
        pts.append((s[0], s[1]))
    return np.array(pts)


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
class OrbitEnv:
    def __init__(self, r_target):
        self.rt = float(r_target)
        self.r_lo = min(R0, self.rt) - 0.1 * self.rt
        self.r_hi = max(R0, self.rt) + 0.1 * self.rt
        # Tangential-speed range hugging the speeds a Hohmann transfer actually visits
        dv1, dv2, _, _ = hohmann_dv(R0, self.rt)
        v_c1, v_c2 = math.sqrt(MU / R0), math.sqrt(MU / self.rt)
        speeds = (v_c1, v_c2, v_c1 + dv1, v_c2 - dv2)
        self.vt_lo, self.vt_hi = 0.9 * min(speeds), 1.1 * max(speeds)
        # One speed bin changes a by roughly 2*a*v*dv (relative); keep tol above that
        bin_w = (self.vt_hi - self.vt_lo) / NVT
        res = 2.0 * self.rt * v_c2 * bin_w
        self.tol = max(TOL_MIN, 1.2 * res)

    def _shape_err(self, s):
        """Potential for shaping: (|r_apo - rt| + |r_peri - rt|)/rt. It is 0 only on the
        target circle and decreases monotonically along a Hohmann transfer."""
        el = elements(s)
        if el is None:
            return ERR_CAP
        a, e = el
        err = (abs(a * (1 + e) - self.rt) + abs(a * (1 - e) - self.rt)) / self.rt
        return min(err, ERR_CAP)

    def _frame(self):
        x, y, vx, vy = self.s
        r = math.hypot(x, y)
        rx, ry = x / r, y / r
        return r, rx, ry, vx * rx + vy * ry, -vx * ry + vy * rx  # r, r_hat, v_r, v_t

    def state(self):
        r, _, _, vr, vt = self._frame()
        ir = min(NR - 1, max(0, int((r - self.r_lo) / (self.r_hi - self.r_lo) * NR)))
        ivr = min(NVR - 1, max(0, int((vr - VR_RANGE[0]) / (VR_RANGE[1] - VR_RANGE[0]) * NVR)))
        ivt = min(NVT - 1, max(0, int((vt - self.vt_lo) / (self.vt_hi - self.vt_lo) * NVT)))
        return (ir * NVR + ivr) * NVT + ivt

    def reset(self):
        self.s = (R0, 0.0, 0.0, math.sqrt(MU / R0))
        self.steps = 0
        self.dv = 0.0
        self.traj = [(self.s[0], self.s[1])]
        return self.state()

    def step(self, ai):
        fr, ft = ACTIONS[ai]
        _, rx, ry, _, _ = self._frame()
        ar, at = A_MAX * fr, A_MAX * ft
        ax, ay = ar * rx - at * ry, ar * ry + at * rx  # r_hat=(rx,ry), t_hat=(-ry,rx)

        phi_old = -self._shape_err(self.s)
        self.s = rk4(self.s, ax, ay, DT)
        self.steps += 1
        dv = math.hypot(ar, at) * DT
        self.dv += dv
        self.traj.append((self.s[0], self.s[1]))
        phi_new = -self._shape_err(self.s)

        reward = -FUEL_W * dv - TIME_W + SHAPE_W * (GAMMA * phi_new - phi_old)

        r = math.hypot(self.s[0], self.s[1])
        el = elements(self.s)
        success = el is not None and abs(el[0] - self.rt) / self.rt < self.tol and el[1] < self.tol
        failed = r < R_CRASH or r > R_ESCAPE or el is None
        if success:
            reward += BONUS
        elif failed:
            reward -= FAIL
        terminated = success or failed
        truncated = not terminated and self.steps >= MAX_STEPS
        return self.state(), reward, terminated, truncated, success


# ----------------------------------------------------------------------------
# Q-learning
# ----------------------------------------------------------------------------
def greedy_episode(env, Q):
    s = env.reset()
    total, success = 0.0, False
    while True:
        s, r, term, trunc, success = env.step(int(Q[s].argmax()))
        total += r
        if term or trunc:
            return total, success, env.dv


def train(target, episodes, eval_every, alpha, out, eps_end=0.05, seed=0):
    rng = np.random.default_rng(seed)
    env = OrbitEnv(target)
    Q = np.zeros((NS, NA))  # optimistic: almost every reward is <= 0, so untried actions look good
    h_dv = hohmann_dv(R0, target)[2]
    print(f"target r={target}  Hohmann dv={h_dv:.4f}  success tol={env.tol:.3f}  table {NS}x{NA}")

    hist = {"episode": [], "score": [], "success": [], "excess": []}
    best_score, best_Q = -np.inf, Q.copy()

    for ep in range(1, episodes + 1):
        eps = max(eps_end, 1.0 - (1.0 - eps_end) * ep / (0.7 * episodes))
        s = env.reset()
        while True:
            a = int(rng.integers(NA)) if rng.random() < eps else int(Q[s].argmax())
            s2, r, term, trunc, _ = env.step(a)
            tgt = r if term else r + GAMMA * Q[s2].max()
            Q[s, a] += alpha * (tgt - Q[s, a])
            s = s2
            if term or trunc:
                break

        if ep % eval_every == 0:
            score, ok, dv = greedy_episode(env, Q)
            excess = (dv - h_dv) / h_dv * 100 if ok else float("nan")
            for k, v in zip(hist, (ep, score, ok, excess)):
                hist[k].append(v)
            if score > best_score:
                best_score, best_Q = score, Q.copy()
            tail = f"dv={dv:.4f} ({excess:+.1f}% vs Hohmann)" if ok else f"dv={dv:.4f} (no success)"
            print(f"ep {ep:6d}  eps={eps:.2f}  eval score={score:8.3f}  {tail}")

    np.savez(out, Q=best_Q, Q_final=Q, target=target, **{k: np.array(v) for k, v in hist.items()})
    print(f"saved best-by-eval Q (score {best_score:.3f}) to {out}")


def plot(out):
    import matplotlib.pyplot as plt

    d = np.load(out)
    target = float(d["target"])
    env = OrbitEnv(target)
    score, ok, dv = greedy_episode(env, d["Q"])
    rl = np.array(env.traj)
    hoh = hohmann_trajectory(R0, target)
    th = np.linspace(0, 2 * np.pi, 200)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax.plot(np.cos(th), np.sin(th), "k--", lw=0.8, label="initial orbit")
    ax.plot(target * np.cos(th), target * np.sin(th), "g--", lw=0.8, label="target orbit")
    ax.plot(*hoh.T, "b-", label=f"Hohmann (dv={hohmann_dv(R0, target)[2]:.3f})")
    ax.plot(*rl.T, "r-", label=f"Q-learning (dv={dv:.3f}, success={ok})")
    ax.plot(0, 0, "yo")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Transfer to r = {target}")

    ax2.plot(d["episode"], d["score"])
    ax2.set_xlabel("training episode")
    ax2.set_ylabel("greedy eval score (return)")
    ax2.set_title("Learning curve")
    plt.tight_layout()
    fname = f"transfer_q_r{target:g}.png"
    fig.savefig(fname, dpi=150)
    print(f"saved {fname}")
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "plot", "hohmann"])
    p.add_argument("--target", type=float, default=1.5)
    p.add_argument("--episodes", type=int, default=10_000)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--out", default="q_orbit.npz")
    args = p.parse_args()

    if args.mode == "train":
        train(args.target, args.episodes, args.eval_every, args.alpha, args.out)
    elif args.mode == "plot":
        plot(args.out)
    else:
        dv1, dv2, tot, t = hohmann_dv(R0, args.target)
        print(f"r2={args.target}  dv1={dv1:+.4f}  dv2={dv2:+.4f}  total={tot:.4f}  t_transfer={t:.3f}")
