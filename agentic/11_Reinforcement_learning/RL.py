"""
Stage 1: 2D circular -> circular orbit transfer with RL, checked against Hohmann.

Units are normalised: central body mu = 1, initial orbit radius r0 = 1
(so circular speed = 1 and the orbital period = 2*pi).

"Fuel" is measured as total delta-v (sum of |thrust accel| * dt). Mass is not
modelled in this stage; delta-v is a monotonic proxy for propellant via the
rocket equation, and it is exactly what Hohmann minimises.

Usage:
    python RL.py train  [--steps 300000] [--model sac_orbit] [--envs 8] [--utd 0.5] [--eval-freq 10000]
    python RL.py eval   [--model sac_orbit]
    python RL.py plot   [--model sac_orbit] [--target 1.5]
    python RL.py hohmann                      # just print the analytic baseline
"""

import argparse

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback

MU = 1.0
R0 = 1.0
EVAL_TARGETS = (0.7, 0.85, 1.2, 1.5, 1.8, 2.0)


# ----------------------------------------------------------------------------
# Physics
# ----------------------------------------------------------------------------
def _deriv(s, acc):
    """s = [..., x, y, vx, vy]; acc = [..., ax, ay] constant thrust (inertial frame).
    Works on a single state (shape (4,)) or a batch (shape (N, 4))."""
    r = np.hypot(s[..., 0], s[..., 1])
    g = -MU / r**3
    return np.stack(
        [s[..., 2], s[..., 3], g * s[..., 0] + acc[..., 0], g * s[..., 1] + acc[..., 1]], axis=-1
    )


def rk4_step(s, acc, dt):
    k1 = _deriv(s, acc)
    k2 = _deriv(s + 0.5 * dt * k1, acc)
    k3 = _deriv(s + 0.5 * dt * k2, acc)
    k4 = _deriv(s + dt * k3, acc)
    return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def orbit_elements(s):
    """Return (semi-major axis, eccentricity). a = inf/negative if unbound."""
    r_vec, v_vec = s[:2], s[2:]
    r = np.linalg.norm(r_vec)
    v2 = v_vec @ v_vec
    inv_a = 2.0 / r - v2 / MU
    a = np.inf if abs(inv_a) < 1e-12 else 1.0 / inv_a
    e_vec = ((v2 - MU / r) * r_vec - (r_vec @ v_vec) * v_vec) / MU
    return a, float(np.linalg.norm(e_vec))


# ----------------------------------------------------------------------------
# Hohmann baseline
# ----------------------------------------------------------------------------
def hohmann_dv(r1, r2):
    """Analytic Hohmann delta-v. Returns (dv1, dv2, total, transfer_time).
    Works for raising (r2 > r1) and lowering (r2 < r1); dv signs are tangential."""
    a_t = 0.5 * (r1 + r2)
    v_c1, v_c2 = np.sqrt(MU / r1), np.sqrt(MU / r2)
    v_t1 = np.sqrt(MU * (2.0 / r1 - 1.0 / a_t))
    v_t2 = np.sqrt(MU * (2.0 / r2 - 1.0 / a_t))
    dv1, dv2 = v_t1 - v_c1, v_c2 - v_t2
    t_transfer = np.pi * np.sqrt(a_t**3 / MU)
    return dv1, dv2, abs(dv1) + abs(dv2), t_transfer


def hohmann_trajectory(r1, r2, dt=0.01):
    """Propagate the impulsive Hohmann transfer with the same integrator. Returns Nx2 xy."""
    dv1, dv2, _, t_tr = hohmann_dv(r1, r2)
    s = np.array([r1, 0.0, 0.0, np.sqrt(MU / r1) + dv1])
    zero = np.zeros(2)
    pts = [s[:2].copy()]
    n = int(round(t_tr / dt))
    for _ in range(n):
        s = rk4_step(s, zero, t_tr / n)
        pts.append(s[:2].copy())
    # apply second burn (tangential, along current velocity), coast a bit on the final orbit
    v_hat = s[2:] / np.linalg.norm(s[2:])
    s[2:] += dv2 * v_hat
    for _ in range(int(2 * np.pi * np.sqrt(r2**3 / MU) / dt)):
        s = rk4_step(s, zero, dt)
        pts.append(s[:2].copy())
    return np.array(pts)


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
class OrbitTransferEnv(gym.Env):
    """
    Observation (rotation invariant, so the agent can't latch onto absolute angle):
        [r, v_radial, v_tangential, r_target, (a - a_t)/a_t, e, time_fraction]
    Action: thrust in the local (radial, tangential) frame, each in [-1, 1].
        If its norm exceeds 1 it is rescaled, so max thrust accel = a_max.
    Reward per step:
        - fuel_weight * delta_v
        + potential-based shaping (gamma*phi(s') - phi(s), phi = -orbit_error)
        + success_bonus on reaching the target orbit / - crash_penalty on failure
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        r_target=None,  # fixed target radius, or None to randomise each episode
        target_cycle=None,  # or a tuple of targets to cycle through (used for eval)
        target_range=(0.7, 2.0),
        a_max=0.5,
        dt=0.05,
        max_steps=400,
        tol_a=0.01,
        tol_e=0.01,
        fuel_weight=10.0,
        shaping_weight=1.0,
        success_bonus=5.0,
        fail_penalty=5.0,
        r_crash=0.3,
        r_escape=4.0,
        gamma=0.999,
    ):
        super().__init__()
        self.fixed_target = r_target
        self.target_cycle = target_cycle
        self._cycle_i = 0
        self.target_range = target_range
        self.a_max, self.dt, self.max_steps = a_max, dt, max_steps
        self.tol_a, self.tol_e = tol_a, tol_e
        self.fuel_weight, self.shaping_weight = fuel_weight, shaping_weight
        self.success_bonus, self.fail_penalty = success_bonus, fail_penalty
        self.r_crash, self.r_escape = r_crash, r_escape
        self.gamma = gamma

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)

    # -- helpers -------------------------------------------------------------
    def _sample_target(self):
        if self.fixed_target is not None:
            return float(self.fixed_target)
        if self.target_cycle is not None:
            r = self.target_cycle[self._cycle_i % len(self.target_cycle)]
            self._cycle_i += 1
            return float(r)
        while True:
            r = self.np_random.uniform(*self.target_range)
            if abs(r - R0) > 0.1:  # avoid trivial near-zero transfers
                return float(r)

    def _frame(self):
        r_vec, v_vec = self.s[:2], self.s[2:]
        r = np.linalg.norm(r_vec)
        r_hat = r_vec / r
        t_hat = np.array([-r_hat[1], r_hat[0]])
        return r, r_hat, t_hat, v_vec @ r_hat, v_vec @ t_hat

    def _error(self):
        a, e = orbit_elements(self.s)
        if not np.isfinite(a) or a <= 0:
            return 10.0  # unbound
        return abs(a - self.r_target) / self.r_target + e

    def _obs(self):
        r, _, _, v_r, v_t = self._frame()
        a, e = orbit_elements(self.s)
        a_err = 10.0 if (not np.isfinite(a) or a <= 0) else (a - self.r_target) / self.r_target
        return np.array(
            [r, v_r, v_t, self.r_target, np.clip(a_err, -10, 10), min(e, 10.0), self.steps / self.max_steps],
            dtype=np.float32,
        )

    # -- gym API -------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.r_target = self._sample_target()
        self.s = np.array([R0, 0.0, 0.0, np.sqrt(MU / R0)])
        self.steps = 0
        self.total_dv = 0.0
        self.trajectory = [self.s[:2].copy()]
        return self._obs(), {"r_target": self.r_target}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        n = np.linalg.norm(action)
        if n > 1.0:
            action = action / n

        _, r_hat, t_hat, _, _ = self._frame()
        acc = self.a_max * (action[0] * r_hat + action[1] * t_hat)

        phi_old = -self._error()
        self.s = rk4_step(self.s, acc, self.dt)
        self.steps += 1
        dv = float(np.linalg.norm(acc) * self.dt)
        self.total_dv += dv
        self.trajectory.append(self.s[:2].copy())
        phi_new = -self._error()

        reward = -self.fuel_weight * dv + self.shaping_weight * (self.gamma * phi_new - phi_old)

        r = np.linalg.norm(self.s[:2])
        a, e = orbit_elements(self.s)
        success = (
            np.isfinite(a)
            and a > 0
            and abs(a - self.r_target) / self.r_target < self.tol_a
            and e < self.tol_e
        )
        failed = r < self.r_crash or r > self.r_escape or not (np.isfinite(a) and a > 0)

        terminated = bool(success or failed)
        truncated = bool(not terminated and self.steps >= self.max_steps)
        if success:
            reward += self.success_bonus
        elif failed:
            reward -= self.fail_penalty

        info = {
            "success": bool(success),
            "is_success": bool(success),  # picked up by EvalCallback for success rate
            "total_dv": self.total_dv,
            "r_target": self.r_target,
        }
        return self._obs(), float(reward), terminated, truncated, info


# ----------------------------------------------------------------------------
# Batched environment: N episodes advanced by one numpy RK4 call per step.
# Same dynamics / reward / termination as OrbitTransferEnv (keep them in sync).
# ----------------------------------------------------------------------------
def elements_batch(s):
    """Batched (a, e, bound) for states of shape (N, 4)."""
    r_vec, v_vec = s[:, :2], s[:, 2:]
    r = np.linalg.norm(r_vec, axis=1)
    v2 = (v_vec**2).sum(1)
    rv = (r_vec * v_vec).sum(1)
    inv_a = 2.0 / r - v2 / MU
    bound = inv_a > 1e-12
    a = np.where(bound, 1.0 / np.where(bound, inv_a, 1.0), np.inf)
    e_vec = ((v2 - MU / r)[:, None] * r_vec - rv[:, None] * v_vec) / MU
    return a, np.linalg.norm(e_vec, axis=1), bound


class OrbitTransferVecEnv(VecEnv):
    def __init__(
        self,
        n_envs=8,
        target_range=(0.7, 2.0),
        a_max=0.5,
        dt=0.05,
        max_steps=400,
        tol_a=0.01,
        tol_e=0.01,
        fuel_weight=10.0,
        shaping_weight=1.0,
        success_bonus=5.0,
        fail_penalty=5.0,
        r_crash=0.3,
        r_escape=4.0,
        gamma=0.999,
        seed=None,
    ):
        obs_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)
        act_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        super().__init__(n_envs, obs_space, act_space)
        self.target_range = target_range
        self.a_max, self.dt, self.max_steps = a_max, dt, max_steps
        self.tol_a, self.tol_e = tol_a, tol_e
        self.fuel_weight, self.shaping_weight = fuel_weight, shaping_weight
        self.success_bonus, self.fail_penalty = success_bonus, fail_penalty
        self.r_crash, self.r_escape, self.gamma = r_crash, r_escape, gamma

        self.rng = np.random.default_rng(seed)
        self.s = np.zeros((n_envs, 4))
        self.r_target = np.ones(n_envs)
        self.steps = np.zeros(n_envs, dtype=int)
        self.total_dv = np.zeros(n_envs)
        self._actions = None
        self.reset_infos = [{} for _ in range(n_envs)]

    # -- helpers -------------------------------------------------------------
    def _sample_targets(self, n):
        r = self.rng.uniform(*self.target_range, size=n)
        bad = np.abs(r - R0) <= 0.1
        while bad.any():
            r[bad] = self.rng.uniform(*self.target_range, size=bad.sum())
            bad = np.abs(r - R0) <= 0.1
        return r

    def _reset_idx(self, idx):
        self.s[idx] = [R0, 0.0, 0.0, np.sqrt(MU / R0)]
        self.r_target[idx] = self._sample_targets(len(idx))
        self.steps[idx] = 0
        self.total_dv[idx] = 0.0

    def _frame(self):
        r = np.linalg.norm(self.s[:, :2], axis=1)
        r_hat = self.s[:, :2] / r[:, None]
        t_hat = np.stack([-r_hat[:, 1], r_hat[:, 0]], axis=1)
        return r, r_hat, t_hat

    def _error(self):
        a, e, bound = elements_batch(self.s)
        err = np.where(bound, np.abs(a - self.r_target) / self.r_target + e, 10.0)
        return err, a, e, bound

    def _obs(self):
        r, r_hat, t_hat = self._frame()
        v = self.s[:, 2:]
        a, e, bound = elements_batch(self.s)
        a_err = np.where(bound, (a - self.r_target) / self.r_target, 10.0)
        obs = np.stack(
            [
                r,
                (v * r_hat).sum(1),
                (v * t_hat).sum(1),
                self.r_target,
                np.clip(a_err, -10, 10),
                np.minimum(e, 10.0),
                self.steps / self.max_steps,
            ],
            axis=1,
        )
        return obs.astype(np.float32)

    # -- VecEnv API ----------------------------------------------------------
    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        return [seed] * self.num_envs

    def reset(self):
        self._reset_idx(np.arange(self.num_envs))
        return self._obs()

    def step_async(self, actions):
        self._actions = np.asarray(actions, dtype=np.float64)

    def step_wait(self):
        act = self._actions
        act = act / np.maximum(np.linalg.norm(act, axis=1, keepdims=True), 1.0)
        _, r_hat, t_hat = self._frame()
        acc = self.a_max * (act[:, :1] * r_hat + act[:, 1:] * t_hat)

        phi_old = -self._error()[0]
        self.s = rk4_step(self.s, acc, self.dt)
        self.steps += 1
        dv = np.linalg.norm(acc, axis=1) * self.dt
        self.total_dv += dv
        err_new, a, e, bound = self._error()

        reward = -self.fuel_weight * dv + self.shaping_weight * (self.gamma * (-err_new) - phi_old)

        r = np.linalg.norm(self.s[:, :2], axis=1)
        success = bound & (np.abs(a - self.r_target) / self.r_target < self.tol_a) & (e < self.tol_e)
        failed = (r < self.r_crash) | (r > self.r_escape) | ~bound
        terminated = success | failed
        truncated = ~terminated & (self.steps >= self.max_steps)
        reward = reward + np.where(success, self.success_bonus, 0.0)
        reward = reward - np.where(failed & ~success, self.fail_penalty, 0.0)

        done = terminated | truncated
        infos = [{} for _ in range(self.num_envs)]
        if done.any():
            term_obs = self._obs()
            idx = np.flatnonzero(done)
            for i in idx:
                infos[i] = {
                    "terminal_observation": term_obs[i].copy(),
                    "TimeLimit.truncated": bool(truncated[i]),
                    "success": bool(success[i]),
                    "total_dv": float(self.total_dv[i]),
                    "r_target": float(self.r_target[i]),
                }
            self._reset_idx(idx)
        return self._obs(), reward.astype(np.float32), done.copy(), infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self, method_name)(*args, **kwargs) for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


# ----------------------------------------------------------------------------
# Train / evaluate / plot
# ----------------------------------------------------------------------------
class ScheduleCallback(BaseCallback):
    """Linearly anneal fuel weight and success tolerance over the first `warm_frac`
    of training: easy (cheap fuel, loose tolerance) -> final objective."""

    def __init__(self, total_steps, warm_frac, fuel=(2.0, 10.0), tol=(0.03, 0.01)):
        super().__init__()
        self.warm_steps = max(1, int(total_steps * warm_frac))
        self.fuel, self.tol = fuel, tol

    def _on_step(self):
        if self.n_calls % 50 == 1:
            f = min(1.0, self.num_timesteps / self.warm_steps)
            fw = self.fuel[0] + (self.fuel[1] - self.fuel[0]) * f
            tl = self.tol[0] + (self.tol[1] - self.tol[0]) * f
            self.training_env.set_attr("fuel_weight", fw)
            self.training_env.set_attr("tol_a", tl)
            self.training_env.set_attr("tol_e", tl)
            self.logger.record("sched/fuel_weight", fw)
            self.logger.record("sched/tolerance", tl)
        return True


class OrbitEvalCallback(EvalCallback):
    """EvalCallback that also prints success rate and mean dv excess over Hohmann
    (successful episodes only) after each evaluation."""

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        if locals_["done"]:
            info = locals_["info"]
            h = hohmann_dv(R0, info["r_target"])[2]
            self._eps = getattr(self, "_eps", [])
            self._eps.append((info["is_success"], (info["total_dv"] - h) / h * 100))

    def _on_step(self):
        cont = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0 and getattr(self, "_eps", None):
            ok = [x for s, x in self._eps if s]
            msg = f"  success {len(ok)}/{len(self._eps)}"
            if ok:
                msg += f", mean dv excess over Hohmann {np.mean(ok):+.1f}%"
            print(msg)
            self.logger.record("eval/dv_excess_pct", float(np.mean(ok)) if ok else float("nan"))
            self._eps = []
        return cont


def train(steps, model_path, n_envs=8, utd=0.5, eval_freq=10_000, fuel_start=2.0, tol_start=0.03, warm_frac=0.5):
    """
    n_envs:    parallel (batched) training episodes.
    utd:       gradient updates per environment transition (SAC default-ish is 1.0;
               lower = faster but less sample-efficient).
    eval_freq: run the eval episodes every this many *environment* steps.
    fuel_start, tol_start, warm_frac: training starts with a cheap fuel weight and loose
               tolerance, annealed to the final values (10.0, 0.01) over warm_frac of training.
               Eval always uses the FINAL objective, so scores are comparable across time.
    """
    from stable_baselines3 import SAC

    env = VecMonitor(OrbitTransferVecEnv(n_envs, fuel_weight=fuel_start, tol_a=tol_start, tol_e=tol_start))

    # Deterministic policy, one episode per fixed target -> score is comparable across evals.
    eval_env = OrbitTransferEnv(target_cycle=EVAL_TARGETS)
    eval_cb = OrbitEvalCallback(
        eval_env,
        n_eval_episodes=len(EVAL_TARGETS),
        eval_freq=max(eval_freq // n_envs, 1),  # counted in vec-env steps
        deterministic=True,
        best_model_save_path=f"{model_path}_best",  # -> <model>_best/best_model.zip
        log_path=f"{model_path}_eval",  # -> evaluations.npz (timesteps, rewards)
        verbose=1,
    )

    model = SAC(
        "MlpPolicy",
        env,
        gamma=0.999,
        learning_rate=3e-4,
        batch_size=256,
        train_freq=1,
        gradient_steps=max(1, int(n_envs * utd)),
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1,
    )
    model.learn(total_timesteps=steps, callback=CallbackList([ScheduleCallback(steps, warm_frac, (fuel_start, 10.0), (tol_start, 0.01)), eval_cb]))
    model.save(model_path)
    print(f"saved {model_path} (best-by-eval: {model_path}_best/best_model.zip)")


def run_episode(model, env, r_target):
    obs, _ = env.reset(options=None)
    # force the target for this episode
    env.r_target = float(r_target)
    obs = env._obs()
    done = False
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
    return info


def evaluate(model_path, targets=(0.7, 0.85, 1.2, 1.5, 1.8, 2.0)):
    from stable_baselines3 import SAC

    model = SAC.load(model_path)
    env = OrbitTransferEnv()
    print(f"{'r_target':>8} {'success':>8} {'RL dv':>8} {'Hohmann dv':>11} {'excess %':>9}")
    for rt in targets:
        info = run_episode(model, env, rt)
        h = hohmann_dv(R0, rt)[2]
        excess = 100 * (info["total_dv"] - h) / h
        print(f"{rt:8.2f} {str(info['success']):>8} {info['total_dv']:8.4f} {h:11.4f} {excess:9.1f}")


def plot(model_path, r_target):
    import matplotlib.pyplot as plt
    from stable_baselines3 import SAC

    model = SAC.load(model_path)
    env = OrbitTransferEnv()
    info = run_episode(model, env, r_target)
    rl = np.array(env.trajectory)
    hoh = hohmann_trajectory(R0, r_target)
    th = np.linspace(0, 2 * np.pi, 200)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(np.cos(th), np.sin(th), "k--", lw=0.8, label="initial orbit")
    ax.plot(r_target * np.cos(th), r_target * np.sin(th), "g--", lw=0.8, label="target orbit")
    ax.plot(*hoh.T, "b-", label=f"Hohmann (dv={hohmann_dv(R0, r_target)[2]:.3f})")
    ax.plot(*rl.T, "r-", label=f"RL (dv={info['total_dv']:.3f})")
    ax.plot(0, 0, "yo")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Transfer to r = {r_target}")
    fname = f"transfer_sac_r{r_target:g}.png"
    fig.savefig(fname, dpi=150)
    print(f"saved {fname}")
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "eval", "plot", "hohmann"])
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--model", default="sac_orbit")
    p.add_argument("--target", type=float, default=1.5)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--utd", type=float, default=0.5)
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--fuel-start", type=float, default=2.0)
    p.add_argument("--tol-start", type=float, default=0.03)
    p.add_argument("--warm-frac", type=float, default=0.5)
    args = p.parse_args()

    if args.mode == "train":
        train(args.steps, args.model, args.envs, args.utd, args.eval_freq,
              args.fuel_start, args.tol_start, args.warm_frac)
    elif args.mode == "eval":
        evaluate(args.model)
    elif args.mode == "plot":
        plot(args.model, args.target)
    else:
        for rt in (0.7, 1.2, 1.5, 2.0):
            dv1, dv2, tot, t = hohmann_dv(R0, rt)
            print(f"r2={rt:4.2f}  dv1={dv1:+.4f}  dv2={dv2:+.4f}  total={tot:.4f}  t_transfer={t:.3f}")
