"""
3D Ising model on an L x L x L cubic lattice with periodic boundary conditions.
Metropolis Monte Carlo with vectorized checkerboard updates.

    H = -J * sum_<ij> s_i s_j  -  h * sum_i s_i      (units: J = k_B = 1)

Known 3D critical temperature: T_c ~ 4.5115
"""

import numpy as np
import matplotlib.pyplot as plt

TC_3D = 4.5115


# ---------------------------------------------------------------- lattice ---
def init_lattice(L, state="random", rng=None):
    rng = np.random.default_rng() if rng is None else rng
    if state == "up":
        return np.ones((L, L, L), dtype=np.int8)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L, L))


def neighbor_sum(s):
    """Sum of the 6 nearest neighbors of every site (periodic BCs)."""
    return (np.roll(s, 1, 0) + np.roll(s, -1, 0) +
            np.roll(s, 1, 1) + np.roll(s, -1, 1) +
            np.roll(s, 1, 2) + np.roll(s, -1, 2))


def energy(s, J=1.0, h=0.0):
    """Total energy. Only the +1 rolls are used so each bond is counted once."""
    bonds = s * (np.roll(s, 1, 0) + np.roll(s, 1, 1) + np.roll(s, 1, 2))
    return -J * bonds.sum() - h * s.sum()


def magnetization(s):
    return s.sum()


def checkerboard_masks(L):
    i, j, k = np.indices((L, L, L))
    black = (i + j + k) % 2 == 0
    return black, ~black


# ---------------------------------------------------------- Monte Carlo ---
def metropolis_sweep(s, beta, masks, J=1.0, h=0.0, rng=None):
    """One full lattice sweep: update black sites, then white sites.
    Requires even L so the checkerboard is consistent across the periodic wrap."""
    rng = np.random.default_rng() if rng is None else rng
    for mask in masks:
        dE = 2 * s * (J * neighbor_sum(s) + h)
        accept = (dE <= 0) | (rng.random(s.shape) < np.exp(-beta * dE))
        s[mask & accept] *= -1
    return s


def simulate(L, T, n_eq=1000, n_meas=2000, n_skip=5, J=1.0, h=0.0,
             s=None, rng=None):
    """Run at one temperature and return per-spin observables."""
    if L % 2:
        raise ValueError("Use even L for the checkerboard update.")
    rng = np.random.default_rng() if rng is None else rng
    s = init_lattice(L, "up", rng) if s is None else s
    masks = checkerboard_masks(L)
    beta, N = 1.0 / T, L**3

    for _ in range(n_eq):
        metropolis_sweep(s, beta, masks, J, h, rng)

    E = np.empty(n_meas)
    M = np.empty(n_meas)
    for n in range(n_meas):
        for _ in range(n_skip):
            metropolis_sweep(s, beta, masks, J, h, rng)
        E[n] = energy(s, J, h)
        M[n] = magnetization(s)

    e, m = E / N, np.abs(M) / N
    results = {
        "T": T,
        "e": e.mean(),
        "m": m.mean(),
        "c": beta**2 * N * e.var(),                       # heat capacity / spin
        "chi": beta * N * (np.mean(m**2) - m.mean()**2),  # susceptibility / spin
        "U": 1 - np.mean(m**4) / (3 * np.mean(m**2)**2),  # Binder cumulant
    }
    return results, s


def temperature_sweep(L, temps, **kwargs):
    """Sweep in increasing T, reusing the previous configuration."""
    rng = np.random.default_rng(kwargs.pop("seed", None))
    s, out = None, []
    for T in temps:
        r, s = simulate(L, T, s=s, rng=rng, **kwargs)
        out.append(r)
        print(f"L={L:3d}  T={T:.3f}  e={r['e']:+.4f}  |m|={r['m']:.4f}  "
              f"c={r['c']:.3f}  chi={r['chi']:.3f}  U={r['U']:.3f}")
    return {k: np.array([r[k] for r in out]) for k in out[0]}


# ------------------------------------------------------------- plotting ---
def plot_results(data_by_L):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panels = [("e", "Energy per spin"), ("m", "|Magnetization| per spin"),
              ("c", "Heat capacity per spin"), ("chi", "Susceptibility per spin")]
    for ax, (key, title) in zip(axes.flat, panels):
        for L, d in data_by_L.items():
            ax.plot(d["T"], d[key], "o-", ms=3, label=f"L={L}")
        ax.axvline(TC_3D, color="gray", ls="--", lw=1, label="$T_c$≈4.51")
        ax.set_title(title)
        ax.set_xlabel("T")
        ax.legend()
    plt.savefig('ising_results.png')
    fig.tight_layout()

    fig2, ax = plt.subplots(figsize=(6, 4))
    for L, d in data_by_L.items():
        ax.plot(d["T"], d["U"], "o-", ms=3, label=f"L={L}")
    ax.axvline(TC_3D, color="gray", ls="--", lw=1)
    ax.set_xlabel("T")
    ax.set_ylabel("Binder cumulant U")
    ax.set_title("Curves for different L cross near $T_c$")
    ax.legend()
    fig2.tight_layout()
    plt.savefig('curves_near_T_c.png')
    # plt.show()


def plot_slice(s, z=0):
    """Show one z-slice of the 3D lattice."""
    plt.imshow(s[:, :, z], cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"z = {z} slice")
    plt.axis("off")
    plt.savefig('z_slice.png')
    # plt.show()


# ----------------------------------------------------------------- main ---
if __name__ == "__main__":
    temps = np.linspace(3.5, 5.5, 21)
    data = {}
    for L in (8, 12, 16):
        data[L] = temperature_sweep(L, temps, n_eq=500, n_meas=1000,
                                    n_skip=3, seed=42)
    plot_results(data)