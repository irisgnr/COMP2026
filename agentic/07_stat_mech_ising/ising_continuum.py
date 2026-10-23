"""
Approaching the continuum limit of the 3D Ising model.

Near T_c the correlation length xi diverges in lattice units, so xi/a -> infinity
is the continuum limit. Holding xi/L fixed while L grows is finite-size scaling.
The universal data of the continuum theory (the 3D Ising CFT) show up as:
    xi/L at T_c         -> ~0.643       (universal, periodic cube)
    Binder U at T_c     -> ~0.465
    chi ~ L^(gamma/nu), gamma/nu = 2 - eta ~ 1.964
    curves collapse vs x = (T - T_c) * L^(1/nu), nu ~ 0.630

Uses the Wolff single-cluster algorithm to beat critical slowing down.
Units: J = k_B = 1.
"""

import numpy as np
import matplotlib.pyplot as plt

TC = 4.511528          # beta_c = 0.2216546
NU = 0.62997
ETA = 0.03630
GAMMA_OVER_NU = 2 - ETA


# ------------------------------------------------------------ Wolff ---
def neighbor_table(L):
    """(N, 6) array of flat neighbor indices on a periodic L^3 lattice."""
    idx = np.arange(L**3).reshape(L, L, L)
    nbrs = [np.roll(idx, sh, ax).ravel() for ax in range(3) for sh in (1, -1)]
    return np.stack(nbrs, axis=1)


def wolff_step(s, nbrs, p_add, rng):
    """Grow one Wolff cluster from a random seed and flip it.
    s is a flat int8 array. Grows layer by layer (vectorized frontier).
    Returns the cluster size."""
    N = s.size
    seed = rng.integers(N)
    spin = s[seed]
    in_cluster = np.zeros(N, dtype=bool)
    in_cluster[seed] = True
    frontier = np.array([seed])
    size = 1
    while frontier.size:
        cand = nbrs[frontier].ravel()                   # each bond tested once
        ok = (s[cand] == spin) & ~in_cluster[cand]
        ok &= rng.random(cand.size) < p_add
        new = np.unique(cand[ok])
        in_cluster[new] = True
        size += new.size
        frontier = new
    s[in_cluster] *= -1
    return size


# ------------------------------------------------------ observables ---
def fourier_moments(s3d):
    """|M(k=0)|^2 and |M(k_min)|^2 averaged over the 3 axes."""
    L = s3d.shape[0]
    phase = np.exp(2j * np.pi * np.arange(L) / L)
    M0 = s3d.sum()
    Mk = 0.0
    for ax in range(3):
        profile = s3d.sum(axis=tuple(a for a in range(3) if a != ax))
        Mk += np.abs(profile @ phase) ** 2
    return M0, Mk / 3


def simulate_wolff(L, T, n_eq=500, n_meas=3000, rng=None, s=None):
    rng = np.random.default_rng() if rng is None else rng
    N = L**3
    nbrs = neighbor_table(L)
    s = np.ones(N, dtype=np.int8) if s is None else s
    p_add = 1 - np.exp(-2.0 / T)

    # one "sweep" ~ enough clusters to touch about N spins
    for _ in range(n_eq):
        wolff_step(s, nbrs, p_add, rng)
    avg_c = np.mean([wolff_step(s, nbrs, p_add, rng) for _ in range(50)])
    per_meas = max(1, int(N / avg_c))

    m2 = np.empty(n_meas); m4 = np.empty(n_meas); mk2 = np.empty(n_meas)
    mabs = np.empty(n_meas)
    for n in range(n_meas):
        for _ in range(per_meas):
            wolff_step(s, nbrs, p_add, rng)
        M0, Mk2 = fourier_moments(s.reshape(L, L, L))
        m = M0 / N
        mabs[n], m2[n], m4[n] = abs(m), m**2, m**4
        mk2[n] = Mk2 / N**2

    chi = N * m2.mean() / T                  # chi without subtraction (FSS standard)
    ratio = m2.mean() / mk2.mean()
    kmin = 2 * np.pi / L
    xi = np.sqrt(max(ratio - 1, 0)) / (2 * np.sin(kmin / 2))
    U = 1 - m4.mean() / (3 * m2.mean() ** 2)
    return {"T": T, "m": mabs.mean(), "chi": chi, "xi": xi,
            "xi_over_L": xi / L, "U": U}, s


def sweep(L, temps, seed=0, **kw):
    rng = np.random.default_rng(seed)
    s, rows = None, []
    for T in temps:
        r, s = simulate_wolff(L, T, rng=rng, s=s, **kw)
        rows.append(r)
        print(f"L={L:3d} T={T:.4f}  xi={r['xi']:6.2f}  xi/L={r['xi_over_L']:.3f}"
              f"  U={r['U']:.3f}  chi={r['chi']:9.2f}")
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


# --------------------------------------------------------- analysis ---
def plot_fss(data):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    for L, d in data.items():
        x = (d["T"] - TC) * L ** (1 / NU)
        ax[0, 0].plot(d["T"], d["xi_over_L"], "o-", ms=3, label=f"L={L}")
        ax[0, 1].plot(x, d["xi_over_L"], "o", ms=3, label=f"L={L}")
        ax[1, 0].plot(d["T"], d["U"], "o-", ms=3, label=f"L={L}")
        ax[1, 1].plot(x, d["chi"] * L ** (-GAMMA_OVER_NU), "o", ms=3, label=f"L={L}")

    ax[0, 0].set(title="ξ/L: curves cross at $T_c$", xlabel="T", ylabel="ξ/L")
    ax[0, 0].axvline(TC, ls="--", c="gray")
    ax[0, 1].set(title="Collapse of ξ/L", xlabel=r"$(T-T_c)L^{1/\nu}$")
    ax[1, 0].set(title="Binder cumulant", xlabel="T", ylabel="U")
    ax[1, 0].axvline(TC, ls="--", c="gray")
    ax[1, 1].set(title=r"Collapse of $\chi L^{-\gamma/\nu}$",
                 xlabel=r"$(T-T_c)L^{1/\nu}$")
    for a in ax.flat:
        a.legend()
    fig.tight_layout()

    # chi at the temperature closest to Tc vs L -> slope gamma/nu
    Ls = np.array(sorted(data))
    chis = np.array([data[L]["chi"][np.argmin(abs(data[L]["T"] - TC))] for L in Ls])
    slope = np.polyfit(np.log(Ls), np.log(chis), 1)[0]
    print(f"\nFit gamma/nu = {slope:.3f}   (continuum value {GAMMA_OVER_NU:.3f})")

    fig2, a = plt.subplots(figsize=(5, 4))
    a.loglog(Ls, chis, "o", label="data")
    a.loglog(Ls, chis[0] * (Ls / Ls[0]) ** GAMMA_OVER_NU, "--", label=r"$L^{1.964}$")
    a.set(xlabel="L", ylabel=r"$\chi(T_c)$", title="Critical scaling")
    a.legend()
    fig2.tight_layout()
    plt.show()


if __name__ == "__main__":
    temps = np.linspace(4.40, 4.62, 12)
    temps = np.sort(np.append(temps, TC))       # make sure T_c itself is sampled
    data = {L: sweep(L, temps, n_eq=300, n_meas=1500) for L in (8, 12, 16, 24)}
    plot_fss(data)