"""
Extract the scaling dimension Delta_sigma of the 3D Ising spin field from the
critical two-point function

    G(r) = <s(0) s(r)>  ~  1 / r^(2 Delta_sigma)      at T = T_c,  1 << r << L

Two estimators:
  1. Direct fit of G(r) on the largest lattice, including the nearest periodic
     image along the measured axis.
  2. Finite-size scaling: scale invariance implies G(r = L/2) ~ L^(-2 Delta_sigma)
     exactly (up to corrections ~ L^-omega). Same for <m^2> = (1/N) sum_r G(r).

Conformal bootstrap value: Delta_sigma = 0.5181489(10)
Requires ising3d_continuum.py in the same folder (for the Wolff update).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from ising_continuum import neighbor_table, wolff_step, TC

DELTA_SIGMA_BOOTSTRAP = 0.5181489
OMEGA = 0.83          # leading correction-to-scaling exponent in 3D Ising


# ------------------------------------------------------ measurement ---
def measure_G(L, T=TC, n_eq=1000, n_meas=4000, n_blocks=20, seed=0):
    """Return G(x) on the full L^3 lattice (block averages, shape
    (n_blocks, L, L, L)) and per-block <m^2>.

    G(x) = (1/N) sum_y <s_y s_{y+x}> computed with FFTs: |FFT(s)|^2 -> IFFT.
    """
    rng = np.random.default_rng(seed)
    N = L**3
    nbrs = neighbor_table(L)
    s = np.ones(N, dtype=np.int8)
    p_add = 1 - np.exp(-2.0 / T)

    for _ in range(n_eq):
        wolff_step(s, nbrs, p_add, rng)
    avg_c = np.mean([wolff_step(s, nbrs, p_add, rng) for _ in range(100)])
    per_meas = max(1, int(round(N / avg_c)))   # ~one lattice volume flipped

    per_block = n_meas // n_blocks
    G_blocks = np.zeros((n_blocks, L, L, L))
    m2_blocks = np.zeros(n_blocks)
    for b in range(n_blocks):
        power = np.zeros((L, L, L))
        m2 = 0.0
        for _ in range(per_block):
            for _ in range(per_meas):
                wolff_step(s, nbrs, p_add, rng)
            S = np.fft.fftn(s.reshape(L, L, L).astype(np.float64))
            power += np.abs(S) ** 2
            m2 += (s.sum() / N) ** 2
        G_blocks[b] = np.fft.ifftn(power / per_block).real / N
        m2_blocks[b] = m2 / per_block
    return G_blocks, m2_blocks


def axis_profile(G):
    """G along the three lattice axes, averaged, for r = 0 .. L/2."""
    L = G.shape[-1]
    r = np.arange(L // 2 + 1)
    prof = (G[..., r, 0, 0] + G[..., 0, r, 0] + G[..., 0, 0, r]) / 3
    return r, prof


def mean_err(blocks, axis=0):
    n = blocks.shape[axis]
    return blocks.mean(axis), blocks.std(axis, ddof=1) / np.sqrt(n)


# -------------------------------------------------------- estimator 1 ---
def image_model(r, A, two_delta, L):
    """Power law plus nearest periodic image along the axis."""
    return A * (r ** -two_delta + (L - r) ** -two_delta)


def fit_direct(L, r, G, dG, r_min=3):
    sel = r >= r_min
    f = lambda rr, A, td: image_model(rr, A, td, L)
    popt, pcov = curve_fit(f, r[sel], G[sel], p0=[1.0, 1.0],
                           sigma=dG[sel], absolute_sigma=True)
    two_delta, err = popt[1], np.sqrt(pcov[1, 1])
    return popt, two_delta / 2, err / 2


def effective_exponent(r, G):
    """Local 2*Delta_eff(r) = -d ln G / d ln r (midpoints)."""
    lr, lG = np.log(r[1:]), np.log(G[1:])
    return np.exp(0.5 * (lr[1:] + lr[:-1])), -np.diff(lG) / np.diff(lr)


# -------------------------------------------------------- estimator 2 ---
def fit_fss(Ls, y, dy, corrections=False):
    """Fit y = A L^(-2 Delta) (1 + B L^-omega)  (B = 0 unless corrections)."""
    Ls = np.asarray(Ls, float)
    if corrections:
        f = lambda L, A, d, B: A * L ** (-2 * d) * (1 + B * L ** -OMEGA)
        p0 = [1.0, 0.52, 0.0]
    else:
        f = lambda L, A, d: A * L ** (-2 * d)
        p0 = [1.0, 0.52]
    popt, pcov = curve_fit(f, Ls, y, p0=p0, sigma=dy, absolute_sigma=True)
    return popt, popt[1], np.sqrt(pcov[1, 1])


# --------------------------------------------------------------- run ---
def run(Ls=(8, 12, 16, 24, 32), n_meas=8000, n_eq=1000):
    results = {}
    for L in Ls:
        Gb, m2b = measure_G(L, n_meas=n_meas, n_eq=n_eq, seed=L)
        r, prof_b = axis_profile(Gb)
        G, dG = mean_err(prof_b)
        m2, dm2 = mean_err(m2b)
        results[L] = dict(r=r, G=G, dG=dG, m2=m2, dm2=dm2)
        print(f"L={L:3d}  G(L/2) = {G[-1]:.5f} ± {dG[-1]:.5f}   "
              f"<m^2> = {m2:.5f} ± {dm2:.5f}")

    Ls = np.array(sorted(results))
    GL2 = np.array([results[L]["G"][-1] for L in Ls])
    dGL2 = np.array([results[L]["dG"][-1] for L in Ls])
    M2 = np.array([results[L]["m2"] for L in Ls])
    dM2 = np.array([results[L]["dm2"] for L in Ls])

    print("\nDelta_sigma estimates   (bootstrap: 0.518149)")
    Lmax = Ls[-1]
    d = results[Lmax]
    popt_dir, ds, e = fit_direct(Lmax, d["r"].astype(float), d["G"], d["dG"])
    print(f"  direct fit of G(r), L={Lmax}:      {ds:.4f} ± {e:.4f}")
    popt_g, ds, e = fit_fss(Ls, GL2, dGL2)
    print(f"  FSS of G(L/2):                   {ds:.4f} ± {e:.4f}")
    popt_m, ds, e = fit_fss(Ls, M2, dM2)
    print(f"  FSS of <m^2>:                    {ds:.4f} ± {e:.4f}")
    if len(Ls) >= 4:
        _, ds, e = fit_fss(Ls, GL2, dGL2, corrections=True)
        print(f"  FSS of G(L/2) + L^-omega corr.:  {ds:.4f} ± {e:.4f}")

    plot(results, popt_dir, popt_g, Ls, GL2, dGL2)
    return results


def plot(results, popt_dir, popt_g, Ls, GL2, dGL2):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    Lmax = Ls[-1]

    for L, d in results.items():
        ax[0].errorbar(d["r"][1:], d["G"][1:], d["dG"][1:], fmt="o", ms=3,
                       label=f"L={L}")
    rr = np.linspace(1, Lmax / 2, 200)
    ax[0].plot(rr, image_model(rr, *popt_dir, Lmax), "k-", lw=1,
               label=f"fit L={Lmax} (with image)")
    ax[0].plot(rr, popt_dir[0] * rr ** (-2 * DELTA_SIGMA_BOOTSTRAP), "k--",
               lw=1, label=r"$r^{-2\Delta_\sigma}$, bootstrap")
    ax[0].set(xscale="log", yscale="log", xlabel="r", ylabel="G(r)",
              title="Critical two-point function")
    ax[0].legend(fontsize=8)

    for L, d in results.items():
        x, td = effective_exponent(d["r"].astype(float), d["G"])
        ax[1].plot(x, td / 2, "o-", ms=3, label=f"L={L}")
    ax[1].axhline(DELTA_SIGMA_BOOTSTRAP, c="k", ls="--", lw=1, label="bootstrap")
    ax[1].set(xscale="log", ylim=(0.3, 1.2), xlabel="r",
              ylabel=r"$\Delta_{\rm eff}(r) = -\frac{1}{2}\,d\ln G/d\ln r$",
              title="Local exponent (plateau only for 1 ≪ r ≪ L)")
    ax[1].legend(fontsize=8)

    ax[2].errorbar(Ls, GL2, dGL2, fmt="o", label="G(L/2)")
    LL = np.linspace(Ls[0], Ls[-1], 100)
    ax[2].plot(LL, popt_g[0] * LL ** (-2 * popt_g[1]), "k-", lw=1,
               label=rf"fit: $\Delta_\sigma$ = {popt_g[1]:.4f}")
    ax[2].set(xscale="log", yscale="log", xlabel="L", ylabel="G(L/2)",
              title="Finite-size scaling")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    plt.savefig('deltasigma.png')
    plt.show()


if __name__ == "__main__":
    run()