"""Quick MCLMC vs HMC comparison on lattice phi^4: susceptibility and
ESS per gradient (from the autocorrelation of phibar) vs lambda."""
import numpy as np, matplotlib.pyplot as plt
from mclmc import MCLMC, make_phi4, integrated_autocorr_time

Lsize, n = 32, 10000
lams = [6.0, 5.0, 4.6, 4.25, 4.0, 3.6]          # annealed from high to low
rng = np.random.default_rng(0)

def hmc(f, x, n, num_steps=20, eps=0.02, warmup=500):
    lp, g = f(x); out, grads = [], 0
    for t in range(warmup + n):
        p = rng.standard_normal(x.size); H0 = -lp + 0.5 * p @ p
        xn, gn, pn = x, g, p + 0.5 * eps * g
        with np.errstate(all="ignore"):   # a too-large step can blow up phi^4
            for i in range(num_steps):
                xn = xn + eps * pn; lpn, gn = f(xn)
                if not np.isfinite(lpn): break
                pn = pn + (eps if i < num_steps - 1 else 0.5 * eps) * gn
            H1 = -lpn + 0.5 * pn @ pn
            acc = min(1.0, np.exp(H0 - H1)) if np.isfinite(H1) else 0.0  # diverged: reject
        if rng.random() < acc: x, lp, g = xn, lpn, gn
        if t < warmup: eps *= np.exp((acc - 0.8) / (t + 1) ** 0.6)  # tune step
        else: out.append(x.mean()); grads += num_steps
    return np.array(out), grads, x

res = {"MCLMC": [], "HMC": []}
x_m = x_h = rng.standard_normal(Lsize**2)
for lam in lams:
    f = make_phi4(Lsize, lam)
    s = MCLMC(f, x_m, seed=1); s.tune(verbose=False)
    pb, _ = s.sample(n, observable=np.mean); x_m = s.x
    res["MCLMC"].append((Lsize**2 * pb.var(), n / integrated_autocorr_time(pb) / s.n_grads))
    pb, grads, x_h = hmc(f, x_h, n // 5)
    res["HMC"].append((Lsize**2 * pb.var(), len(pb) / integrated_autocorr_time(pb) / grads))
    print(lam, res["MCLMC"][-1], res["HMC"][-1])

fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
for name, m in [("MCLMC", "o-"), ("HMC", "s--")]:
    chi, ess = np.array(res[name]).T
    a1.plot(lams, chi, m, label=name); a2.plot(lams, ess, m, label=name)
a1.axvline(4.25, color="gray", lw=0.8); a2.axvline(4.25, color="gray", lw=0.8)
a1.set_ylabel(r"$\chi = L^2\,\mathrm{Var}(\bar\phi)$"); a1.legend()
a2.set_ylabel(r"ESS($\bar\phi$) / gradient"); a2.set_yscale("log")
a2.set_xlabel(r"$\lambda$  ($m^2=-4$, line at $\lambda_c\approx4.25$)")
a1.set_title(f"Lattice $\\phi^4$, L = {Lsize}")
fig.tight_layout(); fig.savefig("comparison_L32.png", dpi=150)