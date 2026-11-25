"""
Fit LCDM cosmological parameters to Planck 2018 CMB power spectra with a genetic algorithm.

Theory spectra: CAMB
Likelihood:     planck_lite_py (plik_lite TT+TE+EE, ell >= 30) + Gaussian prior on tau
Optimizer:      real-valued GA (tournament selection, BLX-alpha crossover,
                Gaussian mutation with decaying width, elitism)

Run from anywhere:  python cosmo_optimization.py
"""
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import camb

from planck_lite_py import PlanckLitePy

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------
PARAM_NAMES = ["H0", "ombh2", "omch2", "tau", "logA", "ns"]   # logA = ln(1e10 As)
LOWER = np.array([60.0, 0.019, 0.10, 0.02, 2.9, 0.92]) #np.array([50.0, 0.015, 0.05, 0.01, 2.0, 0.9])
UPPER = np.array([80.0, 0.025, 0.14, 0.10, 3.2, 1.00]) #np.array([90.0, 0.03, 0.2, 0.12, 4.0, 1.10])
N_PARAMS = len(PARAM_NAMES)

# Planck 2018 TT,TE,EE+lowE best fit (Planck 2018 VI, Table 2; H0 is derived there)
PLANCK_BESTFIT = np.array([67.32, 0.02238, 0.1201, 0.0543, 3.0448, 0.9660])

# Gaussian prior on tau, standing in for the low-ell EE likelihood that plik_lite lacks.
# High-ell data alone only constrains As*exp(-2 tau), so without this tau and As are degenerate.
TAU_PRIOR_MEAN, TAU_PRIOR_SIGMA = 0.054, 0.007

LMAX = 2508          # highest ell plik_lite uses
ELLMIN = 2           # ell of the first element we pass to loglike

# ---------------------------------------------------------------------------
# GA settings
# ---------------------------------------------------------------------------
POP_SIZE = 10
N_GENERATIONS = 10
N_ELITE = 2
TOURNAMENT_K = 3
CROSSOVER_PROB = 0.9
BLX_ALPHA = 0.3
MUT_PROB = 0.3           # per-gene probability of mutation
MUT_SIGMA_START = 0.10   # in unit-cube coordinates
MUT_SIGMA_END = 0.01
SEED = 42

camb.set_feedback_level(0)
like = PlanckLitePy(data_directory=DATA_DIR, year=2018, spectra="TTTEEE", use_low_ell_bins=False)


# ---------------------------------------------------------------------------
# Physics: parameters -> spectra -> log posterior
# ---------------------------------------------------------------------------
def to_physical(u):
    """Map unit-cube coordinates u in [0,1]^N to physical parameters."""
    return LOWER + u * (UPPER - LOWER)


def camb_spectra(theta):
    """Return ell, D_TT, D_TE, D_EE [muK^2] from CAMB for parameters theta."""
    H0, ombh2, omch2, tau, logA, ns = theta
    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0,
                           tau=tau, As=np.exp(logA) * 1e-10, ns=ns,
                           lmax=LMAX, lens_potential_accuracy=0)
    pars.WantTransfer = False
    results = camb.get_results(pars)
    tot = results.get_cmb_power_spectra(pars, CMB_unit="muK", lmax=LMAX)["total"]
    ell = np.arange(tot.shape[0])
    return ell, tot[:, 0], tot[:, 3], tot[:, 1]      # CAMB column order: TT, EE, BB, TE


def log_likelihood(theta):
    _, DTT, DTE, DEE = camb_spectra(theta)
    return like.loglike(DTT[ELLMIN:], DTE[ELLMIN:], DEE[ELLMIN:], ellmin=ELLMIN)


def log_prior(theta):
    if np.any(theta < LOWER) or np.any(theta > UPPER):
        return -np.inf
    tau = theta[PARAM_NAMES.index("tau")]
    return -0.5 * ((tau - TAU_PRIOR_MEAN) / TAU_PRIOR_SIGMA) ** 2


def fitness(theta):
    """Log posterior. CAMB failures (unphysical corners of parameter space) get -inf."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    try:
        return lp + log_likelihood(theta)
    except Exception as err:                       # CAMBError etc.
        print(f"  CAMB failed at {np.round(theta, 5)}: {err}")
        return -np.inf


# ---------------------------------------------------------------------------
# Genetic algorithm operators (all in unit-cube coordinates)
# ---------------------------------------------------------------------------
def tournament_select(pop, fit, rng):
    idx = rng.choice(len(pop), size=TOURNAMENT_K, replace=False)
    return pop[idx[np.argmax(fit[idx])]]


def blx_crossover(p1, p2, rng):
    """Blend crossover: each child gene drawn uniformly from the parents' interval widened by alpha."""
    lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
    span = hi - lo
    c1 = rng.uniform(lo - BLX_ALPHA * span, hi + BLX_ALPHA * span)
    c2 = rng.uniform(lo - BLX_ALPHA * span, hi + BLX_ALPHA * span)
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)


def mutate(child, sigma, rng):
    mask = rng.random(N_PARAMS) < MUT_PROB
    child = child + mask * rng.normal(0, sigma, N_PARAMS)
    # reflect off the walls so mutated genes don't pile up exactly on the bounds
    child = np.abs(child)
    child = np.where(child > 1, 2 - child, child)
    return np.clip(child, 0, 1)


def evaluate(pop):
    return np.array([fitness(to_physical(u)) for u in pop])


def run_ga():
    rng = np.random.default_rng(SEED)
    pop = rng.random((POP_SIZE, N_PARAMS))

    t0 = time.time()
    fit = evaluate(pop)
    dt = time.time() - t0
    print(f"Generation 0 took {dt:.1f} s -> estimated total {dt * (N_GENERATIONS + 1) / 60:.1f} min")

    history = {"best_fit": [], "mean_fit": [], "best_theta": []}

    def record(gen):
        finite = fit[np.isfinite(fit)]
        i = np.argmax(fit)
        history["best_fit"].append(fit[i])
        history["mean_fit"].append(finite.mean() if finite.size else -np.inf)
        history["best_theta"].append(to_physical(pop[i]))
        print(f"gen {gen:3d}  best logpost = {fit[i]:10.2f}  "
              + "  ".join(f"{n}={v:.5g}" for n, v in zip(PARAM_NAMES, to_physical(pop[i]))))

    record(0)

    for gen in range(1, N_GENERATIONS + 1):
        # mutation width decays geometrically: explore early, refine late
        frac = (gen - 1) / max(N_GENERATIONS - 1, 1)
        sigma = MUT_SIGMA_START * (MUT_SIGMA_END / MUT_SIGMA_START) ** frac

        order = np.argsort(fit)[::-1]
        elites, elite_fit = pop[order[:N_ELITE]].copy(), fit[order[:N_ELITE]].copy()

        children = []
        while len(children) < POP_SIZE - N_ELITE:
            p1 = tournament_select(pop, fit, rng)
            p2 = tournament_select(pop, fit, rng)
            if rng.random() < CROSSOVER_PROB:
                c1, c2 = blx_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            children.append(mutate(c1, sigma, rng))
            children.append(mutate(c2, sigma, rng))
        children = np.array(children[:POP_SIZE - N_ELITE])

        pop = np.vstack([elites, children])
        fit = np.concatenate([elite_fit, evaluate(children)])   # elites are not re-evaluated
        record(gen)

    print(f"GA finished in {(time.time() - t0) / 60:.1f} min")
    i = np.argmax(fit)
    return to_physical(pop[i]), fit[i], {k: np.array(v) for k, v in history.items()}


# ---------------------------------------------------------------------------
# Comparison with Planck and plots
# ---------------------------------------------------------------------------
def compare_and_plot(best_theta, best_fit, history):
    planck_fit = fitness(PLANCK_BESTFIT)

    print("\n{:>8s} {:>12s} {:>12s} {:>10s}".format("param", "GA", "Planck2018", "diff %"))
    for n, g, p in zip(PARAM_NAMES, best_theta, PLANCK_BESTFIT):
        print(f"{n:>8s} {g:12.5g} {p:12.5g} {100 * (g - p) / p:10.2f}")
    print(f"\nlog posterior  GA: {best_fit:.2f}   Planck best fit: {planck_fit:.2f}")
    print(f"Delta chi^2 (GA - Planck) = {-2 * (best_fit - planck_fit):.2f}  "
          "(negative or ~0 means the GA found an equally good or better fit)")

    np.savez(os.path.join(HERE, "ga_results.npz"), best_theta=best_theta, best_fit=best_fit,
             planck_theta=PLANCK_BESTFIT, planck_fit=planck_fit, param_names=PARAM_NAMES,
             **{f"history_{k}": v for k, v in history.items()})

    # --- spectra: data vs GA fit vs Planck's theory curve -----------------
    ell, DTT, DTE, DEE = camb_spectra(best_theta)
    files = {
        "TT": ("COM_PowerSpect_CMB-TT-binned_R3.01.txt", DTT),
        "TE": ("COM_PowerSpect_CMB-TE-binned_R3.02.txt", DTE),
        "EE": ("COM_PowerSpect_CMB-EE-binned_R3.02.txt", DEE),
    }

    # best individual from a few earlier generations, to show how the fit improved
    n_gen = len(history["best_theta"]) - 1
    snapshot_gens = sorted({0, n_gen // 8, n_gen // 4, n_gen // 2} - {n_gen})
    snapshots = []
    for g in snapshot_gens:
        _, sTT, sTE, sEE = camb_spectra(history["best_theta"][g])
        snapshots.append((g, {"TT": sTT, "TE": sTE, "EE": sEE}))
    # light -> dark blue as generations progress; final fit is solid C0 on top
    snap_colors = plt.cm.Blues(np.linspace(0.3, 0.7, max(len(snapshots), 1)))

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for ax, (name, (fname, D_ga)) in zip(axes, files.items()):
        data = np.loadtxt(os.path.join(DATA_DIR, fname))
        for (g, spec), c in zip(snapshots, snap_colors):
            ax.plot(ell[2:], spec[name][2:], color=c, alpha=0.5, lw=1, label=f"GA best, gen {g}")
        ax.errorbar(data[:, 0], data[:, 1], yerr=[data[:, 2], data[:, 3]],
                    fmt="o", ms=3, color="k", label="Planck 2018 data")
        ax.plot(data[:, 0], data[:, 4], "s", ms=4, mfc="none", color="C1", label="Planck best fit (binned)")
        ax.plot(ell[2:], D_ga[2:], color="C0", lw=2, label=f"GA best fit, gen {n_gen}")
        ax.set_ylabel(rf"$D_\ell^{{{name}}}\ [\mu K^2]$")
    axes[0].legend(fontsize=8)          # same entries on every panel, so one legend is enough
    axes[-1].set_xlabel(r"$\ell$")
    axes[-1].set_xlim(2, LMAX)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "ga_spectra.png"), dpi=150)

    # --- convergence ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    gens = np.arange(len(history["best_fit"]))
    ax.plot(gens, history["best_fit"], label="best")
    ax.plot(gens, history["mean_fit"], label="population mean")
    ax.axhline(planck_fit, color="k", ls="--", label="Planck best fit")
    ax.set_xlabel("generation")
    ax.set_ylabel("log posterior")
    ax.set_ylim(planck_fit - 200, planck_fit + 10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "ga_convergence.png"), dpi=150)

    plt.show()


if __name__ == "__main__":
    best_theta, best_fit, history = run_ga()
    compare_and_plot(best_theta, best_fit, history)
