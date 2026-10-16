"""
Microcanonical Langevin Monte Carlo (MCLMC) in plain NumPy, following
Robnik & Seljak, "Fluctuation without dissipation: Microcanonical Langevin
Monte Carlo" (arXiv:2303.18221), applied to 2D lattice phi^4 theory.

Interface: like HMC/NUTS, the sampler only needs a function
    logdensity_and_grad(x) -> (log p(x), grad log p(x))
for a flat parameter vector x. For lattice field theory there is no separate
"likelihood": the target is p(phi) ∝ exp(-S[phi]), so log p = -S.

Algorithm follows Robnik, De Luca, Silverstein & Seljak (MCHMC) and
Robnik & Seljak (MCLMC); update formulas match the BlackJAX implementation.
The step-size / L tuning here is a simplified version of the paper's scheme.
"""
import numpy as np
import jax, jax.numpy as jnp
from numpyro.infer import MCMC, NUTS, HMC

LAMBDA_MN = 0.1931833275037836  # minimal-norm (Omelyan) integrator coefficient


# ----------------------------------------------------------------------------
# Core dynamics
# ----------------------------------------------------------------------------
def velocity_update(u, g, eps):
    """Exact isokinetic (ESH) velocity update over time eps.

    u : unit velocity, g : grad log p at current position.
    Returns new unit velocity and the kinetic-energy change.
    """
    d = u.size
    gnorm = np.linalg.norm(g)
    if gnorm == 0.0:
        return u, 0.0
    e = g / gnorm
    ue = u @ e
    delta = eps * gnorm / (d - 1)
    zeta = np.exp(-delta)
    u_new = e * (1 - zeta) * (1 + zeta + ue * (1 - zeta)) + 2 * zeta * u
    u_new /= np.linalg.norm(u_new)
    dK = (delta - np.log(2) + np.log(1 + ue + (1 - ue) * zeta**2)) * (d - 1)
    return u_new, dK


def integrator_step(f, x, u, logp, g, eps, scheme="minimal_norm"):
    """One deterministic step. Returns x, u, logp, g, energy_change, n_grads."""
    logp0 = logp
    dK = 0.0
    if scheme == "leapfrog":
        u, k = velocity_update(u, g, 0.5 * eps); dK += k
        x = x + eps * u
        logp, g = f(x)
        u, k = velocity_update(u, g, 0.5 * eps); dK += k
        n_grads = 1
    elif scheme == "minimal_norm":
        u, k = velocity_update(u, g, LAMBDA_MN * eps); dK += k
        x = x + 0.5 * eps * u
        logp, g = f(x)
        u, k = velocity_update(u, g, (1 - 2 * LAMBDA_MN) * eps); dK += k
        x = x + 0.5 * eps * u
        logp, g = f(x)
        u, k = velocity_update(u, g, LAMBDA_MN * eps); dK += k
        n_grads = 2
    else:
        raise ValueError(scheme)
    energy_change = dK - (logp - logp0)
    return x, u, logp, g, energy_change, n_grads


def partial_refresh(u, eps, L, rng):
    """Langevin-style partial momentum refresh; L = decoherence length."""
    d = u.size
    c1 = np.exp(-eps / L)
    c2 = np.sqrt(1 - c1**2)
    u = c1 * u + c2 * rng.standard_normal(d) / np.sqrt(d)
    return u / np.linalg.norm(u)


def random_unit_vector(d, rng):
    u = rng.standard_normal(d)
    return u / np.linalg.norm(u)


# ----------------------------------------------------------------------------
# Sampler with simplified tuning
# ----------------------------------------------------------------------------
class MCLMC:
    def __init__(self, logdensity_and_grad, x0, scheme="minimal_norm",
                 target_energy_var=0.02, seed=0):
        self.f = logdensity_and_grad
        self.x = np.array(x0, dtype=float)
        self.d = self.x.size
        self.scheme = scheme
        self.target = target_energy_var      # desired <dE^2>/d (paper: 0.02)
        self.rng = np.random.default_rng(seed)
        self.u = random_unit_vector(self.d, self.rng)
        self.logp, self.g = self.f(self.x)
        self.eps = 0.25 * np.sqrt(self.d)    # initial guesses (as in BlackJAX)
        self.nu = np.sqrt(self.d)
        self.n_grads = 0

    def step(self):
        x, u, logp, g, dE, ng = integrator_step(
            self.f, self.x, self.u, self.logp, self.g, self.eps, self.scheme)
        self.n_grads += ng
        if not np.isfinite(dE) or not np.all(np.isfinite(x)):
            return None                      # diverged; caller handles
        self.x, self.logp, self.g = x, logp, g
        self.u = partial_refresh(u, self.eps, self.nu, self.rng)
        return dE

    def tune(self, n_windows=12, window=300, n_L=2000, verbose=True):
        """Stage 1: find eps with <dE^2>/d ~ target. The energy error is a
        very steep function of eps (roughly eps^6 to eps^12), so we do a
        damped Newton/secant iteration in log-log space on windowed averages.
        Stage 2: set L from the sqrt of summed position variances
        (heuristic from Robnik et al. 2212.08549; the paper then refines
        it using the ESS)."""
        for _ in range(window):              # burn-in at initial eps
            if self.step() is None:
                self.eps *= 0.5
                self.u = random_unit_vector(self.d, self.rng)
        hist = []
        for _ in range(n_windows):
            dEs, diverged = [], False
            for _ in range(window):
                dE = self.step()
                if dE is None:
                    diverged = True
                    break
                dEs.append(dE)
            if diverged or not dEs:
                self.eps *= 0.5
                self.u = random_unit_vector(self.d, self.rng)
                continue
            v = max(np.mean(np.square(dEs)) / self.d, 1e-30)
            hist.append((np.log(self.eps), np.log(v)))
            slope = 6.0
            if len(hist) >= 2:
                (a0, b0), (a1, b1) = hist[-2], hist[-1]
                if abs(a1 - a0) > 1e-3:
                    slope = np.clip((b1 - b0) / (a1 - a0), 4.0, 14.0)
            step = np.clip(np.log(self.target / v) / slope, -0.7, 0.7)
            self.eps *= np.exp(step)
        mean = np.zeros(self.d); m2 = np.zeros(self.d)
        for i in range(1, n_L + 1):
            self.step()
            delta = self.x - mean
            mean += delta / i
            m2 += delta * (self.x - mean)
        self.nu = np.sqrt(np.sum(m2 / (n_L - 1)))
        if verbose:
            print(f"tuned: eps = {self.eps:.4f}, L = {self.nu:.4f}")

    def sample(self, n_steps, observable=lambda x: x, thin=1):
        obs, dEs = [], []
        for i in range(n_steps):
            dE = self.step()
            if dE is None:
                raise RuntimeError("divergence during sampling; retune")
            dEs.append(dE)
            if i % thin == 0:
                obs.append(observable(self.x))
        dEs = np.array(dEs)
        return np.array(obs), {"energy_var_per_dim": np.mean(dEs**2) / self.d,
                               "n_grads": self.n_grads}


# ----------------------------------------------------------------------------
# Lattice phi^4 in 2D, Robnik & Seljak (arXiv:2303.18221) Eq. (20):
#   S = sum_ij 2 phi_ij (2 phi_ij - phi_{i+1,j} - phi_{i,j+1}) + m2 phi_ij^2 + lam phi_ij^4
# Paper fixes m2 = -4 (diagonal phi^2 terms cancel) and scans lam; lam_c ~ 4.25.
# For validation use lam = 0 with m2 > 0, which is a Gaussian field.
# ----------------------------------------------------------------------------
def make_phi4(Lsize, lam, m2=-4.0):
    shape = (Lsize, Lsize)

    def logdensity_and_grad(xflat):
        phi = xflat.reshape(shape)
        fwd = np.roll(phi, -1, 0) + np.roll(phi, -1, 1)
        nbr = fwd + np.roll(phi, 1, 0) + np.roll(phi, 1, 1)
        S = np.sum(2 * phi * (2 * phi - fwd) + m2 * phi**2 + lam * phi**4)
        dS = 8 * phi - 2 * nbr + 2 * m2 * phi + 4 * lam * phi**3
        return -S, -dS.ravel()

    return logdensity_and_grad


def free_field_power_exact(Lsize, m2):
    """Exact <|phi~_k|^2> for lam = 0, m2 > 0 (k-space, unitary FFT)."""
    k = 2 * np.pi * np.arange(Lsize) / Lsize
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return 1.0 / (8 + 2 * m2 - 4 * np.cos(kx) - 4 * np.cos(ky))


def power_spectrum(xflat, Lsize):
    """|phi~_kl|^2 with the paper's 1/sqrt(L^2) normalisation."""
    return np.abs(np.fft.fft2(xflat.reshape(Lsize, Lsize)))**2 / Lsize**2


def susceptibility(phibar_samples, Lsize):
    """Paper's estimator chi = L^2 < (phibar - <phibar>)^2 >."""
    return Lsize**2 * np.var(phibar_samples)


def rescaled(lam, chi, Lsize, lam_c=4.25, nu=1.0, gamma=7 / 4):
    """Finite-size-scaling axes used in Fig. 1 (Ising exponents).
    Check the exact convention against Gerdes et al. (arXiv:2207.00283)."""
    lam_bar = Lsize**(1 / nu) * (lam - lam_c) / lam_c
    chi_bar = chi * Lsize**(-gamma / nu)
    return lam_bar, chi_bar


# ----------------------------------------------------------------------------
# Paper's efficiency metric: b2^2 = mean_k [(<|phi~_k|^2>_sampler - truth)/truth]^2,
# ESS = 2 / b2^2.  Report 200 / (gradient calls when b2 first reaches 0.1),
# with tuning gradients included.
# ----------------------------------------------------------------------------
def ess_per_grad(sampler, truth_power, Lsize, max_steps=200000, b2_target=0.1):
    grads_at_start = 0                       # tuning cost already in sampler.n_grads
    running = np.zeros_like(truth_power)
    for n in range(1, max_steps + 1):
        if sampler.step() is None:
            raise RuntimeError("divergence; retune")
        running += (power_spectrum(sampler.x, Lsize) - running) / n
        b2 = np.sqrt(np.mean(((running - truth_power) / truth_power) ** 2))
        if b2 < b2_target and n > 10:
            return (2 / b2_target**2) / (sampler.n_grads + grads_at_start), n
    return np.nan, max_steps


# ----------------------------------------------------------------------------
# Autocorrelation analysis (Sokal automatic windowing)
# ----------------------------------------------------------------------------
def integrated_autocorr_time(series, c=5.0):
    x = np.asarray(series, float) - np.mean(series)
    n = len(x)
    fx = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(fx * np.conj(fx))[:n]
    acf /= acf[0]
    tau = 1.0
    for w in range(1, n):
        tau = 1 + 2 * np.sum(acf[1:w + 1])
        if w >= c * tau:
            break
    return tau


def mean_and_error(series):
    tau = integrated_autocorr_time(series)
    n = len(series)
    return np.mean(series), np.std(series) * np.sqrt(tau / n), tau



# ----------------------------------------------------------------------------
# HMC
# ----------------------------------------------------------------------------

def make_action(L, lam, m2=-4.0):
    def S(x):
        phi = x.reshape(L, L)
        fwd = jnp.roll(phi, -1, 0) + jnp.roll(phi, -1, 1)
        return jnp.sum(2 * phi * (2 * phi - fwd) + m2 * phi**2 + lam * phi**4)
    return S


# ----------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # HMC

    print("=== HMC Comparison: Gaussian field (lam = 4.5, m2 = -4.0) ===")

    L, lam = 16, 4.5
    kernel = NUTS(potential_fn=make_action(L, lam))   # or HMC(potential_fn=..., num_steps=30)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=5000)
    x0 = jax.random.normal(jax.random.PRNGKey(1), (L * L,))
    mcmc.run(jax.random.PRNGKey(0), init_params=x0, extra_fields=("num_steps",))

    phi = mcmc.get_samples()                               # shape (5000, L*L)
    n_grads = mcmc.get_extra_fields()["num_steps"].sum()   # gradient evals while sampling


    # MCLMC


    Lsize = 16
    d = Lsize**2

    print("=== Validation: Gaussian field (lam = 0, m2 = 0.5) ===")
    m2 = 0.5
    f = make_phi4(Lsize, lam=0.0, m2=m2)
    truth = free_field_power_exact(Lsize, m2)
    rng = np.random.default_rng(0)
    # The paper's target (0.02) leaves >10% bias in the high-k modes of this
    # Gaussian test, so b2 never reaches 0.1; a tighter target fixes it.
    for target in [0.02, 5e-4]:
        s = MCLMC(f, x0=rng.standard_normal(d), seed=1, target_energy_var=target)
        s.tune(verbose=False)
        e, n = ess_per_grad(s, truth, Lsize, max_steps=50000)
        print(f"target <dE^2>/d = {target:g}: eps = {s.eps:.2f}, "
              f"ESS per gradient = {e:.4f}  (steps to b2 = 0.1: {n})")

    print("\n=== Annealed scan in lam (m2 = -4), as in Sec. VII ===")
    x = rng.standard_normal(d)               # random start at highest lam
    for lam in [6.0, 5.0, 4.5, 4.25, 4.0, 3.5]:
        f = make_phi4(Lsize, lam=lam)
        s = MCLMC(f, x0=x, seed=int(lam * 100))
        s.tune(verbose=False)
        phibar, info = s.sample(20000, observable=lambda x: np.mean(x))
        x = s.x                              # anneal: carry state to next lam
        chi = susceptibility(phibar, Lsize)
        lb, cb = rescaled(lam, chi, Lsize)
        print(f"lam = {lam:5.2f}  chi = {chi:8.3f}  lam_bar = {lb:6.2f}  "
              f"chi_bar = {cb:.4f}  eps = {s.eps:.3f}  L = {s.nu:.2f}  "
              f"<dE^2>/d = {info['energy_var_per_dim']:.3f}")


    
    