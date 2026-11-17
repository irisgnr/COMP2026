"""
Plot CAMELS N-body total-matter maps from the corners of (Omega_m, sigma_8)
parameter space, along with their power spectra and density PDFs.

Usage:
    python plot_camels_corners.py

LH parameters file should have
one row per simulation with Omega_m and sigma_8 in the first two columns.
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
MAPS_FILE = "Maps_Mtot_Nbody_IllustrisTNG_LH_z=0.00.npy"
PARAMS_FILE = "params_LH_Nbody_IllustrisTNG.txt"
BOX_SIZE = 25.0        # Mpc/h (L25n256 suite)
MAPS_PER_SIM = 15      # CMD stores 15 projected maps per simulation
PROJECTION = 0         # which of the 15 maps to display for each simulation


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_data(maps_file, params_file):
    # mmap_mode="r" reads maps from disk on demand instead of loading ~4 GB
    maps = np.load(maps_file, mmap_mode="r")
    params = np.loadtxt(params_file, comments="#")

    n_sims = maps.shape[0] // MAPS_PER_SIM
    if params.shape[0] != n_sims:
        raise ValueError(
            f"{maps.shape[0]} maps imply {n_sims} simulations, but the "
            f"parameter file has {params.shape[0]} rows. Check the files "
            f"and MAPS_PER_SIM."
        )
    print(f"Loaded {maps.shape[0]} maps of shape {maps.shape[1:]} "
          f"from {n_sims} simulations.")
    return maps, params


def get_sim_maps(maps, sim):
    """All projected maps belonging to one simulation, as float64."""
    start = sim * MAPS_PER_SIM
    return np.asarray(maps[start:start + MAPS_PER_SIM], dtype=np.float64)


# ----------------------------------------------------------------------------
# Choosing corner simulations
# ----------------------------------------------------------------------------
def find_corner_sims(params):
    """Find the simulation closest to each corner of (Omega_m, sigma_8) space,
    measuring distance after rescaling both parameters to [0, 1]."""
    om, s8 = params[:, 0], params[:, 1]
    om_n = (om - om.min()) / (om.max() - om.min())
    s8_n = (s8 - s8.min()) / (s8.max() - s8.min())

    # Ordered to match a 2x2 grid: sigma_8 increases upward,
    # Omega_m increases to the right.
    corners = {
        "low $\\Omega_m$, high $\\sigma_8$": (0, 1),
        "high $\\Omega_m$, high $\\sigma_8$": (1, 1),
        "low $\\Omega_m$, low $\\sigma_8$": (0, 0),
        "high $\\Omega_m$, low $\\sigma_8$": (1, 0),
    }
    chosen = {}
    for label, (a, b) in corners.items():
        sim = int(np.argmin((om_n - a) ** 2 + (s8_n - b) ** 2))
        chosen[label] = sim
        s8_combo = s8[sim] * np.sqrt(om[sim] / 0.3)
        print(f"{label:<35} sim {sim:4d}: Omega_m = {om[sim]:.3f}, "
              f"sigma_8 = {s8[sim]:.3f}, S_8 = {s8_combo:.3f}")
    return chosen


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def power_spectrum_2d(field, box_size, n_bins=25):
    """Azimuthally averaged power spectrum of the overdensity of a 2D map.
    Returns k in h/Mpc and P(k) in (Mpc/h)^2."""
    n = field.shape[0]
    delta = field / field.mean() - 1.0
    delta_k = np.fft.fftn(delta)
    pk_2d = np.abs(delta_k) ** 2 * box_size ** 2 / n ** 4

    kfreq = 2 * np.pi * np.fft.fftfreq(n, d=box_size / n)
    kx, ky = np.meshgrid(kfreq, kfreq, indexing="ij")
    k = np.sqrt(kx ** 2 + ky ** 2).ravel()
    pk_2d = pk_2d.ravel()

    k_fund = 2 * np.pi / box_size
    k_nyq = np.pi * n / box_size
    edges = np.logspace(np.log10(k_fund), np.log10(k_nyq), n_bins + 1)
    which = np.digitize(k, edges)

    k_out, pk_out = [], []
    for i in range(1, len(edges)):
        mask = which == i
        if mask.any():
            k_out.append(np.sqrt(edges[i - 1] * edges[i]))
            pk_out.append(pk_2d[mask].mean())
    return np.array(k_out), np.array(pk_out)


def log_overdensity(field):
    """log10(rho / mean rho): removes the mean-density shortcut and
    compresses the huge dynamic range of the maps."""
    return np.log10(field / field.mean())


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_maps(maps, params, chosen, outfile="corner_maps.png"):
    fields = {label: np.asarray(maps[sim * MAPS_PER_SIM + PROJECTION],
                                dtype=np.float64)
              for label, sim in chosen.items()}
    logs = {label: log_overdensity(f) for label, f in fields.items()}

    # Shared color scale so the contrast differences are visible
    all_vals = np.concatenate([v.ravel() for v in logs.values()])
    vmin, vmax = np.percentile(all_vals, [1, 99.5])

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    for ax, (label, sim) in zip(axes.ravel(), chosen.items()):
        im = ax.imshow(logs[label].T, origin="lower", cmap="magma",
                       vmin=vmin, vmax=vmax,
                       extent=[0, BOX_SIZE, 0, BOX_SIZE])
        om, s8 = params[sim, 0], params[sim, 1]
        ax.set_title(f"{label}\n$\\Omega_m$ = {om:.2f}, "
                     f"$\\sigma_8$ = {s8:.2f}  (sim {sim})\n"
                     f"raw mean = {fields[label].mean():.2e}", fontsize=10)
        ax.set_xlabel("x [Mpc/h]")
        ax.set_ylabel("y [Mpc/h]")
    fig.colorbar(im, ax=axes, shrink=0.8,
                 label=r"$\log_{10}(\rho / \bar{\rho})$")
    fig.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")


def plot_statistics(maps, params, chosen, outfile="corner_statistics.png"):
    """Power spectrum and PDF, averaged over all projections of each sim
    to reduce noise from any single map."""
    fig, (ax_pk, ax_pdf) = plt.subplots(1, 2, figsize=(13, 5),
                                        constrained_layout=True)
    pdf_edges = np.linspace(-1.5, 3.0, 80)
    pdf_centers = 0.5 * (pdf_edges[1:] + pdf_edges[:-1])

    for label, sim in chosen.items():
        sim_maps = get_sim_maps(maps, sim)
        spectra = [power_spectrum_2d(m, BOX_SIZE) for m in sim_maps]
        k = spectra[0][0]
        pk_mean = np.mean([pk for _, pk in spectra], axis=0)

        log_vals = np.concatenate([log_overdensity(m).ravel()
                                   for m in sim_maps])
        pdf, _ = np.histogram(log_vals, bins=pdf_edges, density=True)

        legend = (f"{label}: $\\Omega_m$={params[sim, 0]:.2f}, "
                  f"$\\sigma_8$={params[sim, 1]:.2f}")
        ax_pk.loglog(k, pk_mean, label=legend)
        ax_pdf.semilogy(pdf_centers, pdf, label=legend)

    ax_pk.set_xlabel(r"$k$ [h/Mpc]")
    ax_pk.set_ylabel(r"$P(k)$ [(Mpc/h)$^2$]")
    ax_pk.set_title(f"2D power spectrum (mean of {MAPS_PER_SIM} projections)")
    ax_pk.legend(fontsize=8)

    ax_pdf.set_xlabel(r"$\log_{10}(\rho / \bar{\rho})$")
    ax_pdf.set_ylabel("probability density")
    ax_pdf.set_title("Pixel density PDF")
    ax_pdf.legend(fontsize=8)

    fig.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")


if __name__ == "__main__":
    maps, params = load_data(MAPS_FILE, PARAMS_FILE)
    chosen = find_corner_sims(params)
    plot_maps(maps, params, chosen)
    plot_statistics(maps, params, chosen)
    plt.show()