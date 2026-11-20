#!/usr/bin/env python3
"""
possible next steps: see if mutants behave differently,
make a simulation of a wiggling line that behaves the same way
simulate motion with some different condition
https://zenodo.org/records/1031222, https://zenodo.org/records/1014381, https://zenodo.org/records/1022286, https://zenodo.org/records/1007437, https://zenodo.org/records/1006406

Eigenworm analysis of C. elegans posture from OpenWorm .wcon.zip files.

Pipeline:
  1. Load skeletons (midline x, y points per frame) from WCON files.
  2. Resample each skeleton to a fixed number of points by arc length.
  3. Convert to tangent angles along the body, subtract each frame's mean
     angle (removes overall rotation), so only body *shape* remains.
  4. Pool frames from all worms and run PCA (eigendecomposition of the
     covariance matrix). Eigenvectors = eigenworms.
  5. Project every frame onto the top modes to get mode amplitudes a_k(t).

Usage:
  python eigenworms.py path/to/*.wcon.zip --out results
  python eigenworms.py data_folder/ --n-points 49 --n-modes 4

Requires: numpy, matplotlib
"""

import argparse
import glob
import json
import os
import zipfile

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- loading

def read_wcon(path):
    """Return the parsed WCON JSON from a .wcon, .json, or .wcon.zip file."""
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith((".wcon", ".json"))]
            if not names:
                raise ValueError(f"No .wcon/.json file inside {path}")
            with z.open(names[0]) as f:
                return json.load(f)
    with open(path) as f:
        return json.load(f)


def find_key(obj, key):
    """Search nested dicts/lists for the first value stored under `key`."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = find_key(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_key(v, key)
            if found is not None:
                return found
    return None


def _as_frames(values, n_t):
    """Normalize WCON x/y (one frame or a list of frames) to a list of frames."""
    if n_t == 1 and values and not isinstance(values[0], (list, type(None))):
        return [values]
    return values


def _offsets(chunk, key, n_t):
    """WCON optional origin offsets (ox/oy): scalar, per-frame list, or absent."""
    o = chunk.get(key, 0.0)
    if isinstance(o, list):
        return [0.0 if v is None else v for v in o]
    return [0.0 if o is None else o] * n_t


def load_tracks(wcon):
    """
    Return {worm_id: (t, frames)} where frames is a list of (x, y) arrays,
    or None for frames with no skeleton. WCON data may be split into chunks.
    """
    data = wcon["data"]
    if isinstance(data, dict):
        data = [data]

    raw = {}
    for chunk in data:
        wid = str(chunk.get("id", "0"))
        t = chunk["t"]
        t = t if isinstance(t, list) else [t]
        n_t = len(t)
        xs = _as_frames(chunk["x"], n_t)
        ys = _as_frames(chunk["y"], n_t)
        ox = _offsets(chunk, "ox", n_t)
        oy = _offsets(chunk, "oy", n_t)
        for i in range(n_t):
            xi, yi = xs[i], ys[i]
            if xi is None or yi is None or len(xi) == 0:
                pts = None
            else:
                x = np.array([np.nan if v is None else v for v in xi], float) + ox[i]
                y = np.array([np.nan if v is None else v for v in yi], float) + oy[i]
                pts = (x, y)
            raw.setdefault(wid, []).append((t[i], pts))

    tracks = {}
    for wid, rows in raw.items():
        rows.sort(key=lambda r: r[0])
        tracks[wid] = (np.array([r[0] for r in rows]), [r[1] for r in rows])
    return tracks


# ---------------------------------------------------------------- posture

def tangent_angles(x, y, n_points):
    """
    Resample a skeleton to n_points by arc length and return n_points - 1
    tangent angles with the frame's mean angle removed. NaN row if invalid.
    """
    out = np.full(n_points - 1, np.nan)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3 or not good.all():
        return out  # skip partial skeletons rather than guess
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return out
    s_new = np.linspace(0, s[-1], n_points)
    xr, yr = np.interp(s_new, s, x), np.interp(s_new, s, y)
    theta = np.unwrap(np.arctan2(np.diff(yr), np.diff(xr)))
    return theta - theta.mean()


def track_to_angles(frames, n_points):
    return np.array([
        tangent_angles(f[0], f[1], n_points) if f is not None
        else np.full(n_points - 1, np.nan)
        for f in frames
    ])


# ---------------------------------------------------------------- PCA

def pca(X):
    """PCA via eigendecomposition of the covariance matrix (rows = frames)."""
    mean = X.mean(axis=0)
    cov = np.cov(X - mean, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)          # ascending order
    order = np.argsort(evals)[::-1]
    return mean, evals[order], evecs[:, order]  # columns = eigenworms


# ---------------------------------------------------------------- main

def collect_files(inputs):
    files = []
    for p in inputs:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "*.wcon.zip"))
            files += glob.glob(os.path.join(p, "*.wcon"))
        else:
            files += glob.glob(p)
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="WCON files, globs, or folders")
    ap.add_argument("--n-points", type=int, default=49,
                    help="skeleton points after resampling (default 49)")
    ap.add_argument("--n-modes", type=int, default=4,
                    help="eigenworms to plot and project onto (default 4)")
    ap.add_argument("--out", default="eigenworm_results", help="output folder")
    args = ap.parse_args()

    files = collect_files(args.inputs)
    if not files:
        raise SystemExit("No WCON files found.")
    os.makedirs(args.out, exist_ok=True)

    all_tracks = []  # (label, t, angles)
    for path in files:
        wcon = read_wcon(path)
        ventral = find_key(wcon.get("metadata", {}), "ventral_side")
        # Make the sign convention consistent across worms: a bend toward the
        # ventral side should have the same sign for every animal.
        sign = -1.0 if str(ventral).lower().startswith("anti") else 1.0
        if ventral is None:
            print(f"  note: no ventral_side in metadata for {os.path.basename(path)}; "
                  "check the Zenodo record and flip manually if needed")
        for wid, (t, frames) in load_tracks(wcon).items():
            ang = sign * track_to_angles(frames, args.n_points)
            n_ok = np.isfinite(ang).all(axis=1).sum()
            print(f"{os.path.basename(path)} [worm {wid}]: "
                  f"{n_ok}/{len(ang)} frames with full skeletons "
                  f"(ventral_side={ventral})")
            all_tracks.append((f"{os.path.basename(path)}:{wid}", t, ang))

    # Pool only complete frames for PCA
    X = np.vstack([a[np.isfinite(a).all(axis=1)] for _, _, a in all_tracks])
    if len(X) < args.n_points:
        raise SystemExit("Too few valid frames for PCA.")
    mean, evals, evecs = pca(X)
    frac = evals / evals.sum()
    cum = np.cumsum(frac)
    k = args.n_modes

    print(f"\nPCA on {len(X)} frames, {X.shape[1]} tangent angles each")
    for i in range(min(8, len(frac))):
        print(f"  mode {i+1}: {100*frac[i]:5.1f}%   cumulative {100*cum[i]:5.1f}%")

    # Project every frame (NaN rows stay NaN, so time gaps are preserved)
    projections = {lab: (ang - mean) @ evecs[:, :k] for lab, _, ang in all_tracks}

    np.savez(os.path.join(args.out, "eigenworms.npz"),
             mean_posture=mean, eigenvalues=evals, eigenworms=evecs[:, :k],
             variance_fraction=frac,
             **{f"t_{i}": t for i, (_, t, _) in enumerate(all_tracks)},
             **{f"angles_{i}": a for i, (_, _, a) in enumerate(all_tracks)},
             **{f"proj_{i}": projections[lab] for i, (lab, _, _) in enumerate(all_tracks)},
             labels=np.array([lab for lab, _, _ in all_tracks]))

    # ---- figure
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    n_show = min(10, len(frac))
    ax[0, 0].bar(range(1, n_show + 1), 100 * frac[:n_show], color="0.7", label="per mode")
    ax[0, 0].plot(range(1, n_show + 1), 100 * cum[:n_show], "o-", label="cumulative")
    ax[0, 0].set(xlabel="mode", ylabel="% variance", title="Variance explained")
    ax[0, 0].legend()

    s = np.linspace(0, 1, X.shape[1])
    for i in range(k):
        ax[0, 1].plot(s, evecs[:, i], label=f"eigenworm {i+1}")
    ax[0, 1].set(xlabel="body position (head = 0, tail = 1)",
                 ylabel="weight", title="Eigenworms (overall sign is arbitrary)")
    ax[0, 1].legend()

    P = np.vstack([p[np.isfinite(p).all(axis=1)] for p in projections.values()])
    # Modes 2 and 3 form the quadrature (sine/cosine-like) pair for this data,
    # so a travelling body wave traces a ring in the (a2, a3) plane.
    ax[1, 0].hist2d(P[:, 1], P[:, 2], bins=100, cmap="viridis")
    ax[1, 0].set(xlabel="a2", ylabel="a3", aspect="equal",
                 title="Posture space (a2 vs a3): crawling traces a ring")

    lab0, t0, _ = all_tracks[0]
    p0 = projections[lab0]
    dt = np.nanmedian(np.diff(t0)) if len(t0) > 1 else 1.0
    n_win = int(min(len(t0), max(1, round(10 / dt))))  # ~10 s window
    for i in range(min(k, 3)):
        ax[1, 1].plot(t0[:n_win] - t0[0], p0[:n_win, i], label=f"a{i+1}")
    ax[1, 1].set(xlabel="time (s)", ylabel="amplitude",
                 title=f"Mode amplitudes, first ~10 s\n{lab0[:50]}")
    ax[1, 1].legend()

    fig.tight_layout()
    fig_path = os.path.join(args.out, "eigenworms.png")
    fig.savefig(fig_path, dpi=150)
    print(f"\nSaved {fig_path} and eigenworms.npz to {args.out}/")

    # ---- phase & radius of the (a2, a3) quadrature pair
    if k >= 3:
        fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4))

        a2, a3 = p0[:n_win, 1], p0[:n_win, 2]
        radius = np.hypot(a2, a3)
        phase = np.arctan2(a3, a2)
        ok = np.isfinite(phase)
        phase_u = np.full_like(phase, np.nan)
        phase_u[ok] = np.unwrap(phase[ok])  # NaN gaps are skipped over
        tt = t0[:n_win] - t0[0]

        ax2[0].plot(tt, phase_u / (2 * np.pi))
        ax2[0].set(xlabel="time (s)", ylabel="unwrapped phase (cycles)",
                   title="Phase atan2(a3, a2)\nsteady slope = crawling; sign = direction")
        ax2[1].plot(tt, radius)
        ax2[1].set(xlabel="time (s)", ylabel=r"radius $\sqrt{a_2^2 + a_3^2}$",
                   title="Wave amplitude (low = paused / not undulating)")

        r_all = np.hypot(P[:, 1], P[:, 2])
        ax2[2].hist(r_all, bins=80, color="0.6")
        ax2[2].set(xlabel=r"radius $\sqrt{a_2^2 + a_3^2}$", ylabel="frames",
                   title="Radius, all frames\n(peak away from 0 = ring)")

        fig2.tight_layout()
        fig2_path = os.path.join(args.out, "eigenworms_phase.png")
        fig2.savefig(fig2_path, dpi=150)
        print(f"Saved {fig2_path}")


if __name__ == "__main__":
    main()