#
import sys
sys.path.append('/Users/omert/Desktop/alanates/omfp-transformer/src')

import argparse
import tqdm

import numpy as np
from collections import Counter
from ase.io import read
from ase import Atoms

# Your OMFP imports
from omfp.fJustOMFP import fomfpall
from structure.Structure import Structure, ang2bohr



def atoms2structure(atoms):
    return Structure(atoms.get_atomic_numbers(),
                     atoms.get_positions() * ang2bohr,
                     lattice=atoms.get_cell(True).array * ang2bohr)


def get_energy_per_fu(atoms):
    """Return energy in eV per formula unit."""

    if "energy" in atoms.info:
        e = float(atoms.info["energy"])
    elif "Energy" in atoms.info:
        e = float(atoms.info["Energy"])
    elif atoms.calc is not None:
        e = float(atoms.get_potential_energy())
    else:
        raise ValueError("Energy not found")

    return e / FU_PER_CELL


def get_energy_per_a(atoms):
    """Return energy in eV per atoms."""

    if "energy" in atoms.info:
        e = float(atoms.info["energy"])
    elif "Energy" in atoms.info:
        e = float(atoms.info["Energy"])
    elif atoms.calc is not None:
        e = float(atoms.get_potential_energy())
    else:
        raise ValueError("Energy not found")

    return e / len(atoms)


def omfp_vector(atoms: Atoms) -> np.ndarray:
    """
    Compute OMFP fingerprint for one structure.
    Returns a 1D vector (flattened) for distance computations.
    """
    s = atoms2structure(atoms)
    fp = fomfpall(
        s.ats,
        s.elements,
        lattice=s.lattice,
        ns=1,
        npp=1,
        maxNatSphere=100
    )
    fp = np.asarray(fp, dtype=float)
    return fp.reshape(-1)  # (48,400) -> (19200,)


def l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance."""
    return float(np.linalg.norm(a - b))


def calibrate_delta_from_baseline(
    atoms_gs: Atoms,
    *,
    n_perturb: int = 25,
    disp_A: float = 0.02,
    seed: int = 7,
    percentile: float = 95.0
) -> float:
    """
    Build a vibrational-like baseline by perturbing ground state positions.
    delta = percentile of baseline distances (e.g., 95th).
    """
    rng = np.random.default_rng(seed)

    fp0 = omfp_vector(atoms_gs)
    dists = []

    for _ in range(n_perturb):
        a = atoms_gs.copy()
        # small random displacements (Å)
        a.positions = a.positions + rng.normal(scale=disp_A, size=a.positions.shape)
        fp = omfp_vector(a)
        dists.append(l2_dist(fp0, fp))

    dists = np.array(dists)
    delta = float(np.percentile(dists, percentile))
    print(f"[delta calibration] baseline distances: mean={dists.mean():.4g}, "
          f"p{percentile:.0f}={delta:.4g} (disp={disp_A} Å, n={n_perturb})")
    return delta


def revisit_counts_within_window(
    extxyz_path: str,
    *,
    window_eV: float = 0.050,
    delta: float or None = None,
    calibrate_delta: bool = True,
    # calibration settings
    n_perturb: int = 25,
    disp_A: float = 0.02,
    percentile: float = 95.0,
    seed: int = 7,
    # performance settings
    max_reps_for_search: int or None = None
):
    """
    Compute revisit counts for structures within ΔE ≤ window_eV.

    Online clustering:
      - keep a list of representative fingerprints for each discovered minimum
      - assign each new structure to nearest representative
      - if min distance <= delta -> revisit of that minimum
      - else -> new minimum

    Returns:
      counts_per_minimum: list of counts (length = number of unique minima in window)
      Emin, idx_gs, indices_in_window
    """
    atoms_list = read(extxyz_path, index=":")
    N = len(atoms_list)
    energies = np.array([get_energy_per_fu(a) for a in atoms_list], dtype=float)
    #  energies = np.array([get_energy_per_a(a) for a in atoms_list], dtype=float)

    idx_gs = int(np.argmin(energies))
    Emin = float(energies[idx_gs])

    dE = energies - Emin
    indices_in_window = np.where(dE <= window_eV)[0]
    print(f"Total frames: {N}")
    print(f"Global minimum: idx={idx_gs}, Emin={Emin:.8f} eV")
    print(f"Frames within ΔE ≤ {window_eV:.3f} eV: {len(indices_in_window)}")

    if len(indices_in_window) == 0:
        raise ValueError("No structures within the requested energy window.")

    # Choose delta if not provided
    if delta is None and calibrate_delta:
        delta = calibrate_delta_from_baseline(
            atoms_list[idx_gs],
            n_perturb=n_perturb,
            disp_A=disp_A,
            seed=seed,
            percentile=percentile
        )
    elif delta is None:
        raise ValueError("delta is None. Provide delta=... or set calibrate_delta=True.")

    print(f"Using clustering threshold delta = {delta:.6g}")

    # Online clustering containers
    reps = []          # representative fingerprint vectors
    counts = []        # revisit counts per representative
    rep_frame = []     # one example frame index for each representative

    # Optional: shuffle to avoid ordering bias (often OK to keep original order)
    # Here, keep deterministic order for reproducibility:
    for idx in tqdm.tqdm(indices_in_window, desc="Clustering structures"):
        fp = omfp_vector(atoms_list[idx])

        # If no clusters yet, create first
        if not reps:
            reps.append(fp)
            counts.append(1)
            rep_frame.append(int(idx))
            continue

        # Search nearest representative
        # Optional speed: limit search to first max_reps_for_search representatives
        # (not usually needed for ΔE window sizes that are modest)
        m = len(reps) if max_reps_for_search is None else min(len(reps), max_reps_for_search)

        best_k = -1
        best_d = np.inf
        for k in range(m):
            dd = l2_dist(fp, reps[k])
            if dd < best_d:
                best_d = dd
                best_k = k

        if best_d <= delta:
            counts[best_k] += 1
        else:
            reps.append(fp)
            counts.append(1)
            rep_frame.append(int(idx))

    counts = np.array(counts, dtype=int)

    # Print reviewer-friendly stats
    n_unique = len(counts)
    n_ge2 = int(np.sum(counts >= 2))
    frac_ge2 = 100.0 * n_ge2 / n_unique if n_unique else 0.0
    print("\n=== Revisit stats (within window) ===")
    print(f"Unique minima (by OMFP+delta): {n_unique}")
    print(f"Rediscovered ≥2 times: {n_ge2} ({frac_ge2:.1f}%)")
    print(f"Mean revisit count: {counts.mean():.2f}")
    print(f"Median revisit count: {np.median(counts):.1f}")
    print(f"Max revisit count: {counts.max()}")

    return counts, Emin, idx_gs, indices_in_window, rep_frame, delta



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Give something ...")
    parser.add_argument("-extxyz_path", type=str, required=True, help="..")
    args = parser.parse_args()
    extxyz_path = args.extxyz_path

    N_ATOMS_PER_FU = 6     # LiAlH4
    N_ATOMS_CELL = 48
    FU_PER_CELL = N_ATOMS_CELL // N_ATOMS_PER_FU   # = 8

    window_eV = 0.20    # eV per formula unit (tune as needed, e.g., 0.05 for 50 meV/fu)

    counts, Emin, idx_gs, idx_window, rep_frame, delta = revisit_counts_within_window(
        extxyz_path,
        window_eV=window_eV,
        delta=None,                 # auto-calibrate from baseline
        calibrate_delta=True,
        n_perturb=25,
        disp_A=0.02,                # Å (tune if you want)
        percentile=95.0,
        seed=7
    )

    np.save("revisit_counts.npy", counts)
    np.save("rep_frame_indices.npy", np.array(rep_frame, dtype=int))

