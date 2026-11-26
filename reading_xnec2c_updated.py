#!/usr/bin/env python3
"""Convert an xnec2c radiation-pattern CSV into the .npz that batch_genetic_solver.py loads.

CSV columns (xnec2c export): mhz,phi,theta,gain_total,gain_horiz,gain_vert,gain_rhcp,gain_lhcp

Output arrays (matches the existing genetic-array pattern files):
    data : (181, 360) float64   gain in dB, rows = theta 0..180 deg, cols = phi 0..359 deg
                                 missing/below-floor values set to -99.99
    az   : (360,) int64         0..359
    el   : (181,) int64         0..180  (NEC theta, degrees from zenith)

Azimuth convention expected by batch_genetic_solver.py: it evaluates np.interp(az_solver, az - 90, data[70]),
so the antenna BORESIGHT must sit at az = 90 in this file, and az increases in the same sense as the solver's
azimuth. nec_sd_generator.py puts boresight on +y (NEC phi = 90) so the defaults need no change. If your model
points elsewhere use --boresight-phi, and use --mirror if the solver's left/right is swapped.

Usage:
    reading_xnec2c.py pattern.csv out.npz [--mhz 10.5] [--gain-col gain_total]
                         [--boresight-phi 90] [--mirror]
If the CSV holds several frequencies, pass --mhz, or omit it to write one file per frequency
(out_<mhz>MHz.npz).
"""
import argparse
import os
import numpy as np
import pandas as pd

# Colors for pretty printing output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

FLOOR = -99.99
COLAT_ROW = 70  # row batch_genetic_solver.py reads (20 deg elevation)
NTHETA = 181
NPHI = 360
CHUNK_ROWS = 2_000_000  # rows per CSV chunk while streaming


def stream_convert(csv_path, gain_col, mhz_filter=None, chunk_rows=CHUNK_ROWS):
    """
    Read the CSV in chunks and accumulate (sum, count) grids per frequency,
    instead of loading the whole file into one DataFrame and pivoting it.

    A full-sized xnec2c radiation-pattern export (every theta x phi x
    frequency row logged individually) can run into many millions of rows;
    pandas' pivot_table/reindex path builds large intermediate structures on
    top of the already-large DataFrame, which can exhaust memory on an
    ordinary workstation -- when that happens inside a C extension (pandas/
    numpy), the process can crash with a segfault instead of a clean Python
    MemoryError. Streaming in chunks and aggregating into small, fixed-size
    (181, 360) arrays keeps peak memory roughly independent of file size.

    :param csv_path: path to the xnec2c CSV export
    :param gain_col: which gain column to read (e.g. "gain_total")
    :param mhz_filter: if given, only this frequency (MHz, matched within
                       1e-3) is accumulated; rows for other frequencies are
                       dropped as each chunk is read, so they never need to
                       be held in memory at all
    :param chunk_rows: how many CSV rows to read per chunk
    :return: (data_by_freq, total_rows_seen) where data_by_freq maps each
            frequency (float, rounded to 6 decimals) found to its (181, 360)
            float64 gain grid, and total_rows_seen is the number of data
            rows read from the CSV (for diagnostics)
    """
    # Peek the header so we can match column names after stripping whitespace
    # (xnec2c's exported headers can have stray spaces) without reading the
    # whole file.
    header = pd.read_csv(csv_path, nrows=0, skipinitialspace=True)
    stripped_by_orig = {c: c.strip() for c in header.columns}
    needed = {"mhz", "phi", "theta", gain_col}
    usecols = [orig for orig, stripped in stripped_by_orig.items() if stripped in needed]
    found = {stripped_by_orig[c] for c in usecols}
    missing = needed - found
    if missing:
        raise SystemExit(f"CSV is missing required column(s): {', '.join(sorted(missing))}")

    # float32 for the angle/gain columns roughly halves memory versus the
    # pandas default float64; frequency stays float64 so round(.., 6)
    # grouping below can't be thrown off by reduced precision.
    dtype_map = {orig: (np.float64 if stripped_by_orig[orig] == "mhz" else np.float32)
                 for orig in usecols}

    sums = {}
    counts = {}
    total_rows = 0
    oob_theta = 0

    reader = pd.read_csv(csv_path, usecols=usecols, dtype=dtype_map,
                         skipinitialspace=True, chunksize=chunk_rows)
    for chunk in reader:
        chunk.columns = [stripped_by_orig[c] for c in chunk.columns]
        total_rows += len(chunk)

        if mhz_filter is not None:
            chunk = chunk[np.isclose(chunk["mhz"], mhz_filter, atol=1e-3)]
            if chunk.empty:
                continue

        freq_keys = chunk["mhz"].round(6).to_numpy()
        # xnec2c prints angles like 0.99999999994945232, so round to whole degrees
        theta_i = np.rint(chunk["theta"].to_numpy()).astype(np.int64)
        phi_i = np.rint(chunk["phi"].to_numpy()).astype(np.int64) % 360  # folds phi=360 onto 0
        gains = chunk[gain_col].to_numpy(dtype=np.float64)

        in_range = (theta_i >= 0) & (theta_i < NTHETA)
        n_bad = int((~in_range).sum())
        if n_bad:
            oob_theta += n_bad
            theta_i, phi_i, gains, freq_keys = (arr[in_range] for arr in
                                                (theta_i, phi_i, gains, freq_keys))

        for f in np.unique(freq_keys):
            mask = freq_keys == f
            if f not in sums:
                sums[f] = np.zeros((NTHETA, NPHI), dtype=np.float64)
                counts[f] = np.zeros((NTHETA, NPHI), dtype=np.int64)
            np.add.at(sums[f], (theta_i[mask], phi_i[mask]), gains[mask])
            np.add.at(counts[f], (theta_i[mask], phi_i[mask]), 1)

    if oob_theta:
        print(f"  {YELLOW}warning:{RESET} {oob_theta} row(s) had theta outside [0, 180] and were dropped")

    data_by_freq = {}
    for f, count_grid in counts.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            data = sums[f] / count_grid
        missing_mask = count_grid == 0
        n_missing = int(missing_mask.sum())
        if n_missing:
            print(f"  {YELLOW}warning:{RESET} {n_missing} grid cells missing for {f:.3f} MHz, set to {FLOOR}")
        data = np.where(missing_mask, FLOOR, data)
        data = np.where(data < FLOOR, FLOOR, data)              # xnec2c uses -999.99 for no-data
        data_by_freq[f] = data

    return data_by_freq, total_rows


def reorient(data, boresight_phi=90.0, mirror=False):
    """Rotate/mirror azimuth so boresight lands at az = 90 (what batch_genetic_solver.py assumes)."""
    shift = int(round(boresight_phi - 90.0))
    data = np.roll(data, -shift, axis=1)          # new[j] = old[j + shift]
    if mirror:
        data = data[:, (180 - np.arange(360)) % 360]   # reflect about az = 90
    return data


def report_symmetry(data, half_width=60):
    """Print where the pattern peaks and how left/right asymmetric it is about az = 90. 0 dB means perfectly symmetric"""
    row = data[COLAT_ROW]
    k = np.arange(0, half_width + 1)
    left, right = row[(90 - k) % 360], row[(90 + k) % 360]
    diff = np.abs(left - right)
    peak = int(np.argmax(row[:181]))
    print(f"    row {COLAT_ROW}: peak at az={peak} (boresight should be 90), "
          f"L/R asymmetry within +/-{half_width} deg: mean {diff.mean():.2f} dB, max {diff.max():.2f} dB")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("out")
    p.add_argument("--mhz", type=float, help="select one frequency from a multi-frequency CSV")
    p.add_argument("--gain-col", default="gain_total")
    p.add_argument("--boresight-phi", type=float, default=90.0,
                   help="NEC phi (deg) of the antenna boresight in the CSV; it is rotated to az=90")
    p.add_argument("--mirror", action="store_true", help="reflect azimuth about boresight (swap left/right)")
    p.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS,
                   help="CSV rows to read per chunk while streaming (lower this if memory is still tight)")
    a = p.parse_args()

    data_by_freq, total_rows = stream_convert(a.csv, a.gain_col, mhz_filter=a.mhz,
                                              chunk_rows=a.chunk_rows)
    if a.mhz is not None and not data_by_freq:
        raise SystemExit(f"{RED}{a.mhz} MHz not in CSV{RESET}")
    if not data_by_freq:
        raise SystemExit(f"{RED}No data rows found in {a.csv} (read {total_rows} row(s)){RESET}")

    freqs = sorted(data_by_freq)
    az = np.arange(NPHI, dtype=np.int64)
    el = np.arange(NTHETA, dtype=np.int64)

    for f in freqs:
        data = reorient(data_by_freq[f], a.boresight_phi, a.mirror)
        out = a.out if len(freqs) == 1 else a.out.replace(".npz", f"_{f:.3f}MHz.npz")
        out_dir = os.path.dirname(out)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        np.savez(out, data=data, az=az, el=el)
        print(f"{GREEN}{out}:{RESET} {f:.3f} MHz\n{GREEN}data{RESET}={data.shape} {data.dtype} {GREEN}min{RESET}={data.min():.2f} "
              f"{GREEN}max{RESET}={data.max():.2f}\n{GREEN}az shape{RESET}={az.shape} {GREEN}min..max{RESET}=[{az.min()}..{az.max()}]\n{GREEN}el shape{RESET}={el.shape}\n")
        report_symmetry(data)


if __name__ == "__main__":
    main()
