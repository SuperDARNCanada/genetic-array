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
import numpy as np
import pandas as pd

FLOOR = -99.99
COLAT_ROW = 70  # row batch_genetic_solver.py reads (20 deg elevation)


def convert(df, gain_col):
    df = df.copy()
    # xnec2c prints angles like 0.99999999994945232, so round to whole degrees
    df["theta_i"] = np.rint(df["theta"]).astype(int)
    df["phi_i"] = np.rint(df["phi"]).astype(int) % 360     # folds phi=360 onto 0
    grid = df.pivot_table(index="theta_i", columns="phi_i", values=gain_col, aggfunc="mean")
    grid = grid.reindex(index=np.arange(181), columns=np.arange(360))
    data = grid.to_numpy(dtype=np.float64)
    if np.isnan(data).any():
        print(f"  warning: {int(np.isnan(data).sum())} grid cells missing, set to {FLOOR}")
    data = np.nan_to_num(data, nan=FLOOR)
    data = np.where(data < FLOOR, FLOOR, data)              # xnec2c uses -999.99 for no-data
    return data, np.arange(360, dtype=np.int64), np.arange(181, dtype=np.int64)


def reorient(data, boresight_phi=90.0, mirror=False):
    """Rotate/mirror azimuth so boresight lands at az = 90 (what batch_genetic_solver.py assumes)."""
    shift = int(round(boresight_phi - 90.0))
    data = np.roll(data, -shift, axis=1)          # new[j] = old[j + shift]
    if mirror:
        data = data[:, (180 - np.arange(360)) % 360]   # reflect about az = 90
    return data


def report_symmetry(data, half_width=60):
    """Print where the pattern peaks and how left/right asymmetric it is about az = 90."""
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
    a = p.parse_args()

    df = pd.read_csv(a.csv, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    freqs = np.sort(df["mhz"].round(6).unique())
    if a.mhz is not None:
        freqs = [f for f in freqs if abs(f - a.mhz) < 1e-3]
        if not freqs:
            raise SystemExit(f"{a.mhz} MHz not in CSV")

    for f in freqs:
        sub = df[np.isclose(df["mhz"], f, atol=1e-3)]
        data, az, el = convert(sub, a.gain_col)
        data = reorient(data, a.boresight_phi, a.mirror)
        out = a.out if len(freqs) == 1 else a.out.replace(".npz", f"_{f:.3f}MHz.npz")
        np.savez(out, data=data, az=az, el=el)
        print(f"{out}: {f:.3f} MHz data={data.shape} {data.dtype} min={data.min():.2f} "
              f"max={data.max():.2f} az={az.shape}[{az.min()}..{az.max()}] el={el.shape}")
        report_symmetry(data)


if __name__ == "__main__":
    main()
