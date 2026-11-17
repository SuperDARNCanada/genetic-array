#!/usr/bin/env python3
"""Convert an xnec2c radiation-pattern CSV into the .npz that batch_genetic_solver.py loads.

CSV columns (xnec2c export): mhz,phi,theta,gain_total,gain_horiz,gain_vert,gain_rhcp,gain_lhcp

Output arrays (matches the existing genetic-array pattern files):
    data : (181, 360) float64   gain in dB, rows = theta 0..180 deg, cols = phi 0..359 deg
                                 missing/below-floor values set to -99.99
    az   : (360,) int64         0..359  (NEC phi, 0 = +x; boresight +y = 90)
    el   : (181,) int64         0..180  (NEC theta, degrees from zenith)

Usage:
    xnec2c_csv_to_npz.py pattern.csv out.npz [--mhz 10.5] [--gain-col gain_total]
If the CSV holds several frequencies, pass --mhz, or omit it to write one file per frequency
(out_<mhz>MHz.npz).
"""
import argparse
import numpy as np
import pandas as pd

FLOOR = -99.99


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("out")
    p.add_argument("--mhz", type=float, help="select one frequency from a multi-frequency CSV")
    p.add_argument("--gain-col", default="gain_total")
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
        out = a.out if len(freqs) == 1 else a.out.replace(".npz", f"_{f:.3f}MHz.npz")
        np.savez(out, data=data, az=az, el=el)
        print(f"{out}: {f:.3f} MHz data={data.shape} {data.dtype} min={data.min():.2f} "
              f"max={data.max():.2f} az={az.shape}[{az.min()}..{az.max()}] el={el.shape}")


if __name__ == "__main__":
    main()
