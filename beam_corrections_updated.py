#!/usr/bin/env python3
"""Compute beam-direction corrections for the combined main/interferometer receive beams.

Example:
    python beam_corrections.py ttfd_12000khz.npz outdir 12000 --weights results.hdf5
    python beam_corrections.py ttfd_12000khz.npz outdir 12000            # falls back to cached_weights()

Inputs
    pattern    : .npz element pattern from xnec2c_csv_to_npz.py (keys: data, az, el)
    outdir     : directory for plots (created if missing)
    freq_khz   : frequency in kHz, e.g. 12000
    --weights  : results.hdf5 written by batch_genetic_solver.py. The group named after freq_khz is
                 used and the lowest-penalty row is picked unless --weights-index is given.
"""
import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

import array_factor as af

NUM_ANTENNAS = 16  # main array; the windows below are 16 elements long

HAMMING_40 = np.array([0.08081232549588463, 0.12098514265395757, 0.23455777475180511, 0.4018918165398586,
                       0.594054435182454, 0.7778186328978896, 0.9214100134552521, 1.0,
                       1.0, 0.9214100134552521, 0.7778186328978896, 0.594054435182454,
                       0.4018918165398586, 0.23455777475180511, 0.12098514265395757, 0.08081232549588463])
CHEBYSHEV_30 = np.array([0.2910, 0.3173, 0.4557, 0.6018, 0.7424, 0.8637, 0.9528, 1.0000,
                         1.0000, 0.9528, 0.8637, 0.7424, 0.6018, 0.4557, 0.3173, 0.2910])
WINDOWS = {'chebyshev30': CHEBYSHEV_30, 'hamming40': HAMMING_40, 'none': np.ones(NUM_ANTENNAS)}

PLOT_COLORS = ['tab:blue', 'tab:orange', 'tab:green']


def cached_weights(num_antennas, freq_khz: int):
    """Hard-coded fallback weights. Prefer --weights results.hdf5 for anything new."""
    cached_values_16_antennas = {
        12000: [
            1. - 0.j, -0.59954983 + 0.80033743j, -0.9575112 - 0.2883961j, 0.3799921 - 0.92498976j,
            0. - 0.j, -0.77975094 + 0.6260899j, -0.02049998 - 0.99978983j, -0.33486956 - 0.9422645j,
            -0.8806308 - 0.47380316j, 0.35340735 - 0.9354695j, 0.5615241 - 0.82746035j, -0.9999143 - 0.01309336j,
            -0.00693482 + 0.999976j, 0.86015075 + 0.5100398j, 0. - 0.j, -0.37670207 - 0.92633444j
        ],
        10500: [
            1. - 0.j, -0.48133704 + 0.8765356j, 0.6082936 - 0.7937121j, 0.11753573 - 0.99306864j,
            -0.9724519 - 0.23310351j, -0.35397393 + 0.9352553j, -0.5135601 + 0.8580536j, 0.88627064 - 0.46316776j,
            0.88627064 - 0.46316776j, -0.5135601 + 0.8580536j, -0.35397393 + 0.9352553j, -0.9724519 - 0.23310351j,
            0.11753573 - 0.99306864j, 0.6082936 - 0.7937121j, -0.48133704 + 0.8765356j, 1. - 0.j
        ],
    }
    if num_antennas != 16:
        raise ValueError('Number of antennas not supported')
    if freq_khz not in cached_values_16_antennas:
        raise KeyError(f'No cached weights for {freq_khz} kHz; pass --weights results.hdf5')
    angles = cached_values_16_antennas[freq_khz]
    if len(angles) == num_antennas // 2:
        phases = np.concatenate((angles, np.flip(angles)))
    elif len(angles) == num_antennas:
        phases = np.array(angles)
    else:
        raise ValueError(f"Cached values has incongruous shape: expected {num_antennas} or "
                         f"{num_antennas // 2}, found {len(angles)}")
    if np.isrealobj(phases):
        return np.exp(-1j * np.deg2rad(phases))
    return phases


def load_weights(path, freq_khz, index=None):
    """Load complex weights for freq_khz from a results.hdf5 made by batch_genetic_solver.py."""
    with h5py.File(path, 'r') as f:
        group = f'{freq_khz:05d}'
        if group not in f:
            raise KeyError(f"Group '{group}' not in {path}; available: {list(f.keys())}")
        weights = f[group]['complex_phases'][:]
        penalties = f[group]['penalties'][:]
    if index is None:
        index = int(np.argmin(penalties))
    print(f'Using weights row {index} of {len(weights)} (penalty {penalties[index]:.3f}) from {path}')
    return weights[index]


def calculate_directivities(weights, freq, antenna_spacing, angular_res, normalize=False):
    """Calculates the directivity for a set of weights"""
    # Position of each antenna along y-axis
    antenna_positions = np.arange(len(weights[0])) * antenna_spacing

    # Arrays of spherical coordinate points
    els = np.array([0])  # degrees up from horizon
    azimuths = np.arange(-90, 90, angular_res)

    # Compute the array factor for each set of weights.
    results = af.array_factor(weights, antenna_positions, freq, els, azimuths)

    if normalize:
        directivities = [results[i, 0, :] - np.max(np.abs(results[i, 0, :])) for i in range(results.shape[0])]
    else:
        directivities = [results[i, 0, :] for i in range(results.shape[0])]

    return directivities, azimuths, els


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('pattern', help='.npz element pattern (keys data, az, el) from xnec2c_csv_to_npz.py')
    p.add_argument('outdir', help='directory for output plots (created if missing)')
    p.add_argument('freq_khz', type=int, help='frequency in kHz')
    p.add_argument('--weights', help='results.hdf5 from batch_genetic_solver.py (default: cached_weights())')
    p.add_argument('--weights-index', type=int, help='row in the hdf5 group to use (default: lowest penalty)')
    p.add_argument('--spacing', type=float, default=15.24, help='antenna spacing in m')
    p.add_argument('--resolution', type=float, default=0.01, help='azimuth grid step in degrees')
    p.add_argument('--colat', type=float, default=70.0, help='colatitude (deg from zenith) cut of the element pattern')
    p.add_argument('--window', choices=sorted(WINDOWS), default='chebyshev30', help='main array receive taper')
    p.add_argument('--dir-min', type=float, default=-45.0, help='min nominal beam direction, deg')
    p.add_argument('--dir-max', type=float, default=45.0, help='max nominal beam direction, deg')
    p.add_argument('--dir-step', type=float, default=0.1, help='nominal beam direction step, deg')
    p.add_argument('--num-beams', type=int, default=24, help='number of beams to compute corrections for')
    p.add_argument('--beam-sep', type=float, default=3.24, help='nominal beam separation, deg')
    p.add_argument('--plot-direction', type=float, default=-3.24,
                   help='beam direction (deg) at which to save the example pattern plot')
    args = p.parse_args()
    assert 0.001 <= args.resolution < 2.0, '--resolution invalid'
    assert 2.0 < args.spacing < 20.0, '--spacing invalid'
    assert args.dir_min < args.dir_max and args.dir_step > 0, '--dir-* invalid'
    return args


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    freq_khz = args.freq_khz
    freq = freq_khz * 1000.0  # Hz
    spacing = args.spacing
    angular_res = args.resolution
    window = WINDOWS[args.window]
    directions = np.arange(args.dir_min, args.dir_max + args.dir_step / 2, args.dir_step)
    az_grid = np.arange(-90, 90, angular_res)

    if args.weights:
        main_weights = load_weights(args.weights, freq_khz, args.weights_index)
    else:
        main_weights = cached_weights(NUM_ANTENNAS, freq_khz)

    ### Load in the antenna factor from the NEC simulation ###
    nec_data = np.load(args.pattern)
    colat_idx = int(np.argmin(np.abs(nec_data['el'] - args.colat)))
    interp_data = np.interp(az_grid, nec_data['az'] - 90, nec_data['data'][colat_idx, :])
    interp_data -= interp_data.max()

    pointing_diffs = []
    for direction in directions:
        dirs, _, _ = calculate_directivities(
            [
                main_weights,
                window * af.linear_phase(af.default_antenna_positions(NUM_ANTENNAS, antenna_spacing=spacing),
                                         freq, direction)
            ],
            freq, spacing, angular_res, normalize=True,
        )
        intf_dirs, _, _ = calculate_directivities(
            [af.linear_phase(af.default_antenna_positions(4, antenna_spacing=spacing), freq, direction)],
            freq, spacing, angular_res, normalize=True)
        dirs.append(intf_dirs[0])
        dirs = [np.power(10, d / 20) for d in dirs]
        dirs = [d * np.power(10, interp_data / 20) for d in dirs]  # add in element factor
        dirs = [d / np.max(np.abs(d)) for d in dirs]  # normalize by the peak value

        multiplied = [dirs[i] * np.abs(dirs[0]) for i in range(1, len(dirs))]
        convolved = [multiplied[0] * np.conj(multiplied[0]),
                     multiplied[1] * np.conj(multiplied[1]),
                     multiplied[0] * np.conj(multiplied[1])]

        medians = []
        for c in convolved:
            peak, props = scipy.signal.find_peaks(
                10 * np.log10(np.abs(c)),
                height=-50,
                prominence=(5, None)
            )
            if len(peak) > 1:
                # Only keep the main lobe peak (should be the tallest peak)
                main_idx = np.argmax(props['peak_heights'])

                # Constrain width calculation to stop at the closest null between lobes
                prominences = props['prominences']
                left_bases = props['left_bases']
                right_bases = props['right_bases']

                prominence = prominences[main_idx]
                if main_idx != 0:
                    left_base = left_bases[main_idx - 1]
                    if prominences[main_idx - 1] < prominence:
                        prominence = prominences[main_idx - 1]
                else:
                    left_base = left_bases[main_idx]
                if main_idx != len(peak) - 1:
                    right_base = right_bases[main_idx + 1]
                    if prominences[main_idx + 1] < prominence:
                        prominence = prominences[main_idx + 1]
                else:
                    right_base = right_bases[main_idx]

                peak = np.ones((1,), dtype=np.int64) * peak[main_idx]
                width = scipy.signal.peak_widths(
                    10 * np.log10(np.abs(c)),
                    peak,
                    prominence_data=(
                        np.ones((1,)) * prominence,
                        np.ones((1,), dtype=int) * left_base,
                        np.ones((1,), dtype=int) * right_base
                    ),
                    rel_height=1.0
                )
                lower_bound = int(np.ceil(width[2])[0])
                upper_bound = int(np.floor(width[3])[0])
            else:
                continue

            # Calculate median of the main lobe
            cumsum = np.cumsum(np.abs(c[lower_bound:upper_bound]))
            cdf = cumsum / cumsum[-1]
            medians.append(az_grid[lower_bound:upper_bound][np.argwhere(cdf >= 0.5)[0]][0])

        pointing_diffs.append(medians)

        if np.abs(direction - args.plot_direction) < 0.05:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.yaxis.grid(visible=True)
            fig.tight_layout()
            labels = ['Main', 'Intf', 'Main-Intf']
            af.plot_horizontal_gain(fig, ax, 20 * np.log10(np.abs([dirs[0]])), az_grid, ['$D_t$'], ['tab:purple'],
                                    bounds=False)
            af.plot_horizontal_gain(fig, ax, 10 * np.log10(np.abs(convolved)), az_grid, labels, PLOT_COLORS,
                                    bounds=False)
            ax.axvline(direction, c='black')
            ax.set_ylim(-40)
            ax.set_title('')
            ax.set_ylabel('Relative Power [dB]')
            ax.legend(loc='upper right')
            plt.savefig(f'{args.outdir}/el_directivities.png', dpi=300, bbox_inches='tight')
            plt.close()

    fig, ax = plt.subplots(1, 1, subplot_kw={'aspect': 'equal'})
    ax.plot(directions, directions, c='k')
    for d in [0, 2]:
        ax.plot(directions, np.array([diffs[d] for diffs in pointing_diffs]), c=PLOT_COLORS[d])
    ax.grid(True)
    ax.legend([r'$\phi_d$', 'Main', 'Main-Intf'])
    ax.set_xlabel(r'Nominal Receiver Beam Direction, $\phi_d$ [degrees]')
    ax.set_ylabel('Median Combined Beam Direction [degrees]')
    plt.savefig(f'{args.outdir}/adjusted_beam_directions.png', dpi=300, bbox_inches='tight')
    plt.close()

    nominal_directions = (np.arange(args.num_beams) - (args.num_beams - 1) / 2) * args.beam_sep
    pointing_diffs = np.array(pointing_diffs)
    new_directions = np.round(np.interp(nominal_directions, pointing_diffs[:, -1], directions), decimals=1)
    print(f'XCFs: {new_directions}')
    new_directions = np.round(np.interp(nominal_directions, pointing_diffs[:, 0], directions), decimals=1)
    print(f'ACFs: {new_directions}')


if __name__ == '__main__':
    main()
