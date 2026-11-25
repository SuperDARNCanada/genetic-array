import os

import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from genetic_solver import GeneticSolver
from plotting import Multiple
import array_factor


def plot_trials(weights, gains, angles, title, filename, ripple=None, sidelobe=None, transition_width=None, show_ripple=False, show_side_lobes=False, show_transition_width=False, show_phase_wrapping=False, vlines=None):
    """Plot the gain pattern in the horizontal plane and phases of each antenna.

    Parameters
    ----------
    weights: list of np.array
        List of arrays containing complex weights for each antenna.
    gains: list
        List of arrays with gain values. Array dimensions should match 'angles'
    angles: np.array
        Array of angular directions corresponding to gain patterns.
    title: str
        Title of the plot.
    filename: str
        Name of file to save plot as.
    ripple: float
        The ripple in the gain pattern hardcoded, in dB.
    sidelobe: float
        The peak sidelobe level in the gain pattern hardcoded, in dB.
    transition_width: float
        The transition width in the gain pattern hardcoded, in degrees.
    show_ripple: bool
        If True, show the ripple in the gain pattern.
    show_side_lobes: bool
        If True, show the side lobes in the gain pattern.
    show_transition_width: bool
        If True, show the transition width in the gain pattern.
    show_phase_wrapping: bool
        If True, show the phase wrapped in the phase plot.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(title)

    # Set the y-axis limits for the gain plot
    pad = 2  # Extra breathing room (in dB) before rounding
    # Round min down and max up to the nearest multiple of 5
    y_floor = np.floor((np.min(gains) - pad) / 5) * 5
    y_roof = np.ceil((np.max(gains) + pad) / 5) * 5


    # Initialize the gain plot
    for i in range(len(gains)):
        ax1.plot(angles, gains[i])
        ax1.set_ylim(y_floor, y_roof)

    # Create vertical lines at the FOV boundaries
    if vlines is not None:
        for vline in vlines:
            ax1.axvline(x=vline, color='k', linestyle='--')

    # Create a shaded region for the ripple if requested
    if show_ripple and ripple is not None:
        gain_axis_edges = [np.max(gains), np.max(gains) - ripple]
        if np.min(angles) == 0:
            az_axis_edges = [0, vlines[0]]
        else:
            az_axis_edges = [vlines[0], vlines[1]]
        ax1.fill_between(az_axis_edges, gain_axis_edges[0], gain_axis_edges[1], color='green', alpha=0.3, label='Ripple')

    # Create a shaded region for the transition width if requested
    if show_transition_width and transition_width is not None:
        if np.min(angles) == 0:
            az_axis_edges = [vlines[0], vlines[0] + transition_width]
            ax1.axvspan(az_axis_edges[0], az_axis_edges[1], color='lightgray', alpha=0.3, zorder=0, label='Transition Width')
        else:
            az_axis_edges = [vlines[0], vlines[0] + transition_width]
            ax1.axvspan(-az_axis_edges[0], -az_axis_edges[1], color='lightgray', alpha=0.3, zorder=0)
            ax1.axvspan(az_axis_edges[0], az_axis_edges[1], color='lightgray', alpha=0.3, zorder=0, label='Transition Width')

    # Create a shaded region for the side lobe if requested
    # Note side lobe level is max of gain plus the sidelobe level (which is negative)
    if show_side_lobes and sidelobe is not None:
        gain_axis_edges = [y_roof, np.max(gains) + sidelobe]
        if np.min(angles) == 0:
            az_axis_edges = [vlines[0] + transition_width, np.max(angles)]
            ax1.fill_between(az_axis_edges, gain_axis_edges[0], gain_axis_edges[1], color='blue', alpha=0.3, label='Sidelobe')
        else:
            az_axis_edges = [vlines[0] + transition_width, np.max(angles)]
            ax1.fill_between(-az_axis_edges, gain_axis_edges[0], gain_axis_edges[1], color='blue', alpha=0.3)
            ax1.fill_between(az_axis_edges, gain_axis_edges[0], gain_axis_edges[1], color='blue', alpha=0.3, label='Sidelobe')


    # Create a legend for the shaded regions if any of them are shown
    if (show_ripple or show_side_lobes or show_transition_width) and np.min(angles) == 0:
        ax1.legend(loc='lower left')
    elif (show_ripple or show_side_lobes or show_transition_width):
        ax1.legend(loc='lower center')

    # Add grid lines and labels to the plots
    ax1.yaxis.grid()
    ax1.set_xlabel('Azimuth (degrees)')
    ax1.set_ylabel('Gain (dB)')

    # for i in range(len(weights)):
    #     mask = np.argwhere(np.abs(weights[i]) < 0.01)[:, 0]
    #     plot_arr = np.ma.masked_where(np.abs(weights[i]) < 0.01, np.unwrap(np.angle(weights[i])))
    #     line, = ax2.plot(plot_arr, marker='o')
    #     ax2.scatter(mask, np.zeros(mask.shape), marker='o', fc='white', ec=line.get_color())
    #     ax2.set_xlim(-0.5, weights.shape[1]-0.5)

    for i in range(len(weights)):
        inactive = np.abs(weights[i]) < 0.01
        mask = np.argwhere(inactive)[:, 0]
        if show_phase_wrapping:
            phase = np.mod(np.angle(weights[i]), 2 * np.pi)  # wrapped into [0, 2*pi)
        else:
            phase = np.unwrap(np.angle(weights[i]))
        plot_arr = np.ma.masked_where(inactive, phase)
        # Wrapped phases jump between 0 and 2*pi, so draw markers only (no connecting lines)
        line, = ax2.plot(plot_arr, marker='o', linestyle='-')# linestyle='none' if show_phase_wrapping else '-')
        ax2.scatter(mask, np.zeros(mask.shape), marker='o', fc='white', ec=line.get_color())
        ax2.set_xlim(-0.5, weights.shape[1]-0.5)

    major = Multiple(denominator=1)
    minor = Multiple(denominator=4)
    ax2.yaxis.set_major_locator(major.locator())
    ax2.yaxis.set_minor_locator(minor.locator())
    ax2.yaxis.set_major_formatter(major.formatter())
    ax2.yaxis.grid()

    ax2.set_xlabel("Antenna Index")
    ax2.set_ylabel("Phase (radians)")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def main():
    # Helper function to parse comma-separated lists of floats from arguments
    def comma_list(arg):
        return [float(x) for x in arg.split(',')]

    # Set up argument parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Add optional arguments for configuration
    parser.add_argument('--resolution', type=float, help='angular resolution in degrees', default=0.5)
    parser.add_argument('--spacing', type=float, help='antenna spacing in meters', default=15.24)
    parser.add_argument('--left-bound', type=float, help='left bound of FOV in degrees CW from boresight', default=-30.0)
    parser.add_argument('--right-bound', type=float, help='right bound of FOV in degrees CW from boresight', default=30.0)
    parser.add_argument('--antennas-down', type=str, help='antennas not transmitting, format as 0,1,2,14', default='')
    parser.add_argument('--ripple', type=comma_list, help='ripple values in dB, format as 2,3,4', default='2,3,4')
    parser.add_argument('--side-lobes', type=comma_list, help='side lobe values dB, format as -20,-17,-14', default='-20,-17,-14')
    parser.add_argument('--transition-width', type=comma_list, help='transition width values in Degrees, format as 3,4,5', default='5.0')
    parser.add_argument('--pop-size', type=int, help='population size for genetic algorithm', default=100)
    parser.add_argument('--num-trials', type=int, help='number of trials for genetic algorithm', default=10)

    # Add optional arguments for plotting
    parser.add_argument('--plot-show-ripple', action="store_true", help='show ripple in gain plots')
    parser.add_argument('--plot-show-side-lobes', action="store_true", help='show side lobes in gain plots')
    parser.add_argument('--plot-show-transition-width', action="store_true", help='show transition width in gain plots')
    parser.add_argument('--plot-show-phase-wrapping', action="store_true", help='show phase wrapping in phase plots')

    # Add mandatory arguments
    parser.add_argument('antenna_pattern', type=str, help='.npz file with antenna pattern data')
    parser.add_argument('outdir', type=str, help='directory to put results and plots in')
    parser.add_argument('freqs', nargs='+', type=float, help='frequencies in kHz to optimize for')
    args = parser.parse_args()

    assert 0.1 < args.resolution < 2.0, "--resolution invalid"
    assert 2.0 < args.spacing < 20.0, "--spacing invalid"

    num_antennas = 16   # main array
    bad_antennas = []
    if args.antennas_down != '':
        for entry in args.antennas_down.split(','):
            val = int(entry)
            assert (0 <= val < num_antennas), f"--antennas-down invalid input {args.antennas_down}"
            bad_antennas.append(val)

    angular_res = args.resolution  # degrees
    left_bound = args.left_bound  # FOV boundary, in degrees right of boresight
    assert -90 < left_bound < 0, "--left-bound invalid"
    right_bound = args.right_bound  # FOV boundary, in degrees right of boresight
    assert 0 < right_bound < 90, "--right-bound invalid"
    elevation = 0       # degrees up from horizon
    antenna_spacing = args.spacing     # meters
    labels = []

    if np.isclose(-left_bound, right_bound) and len(bad_antennas) == 0:
        symmetric = True
    else:
        symmetric = False

    # Position of each antenna along y-axis
    antenna_positions = array_factor.default_antenna_positions(num_antennas, antenna_spacing)

    # Arrays of spherical coordinate points
    els = np.array([elevation])
    if symmetric:
        azimuths = np.arange(0, 90, angular_res)
        vlines = (right_bound,)
    else:
        azimuths = np.arange(-90, 90, angular_res)
        vlines = (left_bound, right_bound)
    passband = (left_bound, right_bound)

    def add_configuration(w, label):
        """Add a set of weights with associated label for simulation."""
        weights.append(w)
        labels.append(label)

    # Common mode frequencies
    freqs = [x * 1000 for x in args.freqs]  # Hz

    if args.ripple is None:
        ripples = [2.0, 3.0, 4.0, 5.0]  # dB
    else:
        ripples = args.ripple
    if args.side_lobes is None:
        sidelobe_levels = [-20, -17, -14]  # dB
    else:
        sidelobe_levels = args.side_lobes
    if args.transition_width is None:
        transition_widths = [5.0]  # degrees
    else:
        transition_widths = args.transition_width
    if args.pop_size is None:
        population_size = 100
    else:
        population_size = args.pop_size
    if args.num_trials is None:
        num_trials = 10
    else:
        num_trials = args.num_trials

    # Riley Testing Parameters
    # freqs = [x * 1000 for x in args.freqs]  # Hz
    # ripples = [3.0]  # dB
    # sidelobe_levels = [-20]  # dB
    # transition_widths = [2.0]  # degrees
    # population_size = 100
    # num_trials = 100

    print("Ripples: {}".format(ripples))
    print("Sidelobe Levels: {}".format(sidelobe_levels))
    print("Transition Widths: {}".format(transition_widths))
    print("Freqs: {}".format(freqs))
    print("Population Size: {}".format(population_size))
    print("Number of Trials: {}".format(num_trials))

    for freq in freqs:
        print("Frequency: {:.3f} MHz".format(freq * 1e-6))
        el_factor = np.load(args.antenna_pattern)
        interp_data = np.interp(azimuths, el_factor['az'] - 90,
                                el_factor['data'][70])  # 70 is the colatitude, in degrees
        interp_data -= interp_data.max()
        interp_data = np.power(10, interp_data / 20)
        best_weights = []
        best_penalties = []
        perfect = False
        for ripple in ripples:
            if perfect:
                break
            for sidelobe in sidelobe_levels:
                for width in transition_widths:
                    weights = []
                    penalties = []
                    for trial in range(num_trials):
                        gs = GeneticSolver(
                            num_antennas,
                            antenna_spacing,
                            freq,
                            ripple,
                            sidelobe,
                            passband,
                            width,
                            azimuths,
                            population_size,
                            element_factor=interp_data,
                            bad_antennas=bad_antennas
                        )
                        add_configuration(gs.weights, 'Trial {}'.format(trial))
                        penalties.append(gs.best_penalty)
                        if gs.best_penalty == 0.0:
                            break

                    weights = np.array(weights)

                    # ensure that all weights start out increasing with antenna index, avoids mirrored solutions
                    # looking different.
                    for i in range(weights.shape[0]):
                        mags = np.abs(weights[i])
                        non_zero_weights = weights[i, mags > 0.0]
                        if np.angle(non_zero_weights[1]) < 0.0:
                            weights[i] = weights[i].conj()
                    penalties = np.array(penalties)

                    best_penalty_idx = np.argmin(penalties)
                    best_penalties.append(penalties[best_penalty_idx])
                    best_weights.append(weights[best_penalty_idx, :])

                    perfect = (penalties[best_penalty_idx] == 0.0)

                    # Compute the array factor for each set of weights.
                    af = array_factor.array_factor(weights, antenna_positions, freq, els, azimuths, el_factor=interp_data)

                    gains = [af[i, 0, :] for i in range(af.shape[0])]

                    # Plot the results for this configuration
                    title = '{:.3f} MHz, {:.1f}dB Ripple, {:.1f}dB Peak Sidelobe, {:.1f} degree Transition Width, {} Antennas' \
                            ''.format(freq * 1e-6, ripple, sidelobe, width, num_antennas)
                    filename = f'{args.outdir}/Genetic_Solutions_{freq*1e-6:.3f}MHz_{ripple:.1f}ripple_{sidelobe:.1f}sidelobe_{width:.1f}width_{num_antennas}antennas.png'

                    plot_trials(weights, gains, azimuths, title, filename,
                                ripple, sidelobe, width,
                                show_ripple=args.plot_show_ripple,
                                show_side_lobes=args.plot_show_side_lobes,
                                show_transition_width=args.plot_show_transition_width,
                                show_phase_wrapping=args.plot_show_phase_wrapping,
                                vlines=vlines)

        best_weights = np.array(best_weights)
        best_penalties = np.array(best_penalties)

        best_weights_deg = np.rad2deg(np.unwrap(np.angle(best_weights), axis=-1))

        for s, w in zip(best_penalties, best_weights):
            variable_phases = np.rad2deg(np.unwrap(np.angle(w)))
            print(f"Penalty: {s:.3f}\tPhases (deg): {variable_phases}")

        # Compute the array factor for each set of weights.
        af = array_factor.array_factor(best_weights, antenna_positions, freq, els, azimuths, el_factor=interp_data)

        gains = [af[i, 0, :] for i in range(af.shape[0])]

        # Save the results to an HDF5 file
        base = f'{args.outdir}/results_{int(freq/1000):05d}kHz'
        outfile = f'{base}.hdf5'
        for version in range(1, 10000): # if you have more than 10,000 versions of the same frequency, you have bigger problems
            if not os.path.exists(outfile):
                break
            outfile = f'{base}_v{version}.hdf5'

        with h5py.File(outfile, 'a') as f:
            group_name = f'{int(freq/1000):05d}'
            g = f.create_group(group_name)
            g.create_dataset('complex_phases', data=best_weights)
            g.create_dataset('phases_deg', data=best_weights_deg)
            g.create_dataset('antenna_mags', data=np.abs(best_weights))
            g.create_dataset('penalties', data=best_penalties)
            g.create_dataset('array_factor', data=np.array(gains))
            g.create_dataset('grid', data=azimuths)

        # Plot the results for this configuration
        title = '{:.3f} MHz'.format(freq * 1e-6)
        filename = '{}/Genetic_Solutions_{:.1f}MHz_{}antennas.png' \
                   ''.format(args.outdir, freq * 1e-6, num_antennas)
        plot_trials(best_weights, gains, azimuths, title, filename,
                                        ripple, sidelobe, width,
                                        show_ripple=args.plot_show_ripple,
                                        show_side_lobes=args.plot_show_side_lobes,
                                        show_transition_width=args.plot_show_transition_width,
                                        show_phase_wrapping=args.plot_show_phase_wrapping,
                                        vlines=vlines)


if __name__ == '__main__':
    main()

