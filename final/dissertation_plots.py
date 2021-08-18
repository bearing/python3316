import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


def main():
    data_sets = ["/home/justin/Desktop/dissertation_data/data_all_protons/full.npz",
                 "/home/justin/Desktop/dissertation_data/data_subset_protons/full_109.npz",
                 "/home/justin/Desktop/dissertation_data/data_subset_protons/full_108.npz",
                 "/home/justin/Desktop/dissertation_data/data_subset_protons/full_5_107.npz"]

    total_protons = []
    std_error = []

    for data_file in data_sets:
        with np.load(data_file) as data:
            total_protons.append(data['protons'])
            positions = data['positions']
            raw_ranges = data['raw_ranges']

            fit = stats.linregress(positions, raw_ranges)

            fitted_ranges = fit.intercept + positions * fit.slope
            std_error.append((fitted_ranges - raw_ranges).std())

    print("Total Protons: ", total_protons)
    print("Std Error: ", std_error)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.semilogx(total_protons, 2 * np.array(std_error))
    ax.set_title("Protons vs. Range Uncertainty")
    ax.set_xlabel("Protons")
    ax.set_ylabel(r'$2\sigma$-Range Uncertainty [mm]')
    plt.show()


def plot_physics_too():
    data_sets = ["/home/justin/Desktop/dissertation_data/data_all_protons/full.npz",
                 "/home/justin/Desktop/dissertation_data/data_subset_protons/full_109.npz",
                 "/home/justin/Desktop/dissertation_data/data_subset_protons/full_108.npz",
                 "/home/justin/Desktop/dissertation_data/data_subset_protons/full_5_107.npz"]

    physc_data_sets = ["/home/justin/Desktop/dissertation_data/data_physics/carbon/all_protons.npz",
                       "/home/justin/Desktop/dissertation_data/data_physics/carbon/10_9.npz",
                       "/home/justin/Desktop/dissertation_data/data_physics/carbon/10_8.npz",
                       "/home/justin/Desktop/dissertation_data/data_physics/carbon/5_10_7.npz"]

    physo_data_sets = ["/home/justin/Desktop/dissertation_data/data_physics/oxygen/all_protons.npz",
                       "/home/justin/Desktop/dissertation_data/data_physics/oxygen/10_9.npz",
                       "/home/justin/Desktop/dissertation_data/data_physics/oxygen/10_8.npz",
                       "/home/justin/Desktop/dissertation_data/data_physics/oxygen/5_10_7.npz"]

    rgn_names = ['Original', 'Carbon w/ Kernel', 'Oxygen w/ Kernel']

    total_proton_lists = [[], [], []]
    std_error_lists = [[], [], []]

    for total_proton_list, std_error_list, data_set in \
            zip(total_proton_lists, std_error_lists, (data_sets, physc_data_sets, physo_data_sets)):
        for data_file in data_set:
            with np.load(data_file) as data:
                total_proton_list.append(data['protons'])
                positions = data['positions']
                raw_ranges = data['raw_ranges']

                fit = stats.linregress(positions, raw_ranges)

                fitted_ranges = fit.intercept + positions * fit.slope
                std_error_list.append((fitted_ranges - raw_ranges).std())

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for rgn_name, tps, errs in zip(rgn_names, total_proton_lists, std_error_lists):
        ax.semilogx(tps, 2 * np.array(errs), label=rgn_name)
    ax.set_title("Protons vs. Range Uncertainty")
    ax.set_xlabel("Protons")
    ax.set_ylabel(r'$2\sigma$-Range Uncertainty [mm]')
    ax.legend(loc='best')
    plt.show()


def plot_oxygen_carbon_shift(choose_data):
    physc_data_sets = ["/home/justin/Desktop/dissertation_data/data_physics/carbon/all_protons.npz",
                       "/home/justin/Desktop/dissertation_data/data_physics/carbon/10_9.npz",
                       "/home/justin/Desktop/dissertation_data/data_physics/carbon/10_8.npz",
                       "/home/justin/Desktop/dissertation_data/data_physics/carbon/5_10_7.npz"]

    physo_data_sets = ["/home/justin/Desktop/dissertation_data/data_physics/oxygen/all_protons.npz",
                       "/home/justin/Desktop/dissertation_data/data_physics/oxygen/10_9.npz",
                       "/home/justin/Desktop/dissertation_data/data_physics/oxygen/10_8.npz",
                       "/home/justin/Desktop/dissertation_data/data_physics/oxygen/5_10_7.npz"]

    rgns = ['Carbon', 'Oxygen']
    raw_ranges = [[], []]
    max_values = [[], []]
    positions = [[], []]

    chosen_data_sets = [physc_data_sets[choose_data], physo_data_sets[choose_data]]

    for rrs, max_vals, pos, data_file in zip(raw_ranges, max_values, positions, chosen_data_sets):
        with np.load(data_file) as data:
            max_vals.append(data['max_vals'])
            load_positions = data['positions']
            pos.append(load_positions)

            rrs.append(data['raw_ranges'])
            # raw_ranges = data['raw_ranges']
            # fit = stats.linregress(load_positions, raw_ranges)

            # frs.append(fit.intercept + load_positions * fit.slope)

    tpos = positions[0]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(tpos[0], raw_ranges[1][0] - raw_ranges[0][0])
    ax.set_title("Carbon PG Range - Oxygen PG Range")
    ax.set_xlabel("Positions")
    ax.set_ylabel(r'$\Delta$ Range Uncertainty [mm]')
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(tpos[0], max_values[1][0] / max_values[0][0])
    ax.set_title("Ratio of Maximum Line Profile Intensity")
    ax.set_xlabel("Positions")
    ax.set_ylabel(r'Carbon Max / Oxygen Max')
    plt.show()

    # max_vals = max_values, positions = positions, protons = self.protons, raw_ranges = raw_ranges,
    # was_normed = norm_plots, was_fitted = fit_done, fit_obj = fit_obj, lines = lines, x_proj_range = self.x_proj_range

    pass


if __name__ == "__main__":
    # main()
    # plot_physics_too()
    plot_oxygen_carbon_shift(0)
