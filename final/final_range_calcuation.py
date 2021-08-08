import numpy as np
from matplotlib import pyplot as plt


def lin_interp(x, y, i, level):
    return x[i] + (x[i+1] - x[i]) * ((level - y[i]) / (y[i+1] - y[i]))


def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]


def frac_max_x(y, f=0.5, offset=0):  # (x, y, f=0.5)
    """Returns interpolated indexes of crossing frac of max y value. Indexes are relative to input.
     Offset adds that value to the index. Useful for masked lines"""
    x = np.arange(y.size)
    frac_max = max(y) * f
    signs = np.sign(y - frac_max)
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [offset + lin_interp(x, y, zero_crossings_i[0], frac_max),
            offset + lin_interp(x, y, zero_crossings_i[1], frac_max)]


class RangePlotter(object):

    def __init__(self, image_stack_fname):
        # images, steps, x_proj_range
        data = np.load(image_stack_fname)
        self.images = data['images']
        self.n_steps, self.py, self.px = self.images.shape

        self.x_proj_range = data['x_proj_range']
        self.stage_positions = data['steps']  # Stage Positions in mm
        self.target_positions = 50 - self.stage_positions  # Positions relative to center of collimator
        try:
            self.protons = data['protons']
        except Exception as e:
            self.protons = None


        self.mid_sid = 34  # slice id
        self.slice_hw = 2  # half width

    def plot_single_image(self, i, mids=None, hws=None):
        """i is the index you want to plot, mids and hws can be lists to check for optimal projection location"""
        if mids is None:
            mids = [self.mid_sid]
        if hws is None:
            hws = [self.slice_hw]

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 4), subplot_kw={'xticks': [], 'yticks': []})

        single_image = self.images[i]
        # single_image[np.arange(self.mid_sid - self.slice_hw, self.mid_sid + self.slice_hw + 1), :] = 0
        # single_image[:,150:] = 0
        img = ax0.imshow(single_image, cmap='magma', origin='upper', interpolation='nearest')
        ax0.set_title("Pos " + str(self.stage_positions[i]))
        fig.colorbar(img, ax=ax0, fraction=0.045, pad=0.04)

        if len(mids) * len(hws) > 1:
            colormap = plt.cm.nipy_spectral(np.linspace(0, 1, len(mids) * len(hws)))
            ax1.set_prop_cycle('color', colormap)

        for mid in mids:
            for hw in hws:
                line = np.mean(single_image[np.arange(mid - hw,
                                                      mid + hw + 1), :], axis=0)
                ax1.plot(self.x_proj_range, line,
                         label='mid ' + str(mid) + ', hw ' + str(hw))  # system i
        ax1.legend(loc='best')

        plt.tight_layout()
        plt.show()

    def test_range_thresholds(self, thresh=0.5, norm_plots=False, plot_residuals=True):
        from scipy import stats

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 4))

        colormap = plt.cm.nipy_spectral(np.linspace(0, 1, self.target_positions.size))
        ax0.set_prop_cycle('color', colormap)
        ax0.set_xlabel("[mm]")
        if norm_plots:
            ax0.set_ylabel("Normalized Counts")
        else:
            ax0.set_ylabel("Counts")

        determined_ranges = np.zeros(self.target_positions.size)

        for rid, (image, position) in enumerate(zip(self.images, self.target_positions)):
            line = np.mean(image[np.arange(self.mid_sid - self.slice_hw, self.mid_sid + self.slice_hw + 1), :], axis=0)
            if norm_plots:
                line /= np.max(line)

            ax0.plot(self.x_proj_range, line, label="{z} mm".format(z=position))

            # TODO: Fix artifact so you DON'T need this
            rising_edge, falling_edge = frac_max_x(line[:150], f=thresh)
            determined_ranges[rid] = falling_edge

        # ax0.legend(loc='best')
        determined_ranges += - (200 / 2) - 0.5

        ax1.set_xlabel("Stage Position")
        range_label = "PG 58% Falloff Position [mm]"
        if plot_residuals:
            range_label = "Residuals [mm]"
            fit = stats.linregress(self.target_positions, determined_ranges)
            best_fits = fit.intercept + self.target_positions * fit.slope
            determined_ranges = best_fits - determined_ranges
            print(f"R-Squared: {fit.rvalue**2:.4f}")
            print(f"Best Fit Slope: {fit.slope:.4f}. Best Fit Intercept: {fit.intercept:.4f}")
            print("Standard Deviation of Error: ", determined_ranges.std())
            print("Max Error: ", np.abs(determined_ranges).mean())

        # TODO: Here is where you would determine a global reference point
        ax1.set_ylabel(range_label)
        ax1.plot(self.target_positions, determined_ranges, '--bo')

        plt.tight_layout()
        plt.show()

    def plot_range_thresholds(self, select_images=None, norm_plots=False, mask_offsets=(50, 150), plot_residuals=True):
        """select images = indices of images to select from a stack. Norm_plots normalizes the counts to the max
         of the line. Mask_offsets masks the region of interest for calculation and max calculation. Plot_residuals
         True means plot residuals from a trend linear regression fit line. Otherwise plot absolute position"""
        from scipy import stats

        imgs = self.images
        positions = self.target_positions
        if select_images is not None:
            imgs = self.images[select_images]
            positions = self.target_positions[select_images]

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 4))

        colormap = plt.cm.nipy_spectral(np.linspace(0, 1, positions.size))
        ax0.set_prop_cycle('color', colormap)

        determined_ranges = np.zeros(positions.size)

        for rid, (image, position) in enumerate(zip(imgs, positions)):
            line = np.mean(image[np.arange(self.mid_sid - self.slice_hw, self.mid_sid + self.slice_hw + 1), :],
                           axis=0)
            if norm_plots:
                line /= np.max(line[mask_offsets[0]:mask_offsets[1]])

            ax0.plot(self.x_proj_range, line, label="{z} mm".format(z=position))

            # TODO: Fix artifact so you DON'T need this
            rising_edge, falling_edge = frac_max_x(line[mask_offsets[0]:mask_offsets[1]], f=0.42)
            determined_ranges[rid] = falling_edge

        # ax0.legend(loc='best')
        determined_ranges += - (200 / 2) - 0.5

        if plot_residuals:
            fit = stats.linregress(positions, determined_ranges)
            best_fits = fit.intercept + positions * fit.slope
            determined_ranges = best_fits - determined_ranges
            print(f"R-Squared: {fit.rvalue ** 2:.4f}")
            print(f"Best Fit Slope: {fit.slope:.4f}. Best Fit Intercept: {fit.intercept:.4f}")
            print("Standard Deviation of Error: ", determined_ranges.std())
            print("Max Error: ", np.abs(determined_ranges).mean())

        # TODO: Here is where you would determine a global reference point
        ax1.plot(positions, determined_ranges, '--bo')

        plt.tight_layout()
        plt.show()


def main_test():
    stack = '/home/justin/Desktop/final_images/test/stack.npz'
    rp = RangePlotter(stack)
    rp.plot_single_image(-1, hws=np.array([1, 2, 4]))
    # test_mids = np.array([34])
    # test_widths = np.array([1, 2, 3, 4, 8])
    # rp.plot_single_image(5, mids=test_mids, hws=test_widths)
    # CONCLUSION: id = 34, HW is 1 or 2


def frac_test():
    tst = np.zeros([50])
    tst[6:8] = 1
    print("Test: ", tst)
    print("Frac test: ", frac_max_x(tst))


def test_range_finding(**kwargs):
    stack = '/home/justin/Desktop/final_images/test/stack.npz'
    rp = RangePlotter(stack)
    rp.test_range_thresholds(**kwargs)


def plot_range_finding(**kwargs):
    stack = '/home/justin/Desktop/final_images/test/stack.npz'
    rp = RangePlotter(stack)
    rp.plot_range_thresholds(**kwargs)


if __name__ == "__main__":
    # main_test()
    # frac_test()
    # test_range_finding(thresh=0.42, norm_plots=True, plot_residuals=True)
    plot_range_finding(select_images=None, norm_plots=True, plot_residuals=True)
    # TODO: Plot range thresholds, allow to be selected
