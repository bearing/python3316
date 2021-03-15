import tables
import io
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial

# '/Users/justinellin/Desktop/Davis 2020/Tuesday/2020-10-06-1503.h5'
# -> beam axis is +x-hat, away from earth is +y-hat


class events_recon(object):
    _crystal_coordinates = 'peak_coords_mean.txt'

    def __init__(self, filepath):
        self.h5file = load_signals(filepath)
        self.crd = load_coordinates(self._crystal_coordinates)
        self.pxl_mapper = pixel_mapper(self.crd)
        self.histogram_bins = np.arange(0, 100000, 1000)
        self.gamma_events = 0
        self.proton_events = 0
        # self.histogram_bins = np.linspace(0, 100000, 3000)

    def get_time_between_pulses(self):
        idx = 64
        edges = np.linspace(0, 24000, 3000)
        deltas = np.zeros(edges.size-1)

        folder = '/det' + str(idx)
        table = self.h5file.get_node('/', folder).EventData
        evts = table.nrows
        self.proton_events = evts

        tstamp = table.col('timestamp')

        maw_max = table.col('maw_max')
        trig_after = table.col('maw_after_trig')
        trig_before = table.col('maw_before_trig')

        blk_ind = 0
        chunk = 20000
        process = True
        while process:
            start = blk_ind * chunk
            last_evt = (blk_ind + 1) * chunk + 1
            if last_evt < evts:
                ts = tstamp[start:last_evt]
                tmp_max = 1.0 * maw_max[start:last_evt] - 0x8000000
                tmp_after = 1.0 * trig_after[start:last_evt] - 0x8000000
                tmp_before = 1.0 * trig_before[start:last_evt] - 0x8000000
                blk_ind += 1
            else:
                ts = tstamp[start:]
                tmp_max = 1.0 * maw_max[start:] - 0x8000000
                tmp_after = 1.0 * trig_after[start:] - 0x8000000
                tmp_before = 1.0 * trig_before[start:] - 0x8000000
                process = False

            # This follows Struct

            # time_after = (tmp_max/2) - tmp_after
            # back_interp = tmp_before - tmp_after
            # interp = (1.0 * time_after + 1)/(back_interp +1)
            # interp[interp < 0] = 1


            # real_ts = ts - interp
            real_ts = ts
            delta = np.diff(real_ts)
            # delta[delta<1000]=0

            deltas += np.histogram(delta, edges)[0]
            # if blk_ind == 0:
            #    deltas, edges = np.histogram(delta, bins=200)
            #else:
            #    deltas += np.histogram(delta, edges)[0]

        return deltas, edges

    def plot_sci_pulses(self, value, signals):
        folder = '/det' + str(64)
        node = self.h5file.get_node('/', folder)
        earray = node.raw_data[value:value+signals, :]
        print("Events: ", node.raw_data.nrows)
        return earray

    def get_stamps_between_pulses(self):
        idx = 64
        # edges = np.linspace(0, 1000, 1000)
        edges = np.arange(-1001,1001,1)
        deltas = np.zeros(edges.size-1)
        max_delta = 0

        folder = '/det' + str(idx)
        table = self.h5file.get_node('/', folder).EventData
        evts = table.nrows
        self.proton_events = evts

        tstamp = table.col('timestamp')
        # tstamp.dtype
        # print("First 5 timestamps: ", tstamp[:3])
        # print("Difference of first 5 timestamps: ", np.ediff1d(tstamp[:3]))

        blk_ind = 0
        chunk = 20000
        process = True
        while process:
            start = blk_ind * chunk
            print("Start: ", start)
            last_evt = (blk_ind + 1) * chunk + 1
            if last_evt < evts:
                ts = tstamp[start:last_evt]
                blk_ind += 1
            else:
                ts = tstamp[start:]
                process = False

            # This follows Struct

            # time_after = (tmp_max/2) - tmp_after
            # back_interp = tmp_before - tmp_after
            # interp = (1.0 * time_after + 1)/(back_interp +1)
            # interp[interp < 0] = 1

            # real_ts = ts - interp
            delta = np.mod(np.ediff1d(ts), 1000)
            long = np.max(delta)
            if long > max_delta:
                max_delta = long
            # delta[delta<1000]=0

            deltas += np.histogram(delta, edges)[0]
            # if blk_ind == 0:
            #    deltas, edges = np.histogram(delta, bins=200)
            #else:
            #    deltas += np.histogram(delta, edges)[0]
        print("Largest Delta:", max_delta)
        return deltas, edges

    def get_histograms(self, start_index):
        module = np.arange(4)
        tables = [None] * 4
        energy_process = [None] * 4
        # Ex = [None] * 4
        # Ey = [None] * 4
        neg_Es = np.zeros(4)
        proj = np.zeros([12, 12])

        max_energy_bin = 0
        energy_hist = np.zeros(self.histogram_bins.size - 1)

        for integer in module:
            folder = '/det' + str(int(start_index + integer))
            tables[integer] =  self.h5file.get_node('/', folder).EventData

        evts = tables[0].nrows
        if evts < 1:
            return None, None
        self.gamma_events += evts
        # print("Total Events:", evts)
        blk_ind = 0
        chunk = 10000
        process = True
        while process:
            start = blk_ind * chunk
            last_evt = (blk_ind + 1) * chunk
            for channel in module:
                if last_evt < evts:
                    tmp = tables[channel].col('gate2')[start:last_evt] - 3 * tables[channel].col('gate1')[start:last_evt]
                    blk_ind += 1
                else:
                    tmp = tables[channel].col('gate2')[start:] - 3 * tables[channel].col('gate1')[start:]
                    process = False
                neg_Es[channel] += np.sum(tmp <= 0, axis=0)
                tmp[tmp < 0] = 0
                energy_process[channel] = tmp

            sum = energy_process[0] + energy_process[1] + energy_process[2] + energy_process[3]
            # +x -> beam direction, +y -> sky
            Ex = (energy_process[0] - energy_process[2]) + (energy_process[1] - energy_process[3])
            Ey = (energy_process[0] - energy_process[1]) + (energy_process[2] - energy_process[3])

            scaleX = ((Ex + 1)/0.02)
            scaleY = (Ey + 1)/0.02
            closest_pxls = self.pxl_mapper.query(np.column_stack([Ex, Ey]))[1]
            pxls = np.bincount(closest_pxls, minlength=144)
            proj += pxls.reshape([12, 12])

            energy_hist += np.histogram(sum, bins=self.histogram_bins)[0]
        return energy_hist, proj  #, neg_Es

    def get_histograms2(self, start_index):  # Make 1 when it works
        module = np.arange(4)
        tables = [None] * 4
        energy_process = [None] * 4
        # Ex = [None] * 4
        # Ey = [None] * 4
        neg_Es = np.zeros(4)
        # proj = np.zeros([100, 100])
        proj = np.zeros([200, 200])
        max_energy_bin = 0
        energy_hist = np.zeros(self.histogram_bins.size - 1)
        for integer in module:
            folder = '/det' + str(int(start_index + integer))
            tables[integer] = self.h5file.get_node('/', folder).EventData
        evts = tables[0].nrows
        energy_array = np.zeros(evts, dtype=np.float)
        for channel in module:
            energy_process[channel] = energy_array.copy()
        if evts < 1:
            return None, None
        self.gamma_events += evts
        # print("Total Events:", evts)
        blk_ind = 0
        chunk = 10000
        process = True
        # sanity_check = np.zeros(4)  # Note: -x, +x, -y, +y TODO: WHAT IS GOING ON?
        tmp_check = 0
        energy_process[0] = tables[0].col('gate2') - 3.0 * tables[0].col('gate1')
        energy_process[1] = tables[1].col('gate2') - 3.0 * tables[1].col('gate1')
        energy_process[2] = tables[2].col('gate2') - 3.0 * tables[2].col('gate1')
        energy_process[3] = tables[3].col('gate2') - 3.0 * tables[3].col('gate1')

        sum = energy_process[0] + energy_process[1] + energy_process[2] + energy_process[3]

        # print('Number of zero energies, channel 0:', np.sum(energy_process[0] == 0))
        # print('Number of zero energies, channel 1:', np.sum(energy_process[1] == 0))
        # print('Number of zero energies, channel 2:', np.sum(energy_process[2] == 0))
        # print('Number of zero energies, channel 3:', np.sum(energy_process[3] == 0))
        # print('Number of zero sums:', np.sum(sum == 0))
        # +x -> beam direction, +y -> sky
        Ex = ((energy_process[0] - energy_process[2]) + (energy_process[1] - energy_process[3])) / (1.0 * sum)
        Ey = ((energy_process[0] - energy_process[1]) + (energy_process[2] - energy_process[3])) / (1.0 * sum)
        # scaleX = ((Ex + 1)/0.02)
        # scaleX[scaleX >= 100] = 99
        # scaleX[scaleX < 0] = 0
        # Ex[Ex <= -1] = -0.995
        # Ex[Ex >= 1] = 0.995
        print("Ex:", Ex)
        print("Ex:", Ex.shape)
        # sanity_check[0] = np.sum(Ex <= 0)
        # sanity_check[1] = np.sum(Ex > 0)
        # sanity_check[2] = np.sum(Ey <= 0)
        # sanity_check[3] = np.sum(Ey > 0)
        # scaleY = (Ey + 1)/0.02
        # scaleY[scaleY >= 100] = 99
        # scaleY[scaleY < 0] = 0
        # Ey[Ey <= -1] = -0.995
        # Ey[Ey >= 1] = 0.995
        # binX = np.arange(101)
        binX = np.linspace(-1.0, 1.0, 201)
        # binX = np.linspace(-1.0, 1.0, 201)
        # binY = np.arange(101)
        binY = np.linspace(-1.0, 1.0, 201)
        # binY = np.linspace(-1.0, 1.0, 201)
        # closest_pxls = self.pxl_mapper.query(np.column_stack([Ex, Ey]))[1]
        # pxls = np.bincount(closest_pxls, minlength=144)
        # proj += np.histogram2d(scaleX, scaleY, bins=[binX, binY])[0]
        # proj += np.histogram2d(Ex, Ey, bins=[binX, binY])[0]
        proj += np.histogram2d(Ex, Ey, bins=[binX, binY])[0]
        # proj += pxls.reshape([12, 12])
        # energy_hist += np.histogram(sum, bins=self.histogram_bins)[0]
        energy_hist += np.histogram(sum, bins=self.histogram_bins)[0]
        return energy_hist, proj  # ,neg_Es  #,sanity_check

    def get_energy(self, index, start, chunk):
        folder = '/det' + str(index)
        table = self.h5file.get_node('/', folder).EventData
        events = table.nrows
        process = True

        # last_evt = 0  # processed
        blk_ind = 0
        energies = np.zeros(events)
        g2 = table.col('gate2')
        g1 = table.col('gate1')
        negatives = 0

        while process:
            # start = blk_ind * chunk
            last_evt = (blk_ind + 0) * chunk
            if last_evt < events:
                temp = g2[start:last_evt] - 3 * g1[start:last_evt]
                negatives += np.sum(temp <= 0, axis=0)
                temp[temp < 0] = 0
                energies[start:last_evt] = temp
                blk_ind += 1
            else:
                temp = g2[start:] - 3 * g1[start:]
                negatives += np.sum(temp <= 0, axis=0)
                energies[start:] = temp
                process = False

        return energies, negatives


def load_signals(filepath):
    if tables.is_hdf5_file(filepath):
        h5file = tables.open_file(filepath, 'r')
        return h5file
    else:
        raise ValueError('{fi} is not a hdf5 file!'.format(fi=filepath))


def load_coordinates(calib_file):
    f = open(calib_file, 'r')
    coordinates = eval(f.read().replace(' ', ','))
    f.close()
    return np.array(coordinates)


def pixel_mapper(crds):
    roots = crds[:, :2]  # need to flip, John X is along skyward axis, y along beam. Opposite to mine
    roots[:, [0, 1]] = roots[:, [1, 0]]
    roots[:, 1] *= -1
    roots[:, 1] += 100
    # print('Data points for Tree:, ', spatial.cKDTree(roots).n)
    return spatial.cKDTree(roots)


def find_pileups(filepath):
    if tables.is_hdf5_file(filepath):
        h5file = tables.open_file(filepath, 'r')
        pileup_vals = h5file.root.EventData.col('trigger')
        return h5file, pileup_vals
    else:
        raise ValueError('{fi} is not a hdf5 file!'.format(fi=filepath))


def main():
    # file = '/Users/justinellin/Desktop/Davis 2020/Tuesday/2020-10-06-1503.h5'
    # file ='/Users/justinellin/repos/python_SIS3316/Data/2020-10-07-1600.h5'
    file = '/home/proton/repos/python3316/Data/2020-10-08-1148.h5'
    tst = events_recon(file)
    # print(tst.get_energy(9,0, 20000))
    # print("Coordinate slice:", tst.crd[:][:2])

    # TODO: This should be temporary
    # projection = np.zeros([48, 48])
    projection = np.zeros([100, 100])

    # module_ind = np.arange(1)
    module_ind = np.array([6])
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for idx in module_ind:
        print("Current Index: ", idx)
        # row = 3 - (idx//4)
        # col = idx % 4
        # eng, proj = tst.get_histograms(4 * idx)

        eng, proj = tst.get_histograms2(4 * idx) # TODO: Remove when fixed

        if eng is not None and proj is not None:
            # x = tst.histogram_bins
            x = np.linspace(0, 100000, eng.size)
            # projection[12*row:(12*(row+1)), 12*col:(12*(col+1))] = proj
            projection = proj
            ax1.step(x, eng, label='mod'+str(idx))
    tst.h5file.close()
    print("Total Gamma Events:", tst.gamma_events)
    ax1.set_yscale('log')
    ax1.legend(loc='best')
    im = ax2.imshow(projection, cmap='viridis', interpolation='nearest')
    fig.colorbar(im, ax=ax2)
    plt.show()
    # plt.yscale('log')
    # plt.legend(loc='best')
    # plt.show()

    # eng, proj = tst.get_histograms(12)
    # print("Length of eng:", eng.size)
    # x = np.linspace(0, 100000, eng.size)
    # x = np.linspace(0, 100000, eng.size)
    # plt.step(x, eng)
    # plt.yscale('log')
    # plt.plt(eng)
    # plt.show()


def main2():
    # file = '/Users/justinellin/Desktop/Davis 2020/Tuesday/2020-10-06-1503.h5'
    file = '/home/proton/repos/python3316/Data/2020-10-08-1148.h5'
    tst = events_recon(file)
    dels, bin_edge = tst.get_stamps_between_pulses()
    # flt = np.argmax(bin_edge > 1000)
    # dels[0:flt] = 0 # Why does this not work?

    print("Total Proton Events:", tst.proton_events)
    print("Sum of deltas: ", np.sum(dels))
    plt.hist(dels, bin_edge, histtype='step')
    plt.yscale('log')
    # plt.yscale('log')
    # plt.legend(loc='best')
    plt.show()


def main3():
    file = '/Users/justinellin/repos/python_SIS3316/Data/2020-10-07-1600.h5'
    tst = events_recon(file)
    # print(tst.h5file)
    value = 23400226/2
    pulses = 10
    traces = tst.plot_sci_pulses(value, pulses)
    print(traces.shape)
    #print(tst.plot_4_sci_pulses(0))
    for ind, trace in enumerate(traces):
        plt.plot(np.arange(26), trace, label='mod' + str(ind))
        # plt.step(np.arange(samples), trace, label='mod' + str(ind))
    plt.title('10 Raw Plastic Scintillator Traces 5 Minutes in')
    plt.xlabel('Sample')
    plt.ylabel('ADC')
    plt.show()


if __name__ == "__main__":
    main()
    # main2()
    # main3()