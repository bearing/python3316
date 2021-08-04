import numpy as np
import pickle
import tables
from matplotlib import pyplot as plt
# Use kernel_parser before sysmat_physics_basis


def load_response(sysmat_fname, name='sysmat'):  # pt_angles is other
    """Loads all at once"""
    sysmat_file_obj = load_h5file(sysmat_fname)
    data = sysmat_file_obj.get_node('/', name).read()  # TODO: Check if transpose needed
    sysmat_file_obj.close()  # TODO: Check that data is still saved
    return data


def load_response_table(sysmat_fname, name='sysmat'):  # pt_angles is other
    """Returns table object (index as normal)"""
    sysmat_file_obj = load_h5file(sysmat_fname)
    data_table_obj = sysmat_file_obj.get_node('/', name)  # TODO: Check if need to transpose
    return sysmat_file_obj, data_table_obj  # also returns file object to be closed


def load_h5file(filepath):  # h5file.root.sysmat[:]
    if tables.is_hdf5_file(filepath):
        h5file = tables.open_file(filepath, 'r')
        return h5file
    else:
        raise ValueError('{fi} is not a hdf5 file!'.format(fi=filepath))


class physics_basis(object):  # PMMA

    m_dens = 1.18  # g/cm^3 PMMA
    elements = ('C', 'O')  # 4.4 MeV and 6.1 MeV lines

    # leg_norms =  (2 * np.arange(7) + 1)/2.0  # i.e. up to a6 term, WAS NOT NEEDED

    fields = ['Energy',  # from cross section files
              'Oxy712_sig', 'Oxy712_a20',
              'Oxy692_sig', 'Oxy692_a20', 'Oxy692_a40',
              'Oxy613_sig', 'Oxy613_a20', 'Oxy613_a40', 'Oxy613_a60',
              'C_sig', 'C_a20', 'C_a40',
              'Oxy274_sig',
              'n_dens',  # constants, 1/cm^3
              'frac_O', 'frac_C', 'frac_H',  # from  atom_fractions()
              'projected_8MeV',  # from pstar
              'projected_ranges'
              ]

    # oxy_fields = ('Oxy613_sig', 'Oxy613_a20', 'Oxy613_a40', 'Oxy613_a60')
    # carbon_fields = ('C_sig', 'C_a20', 'C_a40')

    def __init__(self, kfile_name, im_pxl_sze=0.1):
        self.im_pxl_sze = im_pxl_sze
        with open(kfile_name, "rb") as fp:
            self.params = pickle.load(fp)

    def fold_energy_averaged(self, sysmat_fname, angle_fname,
                             x_pixels=201, element='C',  # normalized=True,
                             save=True, include_sin_term=False):
        """Folds in energy averaged cross sections. Can fold in carbon and/or oxygen. Here carbon is 4.4 line,
        and oxygen is only 6.1 MeV. Image pixel size given in cm. Normalized defines probabilities
         relative to a0 term, else calculates absolute given PMMA parameters"""
        both = False
        carbon = False
        oxygen = False

        element = element.upper()
        if element in ('CO', 'OC'):
            both = True
            carbon = True
            oxygen = True
        else:
            if element not in self.elements:
                ValueError("Element {e} not in allowed elements list: {a}".format(e=element.upper(), a=self.elements))
            if element == 'C':
                carbon = True
            else:
                oxygen = True

        mb_convert = 1e-27  # (cm^2/mb)
        l_objs = {}
        save_name = 'e_avg'  # folded system response name
        # if normalized:
        #    save_name += 'norm_'

        tot_legendre = np.polynomial.Legendre(np.zeros(7))

        if carbon:
            save_name += 'C'
            wgt_C = np.array([1, 0,
                              self.params['C_a20'].mean(), 0,  # a2, a3
                              self.params['C_a40'].mean(), 0,  # a4, a5
                              0  # a6
                              ])

            # if not normalized:  # i.e. absolute probabilities, TODO: include if you want absolute values
            #    wgt_C *= self.params['frac_C'] * self.params['C_sig'].mean() * \
            #             self.im_pxl_sze * self.params['n_dens'] * mb_convert
            # else:  # relative probabilities
            #    if both:  # carbon rel. to oxygen
            #        wgt_C *= self.params['frac_C']/ (self.params['frac_C'] + self.params['frac_O'])

            if both:  # carbon rel. to oxygen
                rel_int_prob = self.params['C_sig']/(self.params['C_sig'] + self.params['Oxy613_sig'])
                wgt_C *= rel_int_prob *  self.params['frac_C'] / (self.params['frac_C'] + self.params['frac_O'])

            bas_C = np.polynomial.Legendre(wgt_C)
            tot_legendre += bas_C
            l_objs['Carbon'] = bas_C
            # yld_Oxy613 = bas_Oxy613(costh)

        if oxygen:
            save_name += 'O'
            wgt_613 = np.array([1, 0,
                                self.params['Oxy613_a20'].mean(), 0,  # a2, a3
                                self.params['Oxy613_a40'].mean(), 0,  # a4, a5
                                self.params['Oxy613_a60'].mean()  # a6
                                ])
            # if not normalized:  # absolute, TODO: include if you want absolute values
            #     wgt_613 *=  self.params['frac_O'] * self.params['Oxy613_sig'].mean() *\
            #                 self.im_pxl_sze * self.params['n_dens'] * mb_convert
            # else:  # relative probabilities
            #    if both:  # oxygen rel. to carbon
            #        wgt_613 *= self.params['frac_O'] / (self.params['frac_C'] + self.params['frac_O'])

            if both:  # oxygen rel. to carbon
                rel_int_prob = self.params['Oxy613_sig'] / (self.params['C_sig'] + self.params['Oxy613_sig'])
                wgt_613 *= rel_int_prob * self.params['frac_O'] / (self.params['frac_C'] + self.params['frac_O'])

            bas_Oxy613 = np.polynomial.Legendre(wgt_613)
            tot_legendre += bas_Oxy613
            l_objs['Oxygen'] = bas_Oxy613

        if not save:
            return tot_legendre, l_objs, save_name

        s_file, s_table = load_response_table(sysmat_fname, name='sysmat')
        a_file, a_table = load_response_table(angle_fname, name='pt_angles')

        assert s_table.nrows == a_table.nrows, \
            "sysmat rows: {s}. angle rows: {a}".format(s=s_table.nrows, a=a_table.nrows)

        if include_sin_term:
            save_name += '_wSin'
        else:
            save_name += '_noSin'

        new_file = tables.open_file(save_name, mode="w", title="E Avg Folded System Response")
        folded_sysmat = new_file.create_earray('/', 'sysmat',
                                               atom=tables.atom.Float64Atom(),
                                               shape=(0, 48 * 48),
                                               expectedrows=s_table.nrows)
        if include_sin_term:
            # This corrects for solid angle subtended at constant polar angle (here, relative to beam axis)
            r_angles = a_table.read()  # response angles
            folded_sysmat.append(s_table.read() * tot_legendre(np.cos(r_angles))) / np.sin(r_angles)
        else:
            folded_sysmat.append(s_table.read() * tot_legendre(np.cos(a_table.read())))
        new_file.flush()

        new_file.close()
        s_file.close()
        a_file.close()
        return tot_legendre, l_objs, save_name

    def _pos_weights(self, debug=True):
        """Generate weights for each term for position averaging."""
        ranges = self.params['projected_ranges']/self.m_dens
        r0 = self.params['projected_8MeV'] / self.m_dens
        diff_dist = np.diff(ranges, prepend=r0)  # no gamma emission below 8 MeV

        wgts = []
        idxs = []

        pos = 0
        wgt_prev = 0
        n_energies = ranges.size

        while pos < ranges.max():
            idx = np.argwhere((ranges > pos) & (ranges < (pos + self.im_pxl_sze)))
            if not idx.size:
                idxs.append(None)
                wgts.append(None)
                continue
            wgt = diff_dist[idx]
            wgt[0] *= 1 - wgt_prev

            p_idx = np.max(idx)  # last index fully contained below current edge
            # p_idx = idx[-1]  # since the values are sorted, equivalent
            if p_idx < n_energies - 1:
                n_idx = p_idx + 1  # next index
                wgt_prev = (ranges[n_idx] - (pos + self.im_pxl_sze)) / diff_dist[n_idx]
            else:
                n_idx = []
                wgt_prev = []

            idxs.append(np.append(idx, n_idx).astype('int'))
            wgts.append(np.append(wgt, wgt_prev))
            pos += self.im_pxl_sze

        if debug:
            print("Bins generated: ", len(idxs))
            print("Idxs: ", idxs)
            print("Weights: ", wgts)
        return idxs, wgts

    def _generate_pos_basis(self, element='C', **kwargs):
        """Position folded list of weighted legendre coefficients. First item is closest to end of range"""
        indexes, wgts = self._pos_weights(**kwargs)  # kwargs = Debug
        bins = len(indexes)
        basis = [None] * bins
        # print("Indexes: ", indexes)
        # print("Weights: ", wgts)
        a0, a20, a40, a60 = np.zeros(4)

        for i, (bin_idxs, bin_wgts) in enumerate(zip(indexes, wgts)):
            if bin_idxs is None or bin_wgts is None:
                continue  # basis[i] = None
            if element.upper() == 'C':
                a0 = np.average(self.params['C_sig'][bin_idxs], weights=bin_wgts)
                a20 = np.average(self.params['C_a20'][bin_idxs], weights=bin_wgts)
                a40 = np.average(self.params['C_a40'][bin_idxs], weights=bin_wgts)
                a60 = 0
            if element.upper() == 'O':
                a0 = np.average(self.params['Oxy613_sig'][bin_idxs], weights=bin_wgts)
                a20 = np.average(self.params['Oxy613_a20'][bin_idxs], weights=bin_wgts)
                a40 = np.average(self.params['Oxy613_a40'][bin_idxs], weights=bin_wgts)
                a60 = np.average(self.params['Oxy613_a60'][bin_idxs], weights=bin_wgts)

            # Note, normalized to a0 term
            coeff = np.array([1, 0, a20, 0, a40, 0, a60])

            # basis[i] = np.polynomial.Legendre(self.leg_norms * coeff)
            # basis[i] = self.leg_norms * coeff  # len_norms not needed
            basis[i] = coeff

        return basis

    def fold_position_averaged(self, sysmat_fname, angle_fname,
                               x_pixels=201, element='C',
                               include_sin_term=False,
                               **kwargs):
        """Folds in energy averaged cross sections. Can fold in carbon and/or oxygen. Here carbon is 4.4 line,
        and oxygen is only 6.1 MeV. Image pixel size given in cm. Always normalized. *args is fed to folded_response.
        Must be sysmat_fname and angle_fname. **kwargs is Debug"""

        if element.upper() not in self.elements:
            ValueError("Element {e} not in allowed elements list: {a}".format(e=element.upper(), a=self.elements))

        save_name = 'p_avg' + element  # folded system response name

        pos_basis = self._generate_pos_basis(element=element, **kwargs)  # returns list of coeff.
        pb_length = len(pos_basis)

        s_file, s_table = load_response_table(sysmat_fname, name='sysmat')
        a_file, a_table = load_response_table(angle_fname, name='pt_angles')

        s = s_table.read()
        a = a_table.read()

        try:
            print("s_table.nrows: ", s_table.nrows)
        except Exception as e:
            print(e)

        s_file.close()
        a_file.close()

        assert s.shape == a.shape, "Angles table shape, {a}, and sysmat table shape, {s}," \
                                   " not the same".format(a=a.shape, s=s.shape)

        pxls, dets = s.shape

        geom = s.T.reshape([dets, pxls // x_pixels, x_pixels])
        angs = a.T.reshape([dets, pxls // x_pixels, x_pixels])
        tot = np.copy(geom)  # this keeps first pos basis x-axis values the same
        tot[:, :, pb_length:] = 0.0  # Need to empty this space

        b = np.polynomial.Legendre(np.array([1, 0, 0, 0, 0, 0, 0]))

        for position, coefficients in enumerate(pos_basis):
            # position is relative position of current basis point
            b.coef = coefficients
            if include_sin_term:
                tot[:, :, pb_length:] += geom[:, :, pb_length-position:x_pixels-position] \
                                         * b(np.cos(angs[:, :, pb_length-position:x_pixels-position])) /\
                                         np.sin(angs[:, :, pb_length-position:x_pixels-position])

            else:
                tot[:, :, pb_length:] += geom[:, :, pb_length - position:x_pixels - position] \
                                         * b(np.cos(angs[:, :, pb_length - position:x_pixels - position]))

        if include_sin_term:
            save_name += '_wSin'
        else:
            save_name += '_noSin'

        new_file = tables.open_file(save_name, mode="w", title="P Avg Folded System Response")
        folded_sysmat = new_file.create_earray('/', 'sysmat',
                                               atom=tables.atom.Float64Atom(),
                                               shape=(0, 48 * 48),
                                               expectedrows=s.shape[0])
        folded_sysmat.append(tot.transpose((1, 2, 0)).reshape(s.shape))
        new_file.flush()
        new_file.close()


def test_single_pt_eavg(col=101, row=31, dims=(201, 61), **kwargs):
    base_folder = '/Users/justinellin/Desktop/July_Work/current_sysmat/'
    sysmat_file = base_folder + '2021-07-03-1015_SP1.h5'
    angle_file = base_folder + 'angles_2021-07-19-1827.h5'
    kfile = '/Users/justinellin/repos/sysmat/july_basis/kernels.pkl'

    fold_gen = physics_basis(kfile_name=kfile)
    tot_legendre, _, _ = fold_gen.fold_energy_averaged(sysmat_file, angle_file, save=False, **kwargs)

    s_file, s_table = load_response_table(sysmat_file, name='sysmat')
    a_file, a_table = load_response_table(angle_file, name='pt_angles')

    dx, dy = dims
    idx = (row * dx) + col

    pt_geom = s_table[idx].reshape([48, 48])
    pt_angles = tot_legendre(np.cos(a_table[idx].reshape([48, 48])))
    tot = (s_table[idx] * tot_legendre(np.cos(a_table[idx]))).reshape([48, 48])
    # tot = pt_geom * pt_angles

    s_file.close()
    a_file.close()

    titles = ('geometry', 'emission', 'combined')
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 5),
                            subplot_kw={'xticks': [], 'yticks': []})

    for ax, proj, title in zip(axs, (pt_geom, pt_angles, tot), titles):
        img = ax.imshow(proj, cmap='magma', origin='upper', interpolation='nearest')
        ax.set_title(title)
        if title == 'emission':
            fig.colorbar(img, ax=ax, fraction=0.045, pad=0.04)

    plt.tight_layout()
    plt.show()


def test_single_pt_pavg(col=101, row=31, dims=(201, 61), **kwargs):
    base_folder = '/Users/justinellin/Desktop/July_Work/current_sysmat/'
    sysmat_file = base_folder + '2021-07-03-1015_SP1.h5'
    angle_file = base_folder + 'angles_2021-07-19-1827.h5'
    kfile = '/Users/justinellin/repos/sysmat/july_basis/kernels.pkl'

    fold_gen = physics_basis(kfile_name=kfile)
    pos_basis = fold_gen._generate_pos_basis(**kwargs)
    # pb_length = len(pos_basis)

    s_file, s_table = load_response_table(sysmat_file, name='sysmat')
    a_file, a_table = load_response_table(angle_file, name='pt_angles')

    s = s_table.read()
    a = a_table.read()

    assert s.shape == a.shape, "Angles table shape, {a}, and sysmat table shape, {s}," \
                               " not the same".format(a=a.shape, s=s.shape)

    dx, dy = dims
    idx = (row * dx) + col

    pt_original = s_table[idx].reshape([48, 48])  # i.e. position = 0, original basis
    tot = np.zeros_like(pt_original)

    b = np.polynomial.Legendre(np.array([1, 0, 0, 0, 0, 0, 0]))

    for position, coefficients in enumerate(pos_basis):
        # position is relative position of current basis point
        b.coef = coefficients
        pt_geom = s_table[idx-position].reshape([48, 48])
        pt_angles = a_table[idx-position].reshape([48, 48])

        tot += pt_geom * b(np.cos(pt_angles))

    s_file.close()
    a_file.close()

    titles = ('original', 'physics basis')
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 5),
                            subplot_kw={'xticks': [], 'yticks': []})

    for ax, proj, title in zip(axs, (pt_original, tot), titles):
        img = ax.imshow(proj, cmap='magma', origin='upper', interpolation='nearest')
        ax.set_title(title)
        # fig.colorbar(img, ax=ax, fraction=0.045, pad=0.04)

    plt.tight_layout()
    plt.show()


def main(pos_basis=True, **kwargs):
    base_folder = '/Users/justinellin/Desktop/July_Work/current_sysmat/'
    sysmat_file = base_folder + '2021-07-03-1015_SP1.h5'
    angle_file = base_folder + 'angles_2021-07-19-1827.h5'
    kfile = '/Users/justinellin/repos/sysmat/july_basis/kernels.pkl'

    fold_gen = physics_basis(kfile_name=kfile)
    if pos_basis:
        # fold_gen.fold_position_averaged(sysmat_file, angle_file,
        #                                x_pixels=201, element='C',
        #                                include_sin_term=False, debug=False)
        fold_gen.fold_position_averaged(sysmat_file, angle_file, **kwargs)
    else:
        fold_gen.fold_energy_averaged(sysmat_file, angle_file, **kwargs)
        # fold_energy_averaged(self, sysmat_fname, angle_fname,
        #                     x_pixels=201, element='C',
        #                     save=True, include_sin_term=False):


if __name__ == "__main__":
    pos_basis = True  # True = position basis, False = energy average basis
    element = 'C'
    main(pos_basis=pos_basis, element=element)
    # test_single_pt_eavg(col=101, row=31, dims=(201, 61), element=element)
    # test_single_pt_pavg(col=101, row=31, dims=(201, 61), element=element)
