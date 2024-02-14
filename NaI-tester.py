import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import math
import os
from csv import writer
import datetime

if not os.path.exists('Data/CAMIS-Detector-Testing/testing-results.csv'):
    with open ('Data/CAMIS-Detector-Testing/testing-results.csv','a') as csv_file:
        writer_obj = writer(csv_file)
        writer_obj.writerow(['NaI ID', 'PMT ID', 'Date Collected', 'Gross Counts', 'Net Counts', 'Cs137 Peak Ch', 'Cs137 FWHM [ch]'])

def gauss_new(x, A, x0, sigma):
    # Gaussian signal shape
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss_fit(x, y, var=False):
    # Fits gaussian and returns fit parameters
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss_new, x, y, p0=[max(y), mean, sigma])
    perr = np.sqrt(np.diag(pcov))
    if var:
        return popt, perr
    return popt

# Gathering info and starting test of connected detector
NaI_ID = int(input('\nWhat is the ID # on the NaI Detector?  '))
PMT_ID = int(input('What is the new ID # on the PMT Base?  '))

print('\n--------------------------------------------------------------------')
print('\nRemember to connect all 3 wires and apply voltage to the detector before continuing!')
_ = input('Press enter when ready to begin data collection.\n')
print('--------------------------------------------------------------------\n')

df_name = 'NaI-{}_PMT-{}_Test'.format(NaI_ID, PMT_ID)

run_cmd = ('python data_subscriber.py -f sample_configs/CAMIS_v2.json -i '
            + '192.168.0.3 -s raw_hdf5 -m 30 '
            + '-sf CAMIS-Detector-Testing/{df} >/dev/null 2>&1').format(df=df_name)

print('Collecting data from the detector now.')
os.system(run_cmd)
print('\n--------------------------------------------------------------------\n')

# Processing output file
f = h5py.File('Data/CAMIS-Detector-Testing/{}.h5'.format(df_name), 'r')

energies = f['event_data'][:]['en_max']
energies = energies[energies <= 7e4]

f.close()

bins, step = np.linspace(0, 7e4, 1001, retstep=True)
x_bins = bins[:-1]+(step/2)
counts, _ = np.histogram(energies, bins=bins)

peak_info = find_peaks(counts, prominence=100, width=5)

peak_num = -1
i_low = int(math.floor(peak_info[0][peak_num]-2*peak_info[1]['widths'][peak_num]))
i_high = int(math.ceil(peak_info[0][peak_num]+2*peak_info[1]['widths'][peak_num]))

gf = gauss_fit(x_bins[i_low:i_high], counts[i_low:i_high])

results_list = [NaI_ID, PMT_ID, datetime.datetime.now().strftime('%m-%d-%Y'),
                np.sum(counts[i_low:i_high]), np.sum(gauss_new(x_bins, *gf)), gf[1], 2*np.log(2)*gf[2]]

print('Writing results to file.')
with open('Data/CAMIS-Detector-Testing/testing-results.csv','a') as csv_file:
    writer_obj = writer(csv_file)
    writer_obj.writerow(results_list)
