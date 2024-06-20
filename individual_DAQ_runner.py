import os
import sys
from datetime import datetime
import argparse
now = datetime.now()

ips = {1: 3, # Card isn't used anymore but remnants remain
       14: 3,
       2: 4,
       4: 5,
       6: 6,
       7: 7,
       8: 8,
       10: 9,
       12: 10}

def make_data_dir(card_num):
    try:
        os.mkdir('Data/CAMIS-Data/')
    except:
        pass

    try:
        os.mkdir('Data/CAMIS-Data/{}-{}-{}'.format(now.month, now.day, str(now.year)))
    except:
        pass

    try:
        os.mkdir('Data/CAMIS-Data/{}-{}-{}/Card-{}'.format(now.month, now.day, str(now.year), card_num))
    except:
        pass

    data_dir = 'CAMIS-Data/{}-{}-{}/Card-{}'.format(now.month, now.day, str(now.year), card_num)

    return data_dir

def run_DAQ(card_num, data_dir, filename, gui, measurement_time, continuous_run, print_output, save_raw_waveforms, verbose):
    cmd = 'python data_subscriber.py -i 192.168.0.{ip} -s raw_hdf5 -m {mt} '.format(ip=ips[card_num], mt=measurement_time) + \
          '-f sample_configs/CAMIS.json -sf {dd}/{fn}'.format(dd=data_dir, fn=filename)

    if save_raw_waveforms:
        cmd = cmd + ' --save_raw_waveforms'
    if gui:
        cmd = cmd + ' --gui'
    if continuous_run:
        cmd = cmd + ' --continuous'
    if not print_output:
        cmd = cmd + ' >/dev/null 2>&1'
        if verbose:
            print('DAQ will be run while suppressing prints.')

    os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--card_num', '-c', type=int, help='Number written on top of DAQ card', default=None)
    parser.add_argument('--filename', '-f', type=str, help='Name to call saved files', default=None)
    parser.add_argument('--gui', '-g', action='store_true', help='GUI activation', default=False)
    parser.add_argument('--measurement_time', '-m', type=int, help='# of seconds to collect data for in each iteration', default=30)
    parser.add_argument('--save_raw_waveforms', '-w', action='store_true', help='Toggles saving of raw waveforms', default=False)
    parser.add_argument('--continuous_run', '-r', action='store_true', help='Toggles running continuously', default=False)
    parser.add_argument('--verbose', '-v', action='store_true', default=False)
    parser.add_argument('--print_output', '-p', action='store_true', help='Toggles printing of terminal output from data_subscriber', default=False)

    args = parser.parse_args()
    arg_dict = vars(args)

    if arg_dict['card_num'] is None:
        print("Warning: Card number is required for operation")
        print('Use "python individual_DAQ_runner.py -c CARD_NUM"')
        sys.exit()

    data_dir = make_data_dir(arg_dict['card_num'])
    arg_dict['data_dir'] = data_dir

    if arg_dict['filename'] is None:
        arg_dict['filename'] = '{}_sec-DataRun_{}'.format(arg_dict['measurement_time'], len([f for f in os.listdir('Data/'+data_dir) if f.endswith('.h5')])+1)

    if arg_dict['verbose']:
        print('     DAQ Card Number: {}'.format(arg_dict['card_num']))
        print('      Data directory: {}'.format(arg_dict['data_dir']))
        print('       Save Filename: {}'.format(arg_dict['filename']))
        print('    Measurement Time: {}'.format(arg_dict['measurement_time']))
        print('Saving Raw Waveforms: {}'.format(arg_dict['save_raw_waveforms']))
        print('  Continuous Running: {}'.format(arg_dict['continuous_run']))
        print('    Printing Outputs: {}'.format(arg_dict['print_output']))
        print('            GUI Mode: {}'.format(arg_dict['gui']))
        print('----------------------------------------------')

        print('Running DAQ with given settings now.')

    run_DAQ(**arg_dict)
