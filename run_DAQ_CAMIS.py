import os
import sys
from datetime import datetime
import argparse
import multiprocessing

# Numbers written at the top of the 3316 cards
card_nums = [14, 2, 4, 6, 7, 8, 10, 12]

def start_3316_DAQ(cmd):
    os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-f', type=str, help='Name to call saved files', default=None)
    parser.add_argument('--gui', '-g', action='store_true', help='GUI activation', default=False)
    parser.add_argument('--measurement_time', '-m', type=int, help='# of seconds to collect data for in each iteration', default=30)
    parser.add_argument('--continuous_run', '-r', action='store_true', help='Toggles running continuously', default=False)
    parser.add_argument('--verbose', '-v', action='store_true', default=False)
    parser.add_argument('--print_output', '-p', action='store_true', help='Toggles printing of terminal output from data_subscriber', default=False)
    parser.add_argument('--save_raw_waveforms', '-w', action='store_true', help='Toggles saving of raw waveforms', default=False)

    args = parser.parse_args()
    arg_dict = vars(args)

    if arg_dict['verbose']:
        print('       Save Filename: {}'.format(arg_dict['filename']))
        print('    Measurement Time: {}'.format(arg_dict['measurement_time']))
        print('Saving Raw Waveforms: {}'.format(arg_dict['save_raw_waveforms']))
        print('  Continuous Running: {}'.format(arg_dict['continuous_run']))
        print('    Printing Outputs: {}'.format(arg_dict['print_output']))
        print('            GUI Mode: {}'.format(arg_dict['gui']))
        print('----------------------------------------------')
    del arg_dict['verbose']

    base_cmd = 'python DAQ_individual_runner.py -c {{cn}} -m {mt}'.format(mt=arg_dict['measurement_time'])

    if arg_dict['filename'] is not None:
        base_cmd = base_cmd + ' -f {fn}'.format(fn=arg_dict['filename'])
    if arg_dict['gui']:
        base_cmd = base_cmd + ' -g'
    if arg_dict['continuous_run']:
        base_cmd = base_cmd + ' -r'
    if arg_dict['print_output']:
        base_cmd = base_cmd + ' -p'
    if arg_dict['save_raw_waveforms']:
        base_cmd = base_cmd + ' -w'

    DAQ_0 = multiprocessing.Process(target=start_3316_DAQ, args=(base_cmd.format(cn=card_nums[0]),))
    DAQ_1 = multiprocessing.Process(target=start_3316_DAQ, args=(base_cmd.format(cn=card_nums[1]),))
    DAQ_2 = multiprocessing.Process(target=start_3316_DAQ, args=(base_cmd.format(cn=card_nums[2]),))
    DAQ_3 = multiprocessing.Process(target=start_3316_DAQ, args=(base_cmd.format(cn=card_nums[3]),))
    DAQ_4 = multiprocessing.Process(target=start_3316_DAQ, args=(base_cmd.format(cn=card_nums[4]),))
    DAQ_5 = multiprocessing.Process(target=start_3316_DAQ, args=(base_cmd.format(cn=card_nums[5]),))
    DAQ_6 = multiprocessing.Process(target=start_3316_DAQ, args=(base_cmd.format(cn=card_nums[6]),))
    DAQ_7 = multiprocessing.Process(target=start_3316_DAQ, args=(base_cmd.format(cn=card_nums[7]),))

    DAQ_processes = [DAQ_0, DAQ_1, DAQ_2, DAQ_3, DAQ_4, DAQ_5, DAQ_6, DAQ_7]

    try:
        for dq, DAQ_process in enumerate(DAQ_processes):
            print('Starting Card {}'.format(card_nums[dq]))
            DAQ_process.start()
        print('\033[32m\033[1mStarted all 8 cards!\033[0m')

        for DAQ_process in DAQ_processes:
            DAQ_process.join()
        print('All processes joined')
    except:
        pass
