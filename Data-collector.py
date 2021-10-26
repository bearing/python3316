import os
import time
import sys
import math

meas_dir_name = input('\nWhat do you want to title this data directory? (no spaces)  ')

if not os.path.exists('Data/'+meas_dir_name):
    os.makedirs('Data/'+meas_dir_name)
    print('    Directory "{}" created, writing into said directory'.format(meas_dir_name))
else:
    print('    Directory "{}" already exists, adding to said directory.'.format(meas_dir_name))

name_q = input('\nDo you want to name the data files something different than the directory name?  ').upper()
if name_q in ['YES', 'Y']:
    df_name = input('What do you want to name the data files? (no spaces)  ')
else:
    df_name = meas_dir_name

total_time = int(input('\nHow many minutes do you want to run for?  '))
ind_rt = int(input('How long should each measurement be? (in seconds)  '))
#individual_run_time = 60
total_itr = (total_time*60)/ind_rt
if not total_itr.is_integer():
    print('    Cannot evenly split {} minutes into {} second intervals'.format(total_time, ind_rt))
    total_itr = math.ceil(total_itr)
    print('    Will instead run for {} minutes and {} seconds'.format((total_itr*ind_rt)//60, (total_itr*ind_rt)%60))

if ind_rt >= 60:
    if ind_rt%60 == 0:
        time_name = '{}min'.format(ind_rt//60)
    else:
        time_name = '{}min{}sec'.format(ind_rt//60, ind_rt%60)
else:
    time_name = '{}sec'.format(ind_rt)

print('\nThe files will be named under the format:' +
        '\n    "{}/{}-{}_(measurement number).h5"'.format(meas_dir_name, df_name, time_name))

run_cmd = ('python data_subscriber.py -f sample_configs/CAMIS.json -i '
            + '192.168.1.2 192.168.1.3 192.168.1.4 192.168.1.5 192.168.1.6 '
            + '192.168.1.7 192.168.1.8 192.168.1.9 -s raw_hdf5 -g 10 -m {ind_time} '
            + '-sf {dir}/{df}-{time}_{{meas_num}} >/dev/null 2>&1').format(
            ind_time=ind_rt, dir=meas_dir_name, df=df_name, time=time_name)

files = os.listdir('Data/'+meas_dir_name)
number_ran = []
for file in files:
    number_ran.append(int(file.split('.')[0].split('_')[-1]))

if not number_ran:
    starting_num = 0
else:
    starting_num = max(number_ran)
    print('\n{} minutes of data has already been collected.'.format(starting_num) +
            '\nCompleting this will make for a total of {} minutes of data'.format(starting_num+total_itr))

lq = input('\nIf an instance of data collection fails,\nwould you like to continuously try for the whole run time?  ').upper()
if lq in ['YES', 'Y']:
    print('    Lazy option enabled :)')
    laziness_var = True
else:
    laziness_var = False

data_cont = input('\nWould you like to proceed with data collection?  ').upper()
if data_cont not in ['YES', 'Y']:
    sys.exit()

print('\n')
trying = True
current_num = 1
fails = 0
while trying:
    try:
        if current_num > total_itr:
            trying = False
            print('----------------------')
            if (total_itr*ind_rt)%60 == 0:
                print('Finished running {} minutes of data!'.format(total_time))
            else:
                print('Finshed running {} minutes and {} seconds of data!'.format((total_itr*ind_rt)//60, (total_itr*ind_rt)%60))
            break
        print('----------------------')
        print('Currently running interval #{}'.format(current_num))
        start_time = time.time()
        #os.system(run_cmd.format(individual_run_time, measurement_loc, individual_run_time//60, current_num+starting_num))
        os.system(run_cmd.format(meas_num=current_num+starting_num))
        run_time = time.time() - start_time
        if run_time < ind_rt:
            raise ValueError
        current_num += 1
    except:
        print('\033[31m----------------------\033[39m')
        print('\033[31mCurrent run has failed due to:\n3316 Card Buffer Issue\033[39m')
        if laziness_var:
            print('\033[33mAutomatically re-trying\033[39m')
            fails += 1
        else:
            error_inp = input('\033[33mWould you like to try again?  \033[39m').upper()

            if error_inp not in ['Y', 'YES']:
                trying = False
                print('----------------------')
                print('Stopping after completing {} out of {} intervals :('.format(current_num-1, total_itr))
                os.remove('Data/{}/{}-{}_{}.h5'.format(meas_dir_name, df_name, time_name, current_num+starting_num))
                break

if fails != 0:
    print('Choosing the lazy option saved you sitting here and trying running again {} times'.format(fails))
