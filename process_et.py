'''

Python script for the pre-processing of tandem eye tracking data
for sEEG attentional testing in children

Modified to work with Closed-loop neurostimulation data to test
effects of intracranial stim on eye movement patterns

Nebras M. Warsi, Ibrahim Lab
April 2022


'''

# Setup
import numpy as np
import os
import h5py

# Custom functions
from datafuncs import *
from patient_list import patient_info

# Path definitions
data_path = ""
et_path = ""


######################
#
# Main preproc code
#
##


def load_subjs(subj, load_path, jf_path, save_path):

    if not os.path.isdir(save_path): os.makedirs(save_path) # Create save path if it doesn't exist

    # Select the appropriate trigger definition file
    sub_num = int(subj.split('-')[-1])
    trigpath = '/d/gmi/1/nebraswarsi/CLES/scripts'

    trigdef = os.path.join(trigpath, 'SetShifting_CLeS.json') if sub_num >= 26 else os.path.join(trigpath, 'SetShifting_original.json')

    # Specify relevant eye tracker info
    et_trials = os.path.join(et_path, subj, 'ET/%s_lslCLES.csv' % subj)
    et_data = os.path.join(et_path, subj, 'ET/%s_etCLES.csv' % subj)

    # Load data and synch trials between EEG and eye tracker
    shifts, X, Y, _, _, intent, stim,\
        eyepos, eyefix, eyesacc, pupdiam, L_val, R_val, choice, \
            gazeT, gazeB, gazeOT, ttf, tot, corr, contacts, _, \
                coords, regions, YEO = load_data(load_path, jf_path, save_path, et_trials, et_data, trigdef)

    # Identify correct trials
    cidxs = np.squeeze(np.nonzero((np.asarray(corr) > 0)))

    # Check trials align
    assert len(Y) == X.shape[0], "Error! Trials do not align. Please check data\nand JSON start/end indices"

    print('\nSuccessly loaded data for %s!\n' % subj)
    print('EEG data: %s' % str(X.shape))
    print('Shift trials: %s' % str(np.sum(shifts)))
    print('Eyepos data: %s' % str(eyepos.shape))
    print('Pupil data: %s' % str(pupdiam.shape))
    print('RT data: %s' % str(np.asarray(Y).shape))

    # Now we will save our data to an H5PY file
    h5py_file = h5py.File(os.path.join(save_path, "processed_data.h5"), 'w')

    h5py_file.create_dataset('X', data=X)
    h5py_file.create_dataset('Y', data=Y)
    h5py_file.create_dataset('shifts', data=shifts)
    h5py_file.create_dataset('intent', data=intent)
    h5py_file.create_dataset('stim', data=stim)
    h5py_file.create_dataset('cidxs', data=cidxs)
    h5py_file.create_dataset('eyepos', data=eyepos)
    h5py_file.create_dataset('eyefix', data=eyefix)
    h5py_file.create_dataset('eyesacc', data=eyesacc)
    h5py_file.create_dataset('pupdiam', data=pupdiam)
    h5py_file.create_dataset('L_val', data=L_val)
    h5py_file.create_dataset('R_val', data=R_val)
    h5py_file.create_dataset('choice', data=choice)
    h5py_file.create_dataset('gazeT', data=gazeT)
    h5py_file.create_dataset('gazeB', data=gazeB)
    h5py_file.create_dataset('gazeOT', data=gazeOT)
    h5py_file.create_dataset('ttf', data=ttf)
    h5py_file.create_dataset('tot', data=tot)

    h5py_file.attrs['contacts'] = contacts
    h5py_file.attrs['coords'] = coords
    h5py_file.attrs['regions'] = regions
    h5py_file.attrs['YEO_networks'] = YEO

    h5py_file.close()

    print('*'*50)
    print('\n***Data Saved***\n')
    print('*'*50)


# Runs the script

if __name__ == "__main__":


    print()
    print('#' * 50)
    print('#' * 50)
    print('\nWelcome!')
    print('\nThis script will process tandem eye tracking and intracranial stim data!')
    print('\nEnjoy!\n')
    print('#' * 50)
    print('#' * 50)

    subj = [x for x in patient_info]
    load_paths = [os.path.join(data_path, x) for x in patient_info]
    jfs = [[y for y in patient_info.get(x)['load']] for x in patient_info]
    save_paths = [[os.path.join(et_path, x, 'ET', y) for y in patient_info.get(x)['save']] for x in patient_info]


    for subj_data in zip(subj, load_paths, jfs, np.concatenate(save_paths)):

        print('\nProcessing data for %s...' % subj_data[0])

        load_subjs(subj_data[0], subj_data[1], subj_data[2], subj_data[3])

    print('Done!')