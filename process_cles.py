'''
Unified data processing script for machine learning classifiers 
Prediction of attentional lapses in real-time

Preprocess data from EDF into ML-ready h5py format


Nebras M. Warsi
George M. Ibrahim Lab
Dec 2021

'''


# DSP and data imports 
import os
import numpy as np
import h5py
from scipy.signal import welch
# Stats
from scipy import stats
from scipy.integrate import simpson
# Plotting
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import seaborn as sns
# Useful functions
from cles_cohort import patient_info
from datafuncs import *
import gc
matplotlib.use('Agg')

# Script settings and paths
check_filt = True
plot = True
rt_mode = 'ind_day' # Sets the RT data to use for classification threshold
data_path = ""
out_path = ""

# DSP Params
Fs = 2048
start = int(0*Fs)
stop = int(2.0*Fs) ##### Collects 2 seconds pre-stimulus data


def plot_graphs(tt, fast, slow, coords, graph_dir):

    graph_dir = os.path.join(graph_dir, 'PSD')
    if not os.path.isdir(graph_dir): os.makedirs(graph_dir)

    # Create Figure
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    axs = axs.ravel()

    # Fast 
    average_fast = np.mean(fast, axis=0)

    im1 = axs[0].imshow(average_fast, origin='lower', cmap='jet', 
                            aspect='auto', interpolation='none',
                            norm=LogNorm(np.mean(average_fast), vmax=0.75*np.amax(average_fast)))
    axs[0].set_title('Fast Trials')


    # Slow 
    average_slow = np.mean(slow, axis=0)
    im2 = axs[1].imshow(average_slow, origin='lower', cmap='jet', 
                            aspect='auto', interpolation='none',
                            norm=LogNorm(vmin=np.mean(average_fast), vmax=0.75*np.amax(average_fast)))
    axs[1].set_title('Slow Trials')


    # Slow v Fast
    average_svf = np.subtract(average_slow, average_fast)
    im3 = axs[2].imshow(average_svf, origin='lower', cmap='seismic', aspect='auto', 
                                            interpolation='none', vmin=-2.5, vmax=2.5)
    axs[2].set_title('Slow rel. to fast')


    ims = [im1, im2, im3]

    # colorbars
    for aidx, ax in enumerate(axs):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='5%', pad=0.5)

        label = '% Total Power'
        cb_morl = plt.colorbar(ims[aidx], cax=cax, label=label, orientation="horizontal")
        cb_morl.outline.set_visible(False)

    figname = os.path.join(graph_dir, '%s_PSD.png' % (tt))
    fig.suptitle('%s PSD pre-stimulus' % (tt.capitalize()))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(figname, dpi=300, bbox_inches="tight")
    plt.close()


def calc_psd(X):

    PSDfreqs, dataPSD = welch(X, fs=Fs, nperseg=512, nfft=Fs)
    freq_range = np.where((PSDfreqs >= 4) & (PSDfreqs <=43))[0]

    # Calculate percentage of total power in each band (baselining)
    dataPSD = 100 * (dataPSD[:, :, freq_range] / simpson(dataPSD[:, :, freq_range])[:,:,None])

    return dataPSD


def RT_plots(Y_shift, Y_ns, Y_class_s, Y_class_ns, save_path):

    # We will also test for normality of RT distribution (likely LogNormal)
    _, shift_shapiro = stats.shapiro(Y_shift)
    _, nonshift_shapiro = stats.shapiro(Y_ns)

    # We will plot the reaction time distributions for assessment
    sns.boxplot(y=Y_shift)
    plt.ylabel('RT')
    plt.savefig(os.path.join(save_path, 'shift_RT.png'))
    plt.close()

    sns.boxplot(y=Y_ns)
    plt.ylabel('RT')
    plt.savefig(os.path.join(save_path, 'non_shift_RT.png'))
    plt.close()

    # Also plots n trials in each class bin
    sns.countplot(x=Y_class_s)
    plt.savefig(os.path.join(save_path, 'shift_RT_class.png'))
    plt.close()

    sns.displot(x=Y_class_ns)
    plt.savefig(os.path.join(save_path, 'non_shift_RT_class.png'))
    plt.close()

    # Finally, we also plot RT as a function of trial number
    shift_movmean = pd.Series(Y_shift).rolling(20).mean()
    ns_movmean = pd.Series(Y_ns).rolling(20).mean()

    plt.plot(shift_movmean)
    plt.savefig(os.path.join(save_path, 'shift_RT_by_trial.png'))
    plt.close()

    plt.plot(ns_movmean)
    plt.savefig(os.path.join(save_path, 'non_shift_RT_by_trial.png'))
    plt.close()


def process_raw(subj, load_path, jf_path, save_path):

    print('\nLoading data from EDF file in\n% s...\n' % jf_path)

    # Instaniate output path
    if not os.path.isdir(save_path): os.makedirs(save_path)

    # Select the appropriate trigger definition file
    sub_num = int(subj.split('-')[-1])
    trigpath = ''

    # Subject numbers removed for public access
    trigdef = os.path.join(trigpath, 'SetShifting_CLeS.json') if ((sub_num >= XX) & (sub_num != YY)) else os.path.join(trigpath, 'SetShifting_original.json')

    # Loads data 
    shifts, X_full, Y, _, _,\
        corr, contacts, _,\
            coords, regions, YEO = load_from_edf(load_path, jf_path, save_path, trigdef)

    # X is a 3D array formatted as follows: [Trials, contacts, timeseries]
    # shifts, X_full, Y = np.asarray(shifts)[cidxs], np.asarray(X_full)[cidxs], np.asarray(Y)[cidxs]
    t_original = np.linspace(-2, 2, int(Fs*4) + 1)

    # Check trials align
    assert len(Y) == X_full.shape[0], "Error! Trials do not align. Please check data\nand JSON start/end indices"

    print('Success!\n')
    print('EEG data: %s' % str(X_full.shape))
    print('RT data: %s' % str(np.asarray(Y).shape))
    
    # Plot to check this works without artefacts
    if check_filt:
        for ch in range(X_full.shape[1]):
            plt.plot(t_original, np.mean(X_full[:, ch, :], axis=0), '.-')
            plt.legend(['data', 'resampled'], loc='best')
            graph_path = os.path.join(save_path, 'eeg_preproc/')
            if not os.path.isdir(graph_path): os.makedirs(graph_path)
            plt.savefig(os.path.join(graph_path, '%s.png' % contacts[ch]))
            plt.close()

    # Here we select the pre-stimulus data only
    X = X_full[:, :, start:stop]

    # We will also split fast and slow here 
    rt_paths = patient_info.get(subj)['load']
    Y_class, Y_class_s, Y_class_ns = class_data(Y, load_path, rt_paths, save_path, trigdef, shifts, mode=rt_mode)

    print('\nComputing PSD...\n')

    # Calculate pre-stimulus PSD as main feature for ML
    psd = calc_psd(X)

    # Split shift and non-shift for RT plots
    _, Y_shift, _, Y_ns = split_shift_nonshift(X, Y, shifts)

    # Make RT plots
    RT_plots(Y_shift, Y_ns, Y_class_s, Y_class_ns, save_path)

    # final output data
    X_shift = []
    X_shift_fast = []
    X_shift_slow = []
    X_ns = []
    X_ns_fast = []
    X_ns_slow = []
    X_shift_raw = []
    X_ns_raw = []

    psd_shift = []
    psd_shift_fast = []
    psd_shift_slow = []
    psd_ns = []
    psd_ns_fast = []
    psd_ns_slow = []

    for trl in range(len(shifts)):
        
        if shifts[trl]:

            X_shift.append(X[trl, :, :])
            X_shift_raw.append(X_full[trl, :, :])
            psd_shift.append(psd[trl, :, :])

            if (Y_class[trl]):

                X_shift_slow.append(X[trl, :, :])
                psd_shift_slow.append(psd[trl, :, :])

            else:

                X_shift_fast.append(X[trl, :, :])
                psd_shift_fast.append(psd[trl, :, :])
        
        else:

            X_ns.append(X[trl, :, :])
            X_ns_raw.append(X_full[trl, :, :])
            psd_ns.append(psd[trl, :, :])

            if (Y_class[trl]):

                X_ns_slow.append(X[trl, :, :])
                psd_ns_slow.append(psd[trl, :, :])

            else:

                X_ns_fast.append(X[trl, :, :])
                psd_ns_fast.append(psd[trl, :, :])

    # Convert data to np arrays
    X_shift = np.asarray(X_shift)
    X_shift_fast = np.asarray(X_shift_fast)
    X_shift_slow = np.asarray(X_shift_slow)
    X_ns = np.asarray(X_ns)
    X_ns_fast = np.asarray(X_ns_fast)
    X_ns_slow = np.asarray(X_ns_slow)
    X_shift_raw = np.asarray(X_shift_raw)
    X_ns_raw = np.asarray(X_ns_raw)

    psd_shift = np.asarray(psd_shift)
    psd_shift_fast = np.asarray(psd_shift_fast)
    psd_shift_slow = np.asarray(psd_shift_slow)
    psd_ns = np.asarray(psd_ns)
    psd_ns_fast = np.asarray(psd_ns_fast)
    psd_ns_slow = np.asarray(psd_ns_slow)

    # Plotting
    if plot:

        graph_dir = os.path.join(save_path, 'graphs')
        if not os.path.isdir(graph_dir): os.makedirs(graph_dir)

        ####
        # Figures for fast and slow trials as well as relative difference
        ####

        # Figures
        for tt in ['shift', 'nonshift']:

            fast = psd_shift_fast if tt == 'shift' else psd_ns_fast
            slow = psd_shift_slow if tt == 'shift' else psd_ns_slow

            plot_graphs(tt, fast, slow, coords, graph_dir)

    # Now we will save our data to an H5PY file
    h5py_file = h5py.File(os.path.join(save_path, "processed_data.h5"), 'w')

    h5py_file.create_dataset('X', data=X)
    h5py_file.create_dataset('X_shift', data=X_shift)
    h5py_file.create_dataset('X_ns', data=X_ns)
    h5py_file.create_dataset('X_shift_raw', data=X_shift_raw) ### These two are to select STIM channel based on POST-stim data, NOT for ML model
    h5py_file.create_dataset('X_ns_raw', data=X_ns_raw)
    h5py_file.create_dataset('psd', data=psd)
    h5py_file.create_dataset('psd_shift', data=psd_shift)
    h5py_file.create_dataset('psd_nonshift', data=psd_ns)

    h5py_file.create_dataset('Y', data=Y)
    h5py_file.create_dataset('Y_shift', data=Y_shift)
    h5py_file.create_dataset('Y_ns', data=Y_ns)
    h5py_file.create_dataset('Y_class', data=Y_class)
    h5py_file.create_dataset('Y_class_s', data=Y_class_s)
    h5py_file.create_dataset('Y_class_ns', data=Y_class_ns)

    h5py_file.create_dataset('shifts', data=shifts)

    h5py_file.attrs['contacts'] = contacts
    h5py_file.attrs['coords'] = coords
    h5py_file.attrs['regions'] = regions
    h5py_file.attrs['YEO_networks'] = YEO

    h5py_file.close()

    print('*'*50)
    print('\n***Data Saved***\n')
    print('*'*50)

    gc.collect()




if __name__ == '__main__':

    print()
    print('#' * 50)
    print('#' * 50)
    print('\nWelcome!')
    print('\nThis script will read raw EDFs and \npreprocess sEEG data into ML compatible formats')
    print('\nEnjoy!\n')
    print('#' * 50)
    print('#' * 50)

    subj = [[x for _ in patient_info.get(x)['load']] for x in patient_info]
    load_paths = [[os.path.join(data_path, x) for _ in patient_info.get(x)['load']] for x in patient_info]
    jfs = [[y for y in patient_info.get(x)['load']] for x in patient_info]
    save_paths = [[os.path.join(out_path, x, y) for y in patient_info.get(x)['save']] for x in patient_info]

    for subj_data in zip(np.concatenate(subj), np.concatenate(load_paths), np.concatenate(jfs), np.concatenate(save_paths)):
        process_raw(subj_data[0], subj_data[1], subj_data[2], subj_data[3])