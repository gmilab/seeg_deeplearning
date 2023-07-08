
'''

Data functions to preprocess ML iEEG data including to read from EDF files and load data directly

'''


# General
import os
import numpy as np
import pandas as pd
import re
import json
# DSP
import mne
from scipy import signal
from scipy.ndimage import gaussian_filter
# General
import matplotlib.pyplot as plt
import itertools
from atlasreader.atlasreader import read_atlas_peak
import subprocess, shlex


def readTrigs(trigdef, trigs, shift_trig, ns_trig, correct_trig, incorrect_trig, Fs):

    triggers, times = trigs['val'].values, trigs['idx'].values

    # Get RT and correct/incorrect

    Y, Y_shift, Y_ns = [], [], []  # Store RT
    corr, corr_shift, corr_ns = [], [], []  # Store correct vs. incorrent
    shifts = []  # Store whether trial was shift or not
    idxs, shift_idxs, ns_idxs = [], [], []  # Stores the time index of the trial

    # Now we loop over triggers to read file and
    # determine trial order based on which Presentation file they had

    if '_original' in trigdef:

        for tidx, trig in enumerate(triggers):

            if ((trig in shift_trig) | (trig in ns_trig)):

                stim = times[tidx]
                if stim in idxs:
                    continue  # Skip trial duplicates

                for ti, t in enumerate(triggers[tidx + 1:]):

                    if times[tidx + 1:][ti] == stim:  # Skip duplicate timepoints

                        continue

                    # If we reach the next trial, this trial was a miss (no RT)
                    elif ((t in shift_trig) | (t in ns_trig)):

                        break

                    elif t == correct_trig:

                        rt = round((times[tidx + 1:][ti] - stim) / Fs * 1000)

                        # These are likely to be spurious as the ISI is <4 sec and 150ms is too fast
                        if ((rt > 5000) | (rt < 150)):

                            break

                        Y.append(rt)
                        corr.append(1)
                        idxs.append(stim)

                        # Check if ANY triggers at this timepoint were shift triggers
                        # (Required for original presentation setup as multiple trigs came at once;
                        # not needed for CLeS)

                        if len(np.intersect1d(triggers[np.where(times == stim)[0]], shift_trig)) > 0:

                            shifts.append(1)
                            Y_shift.append(rt)
                            corr_shift.append(1)
                            shift_idxs.append(stim)

                        else:

                            shifts.append(0)
                            Y_ns.append(rt)
                            corr_ns.append(1)
                            ns_idxs.append(stim)

                        break

                    elif t == incorrect_trig:

                        rt = round((times[tidx + 1:][ti] - stim) / Fs * 1000)

                        if ((rt > 5000) | (rt < 150)):

                            break

                        Y.append(rt)
                        corr.append(0)
                        idxs.append(stim)

                        if len(np.intersect1d(triggers[np.where(times == stim)[0]], shift_trig)) > 0:

                            shifts.append(1)
                            Y_shift.append(rt)
                            corr_shift.append(0)
                            shift_idxs.append(stim)

                        else:

                            shifts.append(0)
                            Y_ns.append(rt)
                            corr_ns.append(0)
                            ns_idxs.append(stim)

                        break

    else:  # CLeS file

        for tidx, trig in enumerate(triggers):

            if ((trig == shift_trig) | (trig == ns_trig)):

                stim = times[tidx]
                if stim in idxs:
                    continue  # Skip trial duplicates

                for ti, t in enumerate(triggers[tidx + 1:]):

                    if times[tidx + 1:][ti] == stim:  # Skip duplicate timepoints

                        continue

                    # If we reach the next trial, this trial was a miss (no RT)
                    elif ((t == shift_trig) | (t == ns_trig)):

                        break

                    elif t == correct_trig:

                        rt = round((times[tidx + 1:][ti] - stim) / Fs * 1000)

                        # These are likely to be spurious as the ISI is <4 sec and 150ms is too fast
                        if ((rt > 5000) | (rt < 150)):

                            break

                        Y.append(rt)
                        corr.append(1)
                        idxs.append(stim)

                        # Check if ANY triggers at this timepoint were shift triggers
                        # (Required for original presentation setup as multiple trigs came at once;
                        # not needed for CLeS)

                        if len(np.intersect1d(triggers[np.where(times == stim)[0]], shift_trig)) > 0:

                            shifts.append(1)
                            Y_shift.append(rt)
                            corr_shift.append(1)
                            shift_idxs.append(stim)

                        else:

                            shifts.append(0)
                            Y_ns.append(rt)
                            corr_ns.append(1)
                            ns_idxs.append(stim)

                        break

                    elif t == incorrect_trig:

                        rt = round((times[tidx + 1:][ti] - stim) / Fs * 1000)

                        if ((rt > 5000) | (rt < 150)):

                            break

                        Y.append(rt)
                        corr.append(0)
                        idxs.append(stim)

                        if len(np.intersect1d(triggers[np.where(times == stim)[0]], shift_trig)) > 0:

                            shifts.append(1)
                            Y_shift.append(rt)
                            corr_shift.append(0)
                            shift_idxs.append(stim)

                        else:

                            shifts.append(0)
                            Y_ns.append(rt)
                            corr_ns.append(0)
                            ns_idxs.append(stim)

                        break

    # Double check to make sure no duplicates
    dups = np.intersect1d(shift_idxs, ns_idxs)

    if len(dups) == 0:
        print('\nTriggers read successfully!')

    else:
        print(dups)
        raise Exception('Erorr. Duplicate trials identified.\n \
                            Please check JSON indices: %s' % dups)

    return Y, Y_shift, Y_ns, corr, shifts, idxs, shift_idxs, ns_idxs


def get_mapping(coords, contacts, path):

    print("\n\nGenerating channel anatomical labels... \n")

    regions, YEO = [], []

    for c in coords:

        if np.isnan(c).any():

            regions.append('')
            YEO.append('')

        else:

            # AAL labels
            reg = read_atlas_peak('aal', c)

            if reg == 'no_label':
                regions.append('')
            else:
                regions.append(reg)

            # YEO labels
            bsh = shlex.split("atlasquery -a 'YeoBuckner7' -c %.2f,%.2f,%.2f"
                              % (c[0], c[1], c[2]))
            yl = str(subprocess.check_output(bsh))

            if 'No label found!' in yl:
                YEO.append('')
            else:
                YEO.append(yl.split('% ')[1].split('\\n')[0].split(',')[0])

    print('Done!')

    # Output contact mapping CSV file for review
    cmap = pd.DataFrame()
    cmap['Contacts'] = contacts
    cmap['AAL'] = regions
    cmap['Yeo'] = YEO
    cmap['Coords'] = coords

    # Save
    cmap.to_csv(os.path.join(path, 'contact_mapping.csv'))

    return regions, YEO


def load_from_edf(path, jf, out, trigdef, lf=4,
                  mode='edf', montage='bipolar'):

    # We begin by loading the relevant patient info from the JSON file specified by user
    json_file = os.path.join(path, jf)
    json_file = open(json_file)
    json_data = json.load(json_file)

    # Load start and end times for set shift
    ss = json_data.get('sample_start')
    se = json_data.get('sample_end')

    # Include additional meta files if needed
    if "include" in json_data:

        jf2 = json.load(open(os.path.join(path, json_data['include'])))
        json_data.update(jf2)

    # Path to the EDF
    edf_file = json_data.get('filename')

    # Loads sEEG data through the EDF
    raw_data = mne.io.read_raw_edf(os.path.join(
        path, edf_file), verbose=False, preload=True)
    Fs = raw_data.info['sfreq']

    # Now we can load relevant trigger and channel data for the task
    trigs = pd.read_csv(os.path.join(path, json_data.get('triggers')))
    trig_map = open(trigdef)
    trig_map = json.load(trig_map)

    # Load based on which file version we used for the subject
    if '_original' in trigdef:

        shift_trig = [trig_map.get('ExtraDim'), trig_map.get('IntraDim')]
        ns_trig = [trig_map.get('ColorRule'), trig_map.get('ShapeRule')]

    else:

        shift_trig = trig_map.get('Shift')
        ns_trig = trig_map.get('NoShift')

    correct_trig = trig_map.get('Correct')
    incorrect_trig = trig_map.get('Incorrect')

    # Only triggers for the session of interest are included
    trigs = trigs[(trigs['idx'] >= ss) & (trigs['idx'] <= se)]

    # Read triggers to define trials and calculate RT
    Y, Y_shift, Y_ns, corr, shifts, _, shift_idxs, ns_idxs = \
        readTrigs(trigdef, trigs, shift_trig, ns_trig,
                  correct_trig, incorrect_trig, Fs)

    # This mode means we only need to load RT data from raw files

    if mode == 'RT':

        return Y_shift, Y_ns

    # Otherwise, load the EEG data and epoch the EDF as well (default)

    else:

        # Format the trials into MNE-compatible array
        events = []

        shift_ev = [[x, 0, 0] for x in shift_idxs]
        ns_ev = [[x, 0, 1] for x in ns_idxs]

        events.extend(shift_ev)
        events.extend(ns_ev)

        events.sort(key=lambda x: x[0])
        events = np.stack(events, axis=0)
        events = np.asarray(events, dtype=int)

        # Label trials for MNE
        event_id = {
            'Shift': 0,
            'NoShift': 1
        }

        # Re-reference if required
        if json_data.get('cles_rereference_electrode'):
            reref_e = json_data.get('cles_rereference_electrode')
            print('!!! Rereferencing to {}'.format(reref_e))
            raw_data = raw_data.set_eeg_reference(
                ref_channels=[reref_e], verbose=True)

        # Parse channel names and choose only seeg for analysis
        chs = pd.read_csv(os.path.join(path, json_data.get('channel_labels')))
        chs = chs[chs['Type'] == 'SEEG']
        
        # try:
        #     chs = chs[chs['DataValid']]
        
        # except:
        #     chs = chs[chs['DataValid'] == 'TRUE']

        ch_names = [x for x in chs['Label'].values]

        # Rename channels to label value
        ch_map = zip([x for x in chs['Pinbox'].values],
                     [x for x in chs['Label'].values])
        mne.rename_channels(raw_data.info, {x: y for x, y in ch_map})

        # Pick sEEG channels only in the MNE raw instance for analysis
        raw_data = raw_data.pick(ch_names)

        # Coordinate data in MNI space
        x = [x for x in chs['LocX'].values]
        y = [y for y in chs['LocY'].values]
        z = [z for z in chs['LocZ'].values]
        coords = [[xx, yy, zz] for xx, yy, zz in np.stack((x, y, z), axis=1)]

        # Create bipolar montage if requested by the user
        if montage == 'bipolar':

            raw_data = mne.set_bipolar_reference(
                raw_data, anode=ch_names[:-1], cathode=ch_names[1:])

            # Parse out bipolar sEEG channels
            ls = raw_data.ch_names
            bipchs = []
            for ch in ls:
                if '-' in ch:
                    bipchs.append(ch)

            # Ensure we only save bipolar pairs along the same electrode
            # Also updates bipolar coordinates
            ch_names, bipcoor = [], []

            for cidx, ch in enumerate(bipchs):

                c_elec = re.split(r'(\d+)', ch.split('-')[0])[0]
                a_elec = re.split(r'(\d+)', ch.split('-')[1])[0]

                if c_elec == a_elec:

                    ch_names.append(ch)
                    bipcoor.append(
                        np.mean((coords[cidx], coords[cidx+1]), axis=0))

            coords = bipcoor

        # Save the brain atlas labels for each electrode
        regions, YEO = get_mapping(coords, ch_names, out)        

        # Filter data to remove noise. These are chosen to concord closely with
        # the livestream curry filter parameters

        filt_sos = signal.bessel(2, lf, btype='highpass', output='sos', fs=Fs)
        raw_data._data = signal.sosfilt(
            filt_sos, raw_data._data)  # Highpass filter

        iirb, iira = signal.iirnotch(60, 40, fs=Fs)  # Notch
        raw_data._data = signal.lfilter(
            iirb, iira, raw_data._data)

        iirb, iira = signal.iirnotch(120, 20, fs=Fs)  # Notch for harmonics
        raw_data._data = signal.lfilter(
            iirb, iira, raw_data._data)

        # Plot the PSDs post filtering
        raw_data.plot_psd(fmin=4, fmax=150, show=False)
        plt.savefig(os.path.join(out, 'filtered_PSD.png'))
        plt.close()

        # Epoch the data, taking 2 sec pre-stim and 2 sec post-stim data
        epoch_data = mne.Epochs(raw_data, np.asarray(events), event_id, event_repeated='drop',
                                tmin=-2, tmax=2, picks=ch_names, baseline=None)
        del raw_data  # Clear RAM as we dont need the raw file anymore

        # Load the epoched data as np array
        X = epoch_data.get_data()

        return shifts, X, Y, Y_shift, Y_ns,\
            corr, ch_names, Fs,\
            coords, regions, YEO


def class_data(Y, path, rt_paths, save_path, trigdef, shifts, mode='ind_day'):

################
#
#
#


    if mode == 'ind_day': # This version uses the RT of the current session

        shift, nonshift = [], []

        for i, y in enumerate(Y):

            if shifts[i]:

                shift.append(y)
            
            else: nonshift.append(y)

    elif mode == 'd1': # This mode uses the day one RT to set threshold for fast and slow

        shift, nonshift = load_from_edf(path, rt_paths[0], save_path, trigdef, mode='RT')

    else: # With this mode, we load the data from both days 

        s1, ns1 = load_from_edf(path, rt_paths[0], save_path, trigdef, mode='RT')
        s2, ns2 = load_from_edf(path, rt_paths[1], save_path, trigdef, mode='RT')
        
        shift, nonshift = np.concatenate((s1, s2)), \
                            np.concatenate((ns1, ns2))


    p_shift = np.percentile(shift, 50)
    p_nshift = np.percentile(nonshift, 50)

    fastvslow = []
    fvs_shift = []
    fvs_nonshift = []

    # Percentile for RT and calculated based on trial type
    shift_idx = 0
    for is_shift in shifts:
        
        if is_shift:

            if Y[shift_idx] < p_shift:
                fastvslow.append(0)
                fvs_shift.append(0)

            else:
                fastvslow.append(1)
                fvs_shift.append(1)

        else:

            if Y[shift_idx] < p_nshift:
                fastvslow.append(0)
                fvs_nonshift.append(0)

            else:
                fastvslow.append(1)  
                fvs_nonshift.append(1)

        shift_idx += 1

    return np.asarray(fastvslow), np.asarray(fvs_shift), np.asarray(fvs_nonshift)


def plot_confusion_matrix(cm,
                          target_names,
                          path,
                          mode,
                          title='Confusion matrix',
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(os.path.join(path, '%s_confusion_matrix.png' % mode))
    plt.close()


def split_shift_nonshift(X_full, Y, shifts):
    
    X_shift = []
    Y_shift_RT = []
    X_nonshift = []
    Y_nonshift_RT = []

    shift_idx = 0
    for is_shift in shifts:
        if is_shift:
            X_shift.append(X_full[shift_idx,:,:])
            Y_shift_RT.append(Y[shift_idx])
        else:
            X_nonshift.append(X_full[shift_idx,:,:])
            Y_nonshift_RT.append(Y[shift_idx])
        shift_idx += 1

    return np.asarray(X_shift), np.asarray(Y_shift_RT), \
        np.asarray(X_nonshift), np.asarray(Y_nonshift_RT)


def augment_data_gauss(X, Y, m):

    X_gauss = np.concatenate([X] * (m - 1), axis=0)
    Y_gauss = np.concatenate([Y] * (m - 1), axis=0)

    for trl in range(X_gauss.shape[0]):
        for cont in range(X_gauss.shape[1]):
            noise = np.random.normal(-1, 1, X.shape[2])
            X_gauss[trl, cont] += noise

    return np.concatenate((X, X_gauss), axis=0), np.concatenate((Y, Y_gauss), axis=0)