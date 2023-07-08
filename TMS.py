'''

Processing script for TMS-EEG analysis
to implement non-invasive control of attention

Nebras M. Warsi, 2023


'''

# General imports
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

# DSP imports
import mne
from scipy.integrate import simpson

# Stats imports
from pymer4 import Lmer
from ieeg_stats_utils import random_field_correct
from scipy.stats import norm
from scipy.stats import kruskal, mannwhitneyu, wilcoxon, shapiro

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns


path = '/d/gmi/1/nebraswarsi/CLES/TMS_EEG/'
stim_path = os.path.join(path, 'participants')
ns_path = os.path.join(path, 'EEG_only', 'participants')
analysis_path = os.path.join(path, 'final_analysis')
prior_path = os.path.join(analysis_path, 'Prior')

if not os.path.isdir(analysis_path):
    os.mkdir(analysis_path)
if not os.path.isdir(prior_path):
    os.mkdir(prior_path)

# Dictionary of frequency bands of interest
freq_bands = {'theta': [4, 8], 'alpha': [8, 12], 'beta': [12, 30]}


# For spatial correction later
ch_shape =      np.array([['-', 0, 2, 1, '-'],
                            [3, 4, 5, 6, 7],
                            [8, 9, '-', 10, 11],
                            [12, 13, 14, 15, 16],
                            [17, 18, '-', 19, 20],
                            [21, 22, 23, 24, 25],
                            ['-', 26, '-', 27, '-'],
                            ['-', 29, '-', 29, '-']])


# Load participant data
def load_data(subj, subj_path, si):

    # Load the mat file in a similar format to Matlab struct
    mat = loadmat(os.path.join(subj_path, subj, 'data_all_clean.mat'), struct_as_record=False, squeeze_me=True)['data_all_clean']

    Fs = mat.fsample
    chs = mat.label
    ch_types = ['eeg' if 'EOG' not in x else 'eog' for x in chs]

    if 'EEG_only' not in subj_path:

        cond = mat.trialinfo[:, 0] # The prioritized stimulus
        probe1 = mat.trialinfo[:, 2] # The first probe
        corr1 = mat.trialinfo[:, 3] # The first probe's correctness
        rt1 = mat.trialinfo[:, 4]

        probe2 = mat.trialinfo[:, 5] # The second probe
        corr2 = mat.trialinfo[:, 6] # The second probe's correctness
        rt2 = mat.trialinfo[:, 7]

    else:

        cond = mat.trialinfo[:, 0] # The prioritized stimulus
        probe1 = mat.trialinfo[:, 1] # The first probe
        corr1 = mat.trialinfo[:, 2] # The first probe's correctness
        rt1 = mat.trialinfo[:, 3]

        probe2 = mat.trialinfo[:, 4] # The second probe
        corr2 = mat.trialinfo[:, 5] # The second probe's correctness
        rt2 = mat.trialinfo[:, 6]

    # Save data for Pandas dataframe for statistical analysis
    task_data = pd.DataFrame()
    task_data['cond'] = cond
    task_data['probe1'] = probe1
    task_data['corr1'] = corr1.astype(int)
    task_data['rt1'] = rt1
    task_data['probe2'] = probe2
    task_data['corr2'] = corr2.astype(int)
    task_data['rt2'] = rt2
    task_data['subj'] = [si] * len(cond)

    # Create a MNE info object
    info = mne.create_info(list(chs), Fs, ch_types=ch_types)

    # Make channel montage
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    # Now get the data as an array (in shape epochs x channels x time)
    trl_dat = np.asarray([x for x in mat.trial])

    # Create MNE epochs object
    trl_epoched = mne.EpochsArray(data=trl_dat, info=info)

    # Compute PSDs
    psds = trl_epoched.compute_psd('welch', fmin=4, fmax=100, tmin=4, tmax=5, n_jobs=-1)
    freqs = psds.freqs

    # Convert to np array
    psds = psds.get_data()

    # # Baseline the PSDs
    psds = psds / simpson(psds, freqs, axis=2)[:, :, np.newaxis] * 100

    # Get TBR
    tbr = np.mean(psds[:, :, (freqs >= 4) & (freqs <= 8)], axis=2) /\
            np.mean(psds[:, :, (freqs >= 12) & (freqs <= 30)], axis=2)

    # Plot the PSDs
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for ch_psd in np.mean(psds, axis=0): ax.plot(freqs, ch_psd)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (% total power)')
    ax.set_title('PSD of all trials')
    fig.savefig(os.path.join(subj_path, 'PSD_all_trials_{}.png'.format(subj)))
    plt.close()

    # Get the power for each band in freq_bands
    freq_data = []
    for _, (fmin, fmax) in freq_bands.items():
    
        freq_data.append(np.mean(psds[:, :, (freqs >= fmin) & (freqs <= fmax)], axis=2))

    # Now add theta/beta ratio
    freq_data.append(tbr)

    # Get the correct / incorrect trials for the stimuli, as well as 
    # if they were presented first or second and prioritized or not
    SPAT_corr, SPAT_rt, SPAT_prior, SPAT_order = [], [], [], []
    fig_corr, fig_rt, fig_prior, fig_order = [], [], [], []
    prior_corr, prior_rt, prior_order = [], [], []
    unprior_corr, unprior_rt, unprior_order = [], [], []

    for i, pr in enumerate(probe1):

        if pr == 2: # Means the first probe was spatial

            SPAT_corr.append(corr1[i])
            SPAT_rt.append(rt1[i])
            SPAT_prior.append(cond[i] == 2)
            SPAT_order.append(1)

            fig_corr.append(corr2[i])
            fig_rt.append(rt2[i])
            fig_prior.append(cond[i] == 1)
            fig_order.append(2)
                    
        else: # Means the first probe was figure

            SPAT_corr.append(corr2[i])
            SPAT_rt.append(rt2[i])
            SPAT_prior.append(cond[i] == 2)
            SPAT_order.append(2)

            fig_corr.append(corr1[i])
            fig_rt.append(rt1[i])
            fig_prior.append(cond[i] == 1)
            fig_order.append(1)

        if cond[i] == pr: # Means first stimulus was prioritized

            prior_corr.append(corr1[i])
            prior_rt.append(rt1[i])
            prior_order.append(1)

            unprior_corr.append(corr2[i])
            unprior_rt.append(rt2[i])
            unprior_order.append(2)

        else:

            prior_corr.append(corr2[i])
            prior_rt.append(rt2[i])
            prior_order.append(2)

            unprior_corr.append(corr1[i])
            unprior_rt.append(rt1[i])
            unprior_order.append(1)

    # Save to dataframe
    task_data['SPAT_corr'] = SPAT_corr
    task_data['SPAT_rt'] = SPAT_rt
    task_data['SPAT_rs'] = SPAT_rt > np.median(SPAT_rt)
    task_data['SPAT_prior'] = SPAT_prior
    task_data['SPAT_order'] = SPAT_order
    task_data['fig_corr'] = fig_corr
    task_data['fig_rt'] = fig_rt
    task_data['fig_rs'] = fig_rt > np.median(fig_rt)
    task_data['fig_prior'] = fig_prior
    task_data['fig_order'] = fig_order
    task_data['prior_corr'] = prior_corr
    task_data['prior_rt'] = prior_rt
    task_data['prior_rs'] = prior_rt > np.median(prior_rt)
    task_data['prior_order'] = prior_order
    task_data['unprior_corr'] = unprior_corr
    task_data['unprior_rt'] = unprior_rt
    task_data['unprior_rs'] = unprior_rt > np.median(unprior_rt)
    task_data['unprior_order'] = unprior_order

    return task_data, np.stack(freq_data, axis=1), info



# Run the model
def run_model(task_data, freq_data, info):
    
    print()
    print('*' * 50)
    print('*' * 50)
    print()
    print('Running models...')
    print()
    print('*' * 50)
    print('*' * 50)
    print()

    # Bands of interest
    fbs = ['theta', 'alpha', 'beta', 'tbr']

    # Define the variables to model
    vars = ['corr', 'rt']

    for var in vars:

        # Run the model for each channel and each frequency band
        for fi, fb in enumerate(fbs):

            prior_es, prior_p = [], []
            unprior_es, unprior_p = [], []

            prior_int_es, prior_int_p = [], []
            unprior_int_es, unprior_int_p = [], []

            for ci, _ in enumerate(info.ch_names[:-2]):

                data = task_data.copy()
                data['Power'] = np.log10(freq_data[:, fi, ci])

                # Run models for the prioritized and unprioritized trials
                
                # Drop missing trials from the dataframe
                p_data = data[data['prior_corr'] != 888]
                p_data = p_data[p_data['prior_corr'] != 999]
                # Get relevant prior and order variables
                p_data['order'] = p_data['prior_order']

                if 'rt' in var:

                    # Correct trials only for RT analysis
                    p_data = p_data[p_data['prior_corr'] == 1]

                    # Log transform RT
                    p_data['prior_rt'] = np.log10(p_data['prior_rt'])

                mod_obj = Lmer(data=p_data, formula='prior_%s ~ Power + Stim + Power:Stim + (1|subj)' % var)
                mod = mod_obj.fit()
                prior_es.append(mod['Estimate'][1])
                prior_p.append(mod['P-val'][1])
                prior_int_es.append(mod['Estimate'][-1])
                prior_int_p.append(mod['P-val'][-1])

                #
                #
                # # Now do the unprioritized trials

                # Drop missing trials from the dataframe
                up_data = data[data['unprior_corr'] != 888]
                up_data = up_data[up_data['unprior_corr'] != 999]
                # Get relevant prior and order variables
                up_data['order'] = up_data['unprior_order']

                if 'rt' in var:

                    # Correct trials only for RT analysis
                    up_data = up_data[up_data['unprior_corr'] == 1]

                    # Log transform RT
                    up_data['unprior_rt'] = np.log10(up_data['unprior_rt'])

                mod_obj = Lmer(data=up_data, formula='unprior_%s ~ Power + Stim + Power:Stim + (1|subj)' % var)
                mod = mod_obj.fit()
                unprior_es.append(mod['Estimate'][1])
                unprior_p.append(mod['P-val'][1])
                unprior_int_es.append(mod['Estimate'][-1])
                unprior_int_p.append(mod['P-val'][-1])


            # Now correct the p-values and plot the
            # prior and unprior effect sizes

            for covar in ['Power', 'Power:Stim']:

                for cond in ['prior', 'unprior']:

                    if cond == 'prior':

                        if covar == 'Power':

                            es = prior_es; p = prior_p

                        else:

                            es = prior_int_es; p = prior_int_p; es_fixed = prior_es; p_fixed = prior_p

                    else:

                        if covar == 'Power':

                            es = unprior_es; p = unprior_p

                        else:

                            es = unprior_int_es; p = unprior_int_p; es_fixed = unprior_es; p_fixed = unprior_p

                    # Convert p-values to Z-scores
                    p = np.array(p)
                    p = np.where(p == 0, 1e-10, p)
                    z = np.abs(norm.ppf(p / 2)) * np.sign(es)

                    # Reshape the array into same positons as the sensors
                    z_scalp = np.zeros((ch_shape.shape[0], ch_shape.shape[1]))

                    for ri in range(ch_shape.shape[0]):
                        for ci in range(ch_shape.shape[1]):

                            if ch_shape[ri, ci] != '-':
                                z_scalp[ri, ci] = z[int(ch_shape[ri, ci])]

                    # Mask the remaining values to ensure that they are not plotted
                    z_scalp = np.ma.masked_where(ch_shape == '-', z_scalp)

                    # Cluster correct the p-values
                    sig = random_field_correct(z_scalp, sigma=1.5, alpha=0.05)[0].ravel()

                    # Remove the '-' values from the mask
                    sig = np.delete(sig, np.where(ch_shape.ravel() == '-'))

                    # Get the sig effect sizes
                    sig_es = np.where(sig == 0, 0, es)
                    
                    if np.amax(np.abs(sig_es)) > 0: vlim = (-0.75*np.amax(np.abs(sig_es)), 0.75*np.amax(np.abs(sig_es)))
                    else: vlim = (-0.05, 0.05)

                    if 'rt' in var: opath = os.path.join(prior_path, 'rt')                    
                    else: opath = os.path.join(prior_path, 'corr')
                    if not os.path.exists(opath): os.makedirs(opath)

                    if 'corr' in var: es = np.multiply(es, -1) # Change sign to match the RT analysis

                    # Plot the topomap
                    mne.viz.plot_topomap(es, info, mask=sig, cmap='RdBu_r', vmin=vlim[0], vmax=vlim[1], show=False)

                    # Print max and min
                    print(np.amax(es), np.amin(es))

                    # Save the figure
                    plt.savefig(os.path.join(opath, '%s_%s_%s.png' % (covar, cond, fb)), dpi=300)
                    plt.close()



# Plot the data
def plot_data(task_data):

    props = {
        'boxprops': {'facecolor': 'none', 'edgecolor': 'silver'},
        'medianprops': {'color': 'silver'},
        'whiskerprops': {'color': 'silver'},
        'capprops': {'color': 'none'}
    }

    # Plot barplot of proportion correct for these subjects
    plot_df = task_data[task_data['prior_corr'] != 888]
    plot_df = plot_df[plot_df['prior_corr'] != 999]

    summary_df = plot_df.groupby(['Stim', 'subj']).agg({'prior_corr': 'mean'}).reset_index()

    # Make a boxplot of the proportion correct
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(x='Stim', y='prior_corr', data=summary_df, ax=ax, showfliers=False, **props)
    sns.stripplot(x='Stim', y='prior_corr', data=summary_df,
                  color='teal', size=6.5, ax=ax, jitter=0.12)
    fig.savefig(os.path.join(analysis_path, 'prior_proportion_correct.png'), dpi=300)
    plt.close()

    ns = summary_df[summary_df['Stim'] == 0]['prior_corr'].values
    s = summary_df[summary_df['Stim'] == 1]['prior_corr'].values

    # Check for normality
    print('Normality')
    print(shapiro(s))
    print(shapiro(ns))

    # Print p-value
    print('Prior proportion correct')
    print(mannwhitneyu(s, ns))

    # Now do this for unprior
    plot_df = task_data[task_data['unprior_corr'] != 888]
    plot_df = plot_df[plot_df['unprior_corr'] != 999]

    summary_df = plot_df.groupby(['Stim', 'subj']).agg({'unprior_corr': 'mean'}).reset_index()

    # Make a boxplot of the proportion correct
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(x='Stim', y='unprior_corr', data=summary_df, showfliers=False, **props)
    sns.stripplot(x='Stim', y='unprior_corr', data=summary_df,
                  color='teal', size=6.5, ax=ax, jitter=0.12)
    fig.savefig(os.path.join(analysis_path, 'unprior_proportion_correct.png'), dpi=300)
    plt.close()

    ns = summary_df[summary_df['Stim'] == 0]['unprior_corr'].values
    s = summary_df[summary_df['Stim'] == 1]['unprior_corr'].values

    # Print p-value
    print('Unprior proportion correct')
    print(mannwhitneyu(s, ns))

    # Now do this again but compare SPAT_prior to fig_prior in Stim == 1
    plot_df = task_data[task_data['prior_corr'] != 888]
    plot_df = plot_df[plot_df['prior_corr'] != 999]
    plot_df = plot_df[plot_df['Stim'] == 1]

    spat_df = plot_df[plot_df['SPAT_prior'] == 1]
    spat_summary = spat_df.groupby(['subj']).agg({'prior_corr': 'mean'}).reset_index()
    fig_df = plot_df[plot_df['fig_prior'] == 1]
    fig_summary = fig_df.groupby(['subj']).agg({'prior_corr': 'mean'}).reset_index()

    # Now combine these two dataframes
    summary_df = pd.concat([spat_summary, fig_summary])
    summary_df['variable'] = np.concatenate([['SPAT_prior'] * len(spat_summary), 
                                            ['fig_prior'] * len(fig_summary)])

    # Make a boxplot of the proportion correct
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(x='variable', y='prior_corr', data=summary_df, ax=ax, showfliers=False, **props)
    sns.stripplot(x='variable', y='prior_corr', data=summary_df,
                    color='teal', size=6.5, ax=ax, jitter=0.12)
    fig.savefig(os.path.join(analysis_path, 'SPATxfig_prior_proportion_correct.png'), dpi=300)
    plt.close()

    spat = summary_df[summary_df['variable'] == 'SPAT_prior']['prior_corr'].values
    fig = summary_df[summary_df['variable'] == 'fig_prior']['prior_corr'].values

    # Check for normality
    print('Normality')
    print(shapiro(spat))
    print(shapiro(fig))

    # Print p-value
    print('SPAT vs fig proportion correct')
    print(mannwhitneyu(spat, fig))

    # Finally, do this for RT in the prior condition
    plot_df = task_data[task_data['prior_corr'] != 888]
    plot_df = plot_df[plot_df['prior_corr'] != 999]

    summary_df = plot_df.groupby(['Stim', 'subj']).agg({'prior_rt': 'mean'}).reset_index()

    # Make a boxplot of the mean RT for each subject
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(x='Stim', y='prior_rt', data=summary_df, ax=ax, showfliers=False, **props)
    sns.stripplot(x='Stim', y='prior_rt', data=summary_df,
                    color='teal', size=6.5, ax=ax, jitter=0.12)
    fig.savefig(os.path.join(analysis_path, 'prior_RT.png'), dpi=300)
    plt.close()

    ns = summary_df[summary_df['Stim'] == 0]['prior_rt'].values
    s = summary_df[summary_df['Stim'] == 1]['prior_rt'].values

    # Check for normality
    print('Normality')
    print(shapiro(s))
    print(shapiro(ns))

    # Print p-value
    print('Prior RT')
    print(mannwhitneyu(s, ns))


    #####################
    # Plot proportion correct / incorrect based on Theta power at frontal midline channels for the stim patients
    task_data['Power'] = np.mean(fd[:, 0, [2, 5, 9, 10]], axis=-1)
    plot_df = task_data[task_data['prior_corr'] != 888]
    plot_df = plot_df[plot_df['prior_corr'] != 999]

    # Now get subsets for high and low power based on high/low theta power, do this for each subject
    high_power, low_power = [], []
    for subj in plot_df['subj'].unique():
        subj_df = plot_df[plot_df['subj'] == subj]
        subj_df = subj_df.sort_values(by='Power', ascending=False)
        high_power.append(subj_df.iloc[:int(len(subj_df) / 2)])
        low_power.append(subj_df.iloc[-int(len(subj_df) / 2):])

    high_power = pd.concat(high_power)
    low_power = pd.concat(low_power)
    high_power['Power'] = [1] * len(high_power)
    low_power['Power'] = [0] * len(low_power)

    plot_df = pd.concat([high_power, low_power])
    summary_df = plot_df.groupby(['Power', 'Stim', 'subj']).agg({'prior_corr': 'mean'}).reset_index()

    # Make a boxplot of the proportion correct
    g = sns.catplot(kind='box', x='Power', y='prior_corr', hue='Stim', 
                                    data=summary_df, ax=ax, **props)
    g.map_dataframe(sns.stripplot, x='Power', y='prior_corr', hue='Stim', data=summary_df,
                    color='teal', size=6.5, ax=ax, jitter=0.12, dodge=True)
    plt.savefig(os.path.join(analysis_path, 'theta_power_proportion_correct.png'), dpi=300)
    plt.close()

    # Split conditions for stats
    hp_stim = summary_df[(summary_df['Power'] == 1) & (summary_df['Stim'] == 1)]['prior_corr']
    hp_nostim = summary_df[(summary_df['Power'] == 1) & (summary_df['Stim'] == 0)]['prior_corr']
    lp_stim = summary_df[(summary_df['Power'] == 0) & (summary_df['Stim'] == 1)]['prior_corr']
    lp_nostim = summary_df[(summary_df['Power'] == 0) & (summary_df['Stim'] == 0)]['prior_corr']

    # Run statistical tests
    print('-'*50)
    print('Normality tests')
    print(shapiro(hp_stim))
    print(shapiro(hp_nostim))
    print(shapiro(lp_stim))
    print(shapiro(lp_nostim))
    print()

    print('-'*50)
    print('Statistical tests')
    
    print("Stim vs no stim")
    print(kruskal(hp_stim, hp_nostim, lp_stim, lp_nostim))
    print(mannwhitneyu(hp_stim, hp_nostim))
    print(mannwhitneyu(lp_stim, lp_nostim))
    print(mannwhitneyu(lp_stim, hp_nostim))
    print(mannwhitneyu(hp_stim, lp_nostim))

    print("High vs low power")
    print(wilcoxon(hp_stim, lp_stim))
    print(wilcoxon(hp_nostim, lp_nostim))

    print()



    







# Run the script
if __name__ == '__main__':

    # Loop through path directory to find subjects
    subjects = [x for x in os.listdir(stim_path) if os.path.isdir(os.path.join(stim_path, x))]

    dfs, fd = [], []
    # First load the TMS subjects
    for si, subj in enumerate(subjects):
       
        # Load data
        data, freq_data, info = load_data(subj, stim_path, si)
        dfs.append(data)
        fd.append(np.array(freq_data))

    # Get length of stims
    n_stims = len(pd.concat(dfs))

    # Get number of subjects
    n_stim_subj = len(subjects)
    
    # Now load the non-TMS subjects
    # Loop through path directory to find subjects
    subjects = [x for x in os.listdir(ns_path) if os.path.isdir(os.path.join(ns_path, x))]

    for si, subj in enumerate(subjects):
       
        # Load data
        data, freq_data, info = load_data(subj, ns_path, si + n_stim_subj)
        dfs.append(data)
        fd.append(np.array(freq_data))

    # Concatenate dataframes and power data
    task_data = pd.concat(dfs)
    task_data['Stim'] = np.concatenate([[1] * n_stims, [0] * (len(task_data) - n_stims)])
    fd = np.concatenate(fd, axis=0)

    # Plot the data for assessment
    plot_data(task_data)

    # Run the model
    run_model(task_data, fd, info)