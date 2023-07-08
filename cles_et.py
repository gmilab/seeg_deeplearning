
'''

Python script for the analysis of tandem eye tracking data
for sEEG attentional testing in children

Modified to assess effects of intracranial stim on eye movements 

Nebras M. Warsi, Ibrahim Lab
April 2022


'''



# Setup
import numpy as np
import pandas as pd
import os
import h5py
from tqdm import tqdm
import gc
from multiprocessing import Pool
from itertools import repeat

# Stats tools
from pymer4 import Lmer
from scipy.stats import ttest_ind
import ieeg_stats_utils as statutil
from scipy.stats import zscore
import statsmodels.stats.api as sms
from scipy.ndimage.filters import gaussian_filter

from mne.stats import permutation_cluster_test

# Custom ET functions
from plot_gaze import *

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# Path definitions
load_path = ""
out_path = "/d/gmi/1/nebraswarsi/CLES/analysis/ET"
gaze_path = os.path.join(out_path, 'gaze_analysis/')
glme_path = os.path.join(out_path, 'gazeLME/')
pupil_path = os.path.join(out_path, 'pupil_analysis/')

if not os.path.isdir(out_path): os.mkdir(out_path)
if not os.path.isdir(gaze_path): os.mkdir(gaze_path)
if not os.path.isdir(glme_path): os.mkdir(glme_path)
if not os.path.isdir(pupil_path): os.makedirs(pupil_path)


#####
# Stat threshold
alpha = 0.05


#####
# Which correction method for multiple comps
multitest = 'RFT'


stim_file='/d/gmi/1/nebraswarsi/EyeTrack/stimuli/setshifting_task_example.png'
fix_file='/d/gmi/1/nebraswarsi/EyeTrack/stimuli/setshifting_fixation_example.png'

n_jobs = 20

##########
# Epochs for windowing
epochs = {

    'post': [60, 120]

}


######################
#
# Main analysis code
#
##



def get_resp_speed(respTime, thresh):

    if respTime >= thresh[1]:

        return 2

    if respTime <= thresh[0]:

        return 0

    else:

        return 1



def plot_dems(analysis_df):

    subjs = analysis_df['subject'].unique()

    # First, we begin by plotting patient ages 
    age_data = np.asarray([analysis_df[analysis_df['subject'] == y]['Age'].values[0] for y in subjs])

    yng = round(np.count_nonzero(np.where(age_data <= 10, 1, 0)) / len(age_data) * 100)
    mid = round(np.count_nonzero(np.where((age_data > 10) & (age_data < 14), 1, 0)) / len(age_data) * 100)
    old = round(np.count_nonzero(np.where(age_data >= 14, 1, 0)) / len(age_data) * 100)
    age_pie = [yng, mid, old]

    plt.figure(figsize=(5, 5))
    plt.pie(age_pie, labels=['≤10', '10-14', '14-18'], colors=sns.color_palette('tab10'))
    plt.legend()
    plt.savefig(os.path.join(out_path, 'age_pie.png'))
    plt.close()

    # We will also plot the distribution of biological sex
    sex_data = np.asarray([analysis_df[analysis_df['subject'] == y]['Sex'].values[0] for y in subjs])

    m = round(np.count_nonzero(np.where(sex_data == 'M', 1, 0)) / len(sex_data) * 100)
    fm = round(np.count_nonzero(np.where(sex_data == 'F', 1, 0)) / len(sex_data) * 100)
    sex_pie = [m, fm]

    plt.figure(figsize=(5, 5))
    plt.pie(sex_pie, labels=['Male', 'Female'], colors=sns.color_palette('tab10'))
    plt.legend()
    plt.savefig(os.path.join(out_path, 'sex_pie.png'))
    plt.close()



def gaze_analysis(analysis_df):

    ####
    #
    # Gaze Data Analysis with LME model
    #
    ####

    ####
    #
    # Split by trial type
    shift_df = analysis_df[(analysis_df['Shift'] == 1) & (analysis_df['Correct'] == 1) & 
        (analysis_df['pi'] == 0)].drop(columns=['Pos', 'Fix', 'Sacc', 'PD', 'ontarget'])

    # RT ~ gaze LME
    shift_lme_obj = Lmer("RT ~ gaze_target + Age + trial + (1|subject)", data=shift_df.dropna(subset=['gaze_target']))
    shift_lme = shift_lme_obj.fit()
    
    s_data = shift_lme_obj.data

    # Plot model summaries and fits
    shift_lme_obj.plot_summary(plot_intercept=False)
    plt.savefig(os.path.join(glme_path, 'shift_gaze_model.png'))
    plt.close()

    #########
    #
    # Plot raw data to visualize fit

    scRT = []

    for sid, subj in enumerate(np.unique(s_data['subject'].values)):
    
        s_ranef = shift_lme_obj.ranef['X.Intercept.'][sid]
        scRT.append(s_data[s_data['subject'] == subj]['RT'] - s_ranef)

    s_data['cRT'] = np.concatenate(scRT, axis=0)

    sns.regplot(x='gaze_target', y='cRT', data=s_data, x_estimator=np.mean, ci=None)
    plt.savefig(os.path.join(glme_path, 'shift_TARGET_regr.png'))
    plt.close()


    # Summary results    
    summary_results = ['%s (%s)' % (shift_lme['Estimate'][1], shift_lme['SE'][1]), 
        '%s (%s)' % (shift_lme['T-stat'][1], shift_lme['P-val'][1])]

    summary_results2 = ['%s (%s)' % (shift_lme['Estimate'][2], shift_lme['SE'][2]), 
        '%s (%s)' % (shift_lme['T-stat'][2], shift_lme['P-val'][2])]

    cols = pd.MultiIndex.from_product([['Shift Trials'], ['Coefficient (SE)', 't-value (p-value)']], names=['Clinical Variable', ''])
    gaze_lme = pd.DataFrame(np.vstack([summary_results, summary_results2]), index=['gaze_target', 'Age'], columns=cols)
    gaze_lme.to_csv(os.path.join(glme_path, 'gaze_LME.csv'))

    shift_df = shift_df[shift_df['Intent'] > 0]

    gazevars = ['gaze_target', 'gaze_bottom', 'gaze_off_target', 'time_to_first']
    event_lme_res = []

    for var in tqdm(gazevars):

        ##########
        # Stim LME

        s_event_obj = Lmer("%s ~ Stim + Age + trial + (1|subject)" % var, data=shift_df.dropna(subset=[var]))
        s_event_lme = s_event_obj.fit()

        # Plot model summaries and fits
        s_event_obj.plot_summary(plot_intercept=False)
        plt.savefig(os.path.join(glme_path, 'stim_%s_model.png' % var))
        plt.close()

        sns.pointplot(x='Stim', y=var, data=shift_df, ci=68)
        plt.savefig(os.path.join(glme_path, 'stim_%s.png' % var))
        plt.close()

        # Summary results    
        summary_results = ['%s (%s)' % (s_event_lme['Estimate'][1], s_event_lme['SE'][1]), 
            '%s (%s)' % (s_event_lme['T-stat'][1], s_event_lme['P-val'][1])]
        event_lme_res.append(summary_results)

    event_lme = pd.DataFrame(np.vstack(event_lme_res), index=gazevars, columns=cols)
    event_lme.to_csv(os.path.join(glme_path, 'Stim_gaze_LME.csv'))



def gaze_plots(analysis_data):

    mean_path = os.path.join(gaze_path, 'means/')
    diff_path = os.path.join(gaze_path, 'diff_maps/')

    if not os.path.isdir(mean_path): os.mkdir(mean_path)
    if not os.path.isdir(diff_path): os.makedirs(diff_path)

    # Analyze shift data only
    analysis_data = analysis_df[analysis_df['Shift']]
    analysis_data = analysis_data[analysis_data['Intent'] == 1] # Only those in which there was a stim intent
    analysis_data = analysis_data[analysis_data['Correct'] == 1] # Correct trials only
    analysis_data = analysis_data.reset_index()

    print("Analyzing...")

    # Generate trial-wise fixation maps windowed by time
    hms = []

    for trl in tqdm(range(len(analysis_data))):
        
        gazedist = np.asarray(analysis_data.iloc[trl]['Pos'])
        ep_hms = []

        for ep in epochs.keys():

            gd = gazedist[epochs.get(ep)[0]:epochs.get(ep)[1], :][~np.isnan(gazedist[epochs.get(ep)[0]:epochs.get(ep)[1], :]).any(axis=1)]

            if gd.any():

                ep_hms.append(get_gaze_heatmap(gd))

            else:

                ep_hms.append(np.zeros((1080, 1920)))

        hms.append(np.stack(ep_hms, axis=0))

    hms = np.stack(hms, axis=0) # Dimensions are trial, window, X, Y


    ###########################################3
    #
    # Trial indices for the various comparisons
    #

    # Stim and no stim
    stim = analysis_data[analysis_data['Stim'] > 0].index.tolist()
    no_stim = analysis_data[analysis_data['Stim'] == 0]

    # Plot mean fixation maps
    contrasts = [stim, no_stim.index.tolist()]
    labels = ['stim', 'no_stim']

    mean_maps = []

    for e, ep in enumerate(epochs.keys()):

        for ci, c in tqdm(enumerate(contrasts)):

            fig, ax = draw_display([1920, 1080], imagefile=(stim_file if ep != 'pre' else fix_file))

            # Mean for condition
            hmap = np.mean(hms[c, e, :, :], axis=0)
            mean_maps.append(hmap)

            # Remove zeros
            lowbound = np.mean(hmap[hmap>0])
            hmap[hmap<lowbound] = np.NaN

            # draw heatmap on top of task image
            ax.imshow(hmap, cmap='jet', alpha=0.5, origin='lower', vmin=0, vmax=0.3)

            # FINISH PLOT
            # invert the y axis, as (0,0) is top left on a display
            ax.invert_yaxis()
            fig.savefig(os.path.join(mean_path, '%s_%s_MEAN.png' % (ep, labels[ci])))
            plt.close()

            # Save RAM
            del fig, ax, hmap
            gc.collect()

    # Clear RAM
    del contrasts, labels
    gc.collect()

    # Statistical comparisons
    contrasts = [[stim, no_stim.index.tolist()]]
    labels = ['stim_vs_nostim']

    for e, ep in enumerate(epochs.keys()):

        for ci, c in tqdm(enumerate(contrasts)):

            fig, ax = draw_display([1920, 1080], imagefile=(stim_file if ep != 'pre' else fix_file))

            # Generate and correct T-map of differences
            tmap, _ = ttest_ind(hms[c[0], e, :, :], hms[c[1], e, :, :], nan_policy='propagate')

            # draw heatmap on top of task image
            ax.imshow(tmap, cmap='seismic', alpha=0.5, origin='lower', vmin=-7, vmax=7)

            # FINISH PLOT
            # invert the y axis, as (0,0) is top left on a display
            ax.invert_yaxis()
            fig.savefig(os.path.join(diff_path, '%s_%s.png' % (ep, labels[ci])))
            plt.close()

            # Save RAM
            del fig, ax, tmap #, sig_t
            gc.collect()



def run_hm_lme(analysis_data, eyefix):
    
    if not eyefix.any(): return [0, 1, 0, 0, 1, 0, 0, 1, 0]

    analysis_data['Eyefix'] = eyefix

    try:

        lme = Lmer("Eyefix ~ Stim + Age + trial + (1|subject)", data=analysis_data).fit()

        Stim_cor = lme['Estimate'][1]
        Stim_p = lme['P-val'][1]
        Stim_Z = lme['T-stat'][1]

    except:

        Stim_cor = 0
        Stim_p = 1
        Stim_Z = 0

    RT_Z = 0 if ((RT_Z == '') | (RT_Z == 'nan')) else RT_Z
    RT_p = 1.0 if ((RT_p == '') | (RT_p == 'nan')) else RT_p
    age_Z = 0 if ((age_Z == '') | (age_Z == 'nan')) else age_Z
    age_p = 1.0 if ((age_p == '') | (age_p == 'nan')) else age_p
    Stim_Z = 0 if ((Stim_Z == '') | (Stim_Z == 'nan')) else Stim_Z
    Stim_p = 1.0 if ((Stim_p == '') | (Stim_p == 'nan')) else Stim_p


    lme_res = [RT_cor, RT_p, RT_Z, age_cor, age_p, age_Z, Stim_cor, Stim_p, Stim_Z]
    

    return lme_res



def heatmap_lme(analysis_df):

    lme_path = os.path.join(gaze_path, 'LME/')
    csv_path = os.path.join(gaze_path, 'LME_csv/')

    if not os.path.isdir(lme_path): os.mkdir(lme_path)
    if not os.path.isdir(csv_path): os.mkdir(csv_path)

    # Split out shift trials
    analysis_data = analysis_df[analysis_df['Shift']]
    analysis_data = analysis_df[analysis_df['Intent'] > 0] # Intent to stim trials only
    analysis_data = analysis_data[analysis_data['Correct'] == 1] # Correct trials only
    analysis_data = analysis_data.reset_index()

    print("Analyzing...")

    # Generate trial-wise fixation maps windowed by time
    hms = []

    for trl in tqdm(range(len(analysis_data))):
        
        gazedist = np.asarray(analysis_data.iloc[trl]['Pos'])
        ep_hms = []

        for ep in epochs.keys():

            gd = gazedist[epochs.get(ep)[0]:epochs.get(ep)[1], :][~np.isnan(gazedist[epochs.get(ep)[0]:epochs.get(ep)[1], :]).any(axis=1)]

            if gd.any():

                ep_hms.append(get_gaze_heatmap(gd))

            else:

                ep_hms.append(np.zeros((1080, 1920), dtype=np.float16))

            del gd
            gc.collect()

        hms.append(np.stack(ep_hms, axis=0))

        del ep_hms, gazedist
        gc.collect()

    hms = np.stack(hms, axis=0) # Dimensions are trial, window, X, Y

    # Need this for LME to run 
    analysis_data = analysis_data.drop(columns=['Pos', 'Fix', 'Sacc', 'PD', 'Sex', 'ontarget'])

    for e, ep in enumerate(epochs):

        vax = hms.shape[2]
        hax = hms.shape[3]

        # Correlation and p-matrices for our regressors
        RT_cor_mat = np.ndarray((vax,hax))
        RT_p_mat = np.ndarray((vax,hax))
        RT_Z_mat = np.ndarray((vax, hax))

        age_cor_mat = np.ndarray((vax,hax))
        age_p_mat = np.ndarray((vax,hax))
        age_Z_mat = np.ndarray((vax, hax))

        # Stim correlations
        Stim_cor_mat = np.ndarray((vax, hax))
        Stim_p_mat = np.ndarray((vax, hax))
        Stim_Z_mat = np.ndarray((vax, hax))

        # Regression code (LME) with correction for multiple comparisons
        for y in tqdm(range(vax)):
            
            with Pool(n_jobs) as pool:

                lme_res = pool.starmap(run_hm_lme, zip(repeat(analysis_data), 
                                            [hms[:,e,y,x] 
                                            for x in range(hax)]))

            lme_res = np.stack(lme_res, axis=0)
            
            RT_cor_mat[y] = lme_res[:,0]
            RT_p_mat[y] = lme_res[:,1]
            RT_Z_mat[y] = lme_res[:,2]
            age_cor_mat[y] = lme_res[:,3]
            age_p_mat[y] = lme_res[:,4]
            age_Z_mat[y] = lme_res[:,5]
            Stim_cor_mat[y] = lme_res[:,6]
            Stim_p_mat[y] = lme_res[:,7]
            Stim_Z_mat[y] = lme_res[:,8]

            del lme_res
            gc.collect()


        # Save data 
        RT_data = pd.DataFrame(RT_Z_mat)
        RT_data.to_csv(csv_path + '%s~RT_Z.csv' % ep)
        pd.DataFrame(RT_cor_mat).to_csv(csv_path + '%s~RT_corcoeffs.csv' % ep)

        age_data = pd.DataFrame(age_Z_mat)
        age_data.to_csv(csv_path + '%s~age_Z.csv' % ep)
        pd.DataFrame(age_cor_mat).to_csv(csv_path + '%s~age_corcoeffs.csv' % ep)

        Stim_data = pd.DataFrame(Stim_Z_mat)
        Stim_data.to_csv(csv_path + '%s~Stim_Z.csv' % ep)
        pd.DataFrame(Stim_cor_mat).to_csv(csv_path + '%s~Stim_corcoeffs.csv' % ep)

        del RT_data, age_data, Stim_data
        gc.collect()


        ######
        #
        # Plot results
        #


        var_names = ['age', 'RT', 'Stim']        
        var_list = [age_Z_mat, RT_Z_mat, Stim_Z_mat]


        for idx, var in enumerate(var_list):

            fig, ax = draw_display([1920, 1080], imagefile=(stim_file if ep != 'pre' else fix_file))

            if multitest == 'RFT':

                # Random-field based cluster correction:
                clusters, _ = statutil.random_field_correct(var, sigma=1, alpha=alpha)

            # Mask pixels that were not cluster-level significant 
            var = np.ma.masked_where((clusters != 1), var)

            # draw heatmap on top of task image
            ax.imshow(var, cmap='seismic', alpha=0.5, origin='lower', vmin=-3, vmax=3)

            # FINISH PLOT
            # invert the y axis, as (0,0) is top left on a display
            ax.invert_yaxis()

            fig.savefig(lme_path + '%s_%s.png' % (ep, var_names[idx]))
            plt.close()

            del fig, ax, var, clusters

        gc.collect()



def plot_lme():

    '''

    Plots aggregate LME data


    '''
    
    print('*'*50)
    print('*'*50)
    print('\nPlotting aggregate LME data...\n')
    print('*'*50)
    print('*'*50 + '\n')


    for e in epochs.keys():

        for fname in os.listdir(os.path.join(gaze_path, 'LME_csv/')):

            if ((e in fname) & ('_corcoeff' in fname)):

                lme_res = pd.read_csv(os.path.join(os.path.join(gaze_path, 'LME_csv/'), fname)).to_numpy()[:,1:]

                if multitest == 'RFT':

                    # Random-field based cluster correction:
                    clusters, _ = statutil.random_field_correct(zscore(lme_res, nan_policy='omit'), sigma=3, alpha=0.001)


                #### 
                #
                # Correction for multiple comparisons across all comparisons
                #  

                lme_res = np.where(np.abs(lme_res) < 
                    0.1, np.nan, lme_res) 
                
                lme_res = np.where(clusters != 1, np.nan, lme_res)

                if '~RT' in fname:
                        
                        RT = lme_res

                elif '~age' in fname:
                                                    
                        age = lme_res

                elif '~Stim' in fname:
                        
                        Stim = lme_res


        vars = [Stim, RT, age]
        var_names = ['Stim', 'RT', 'age']

        for idx, var in enumerate(vars):

            fig, ax = draw_display([1920, 1080], imagefile=(stim_file if e != 'pre' else fix_file))

            # Create custom colormap
            cmap1 = plt.cm.Oranges_r.resampled(256)
            cmap2 = plt.cm.YlGn.resampled(256)
            cmaps = np.vstack((cmap1(np.linspace(0, 1, 128)), cmap2(np.linspace(0, 1, 256))))
            newcmp = ListedColormap(cmaps, name='OrangeGreen')
            
            # draw heatmap on top of task image
            ax.imshow(var, cmap=newcmp, alpha=0.5, vmin=-0.5, vmax=0.5, origin='lower')

            # FINISH PLOT
            # invert the y axis, as (0,0) is top left on a display
            ax.invert_yaxis()

            fig.savefig(os.path.join(gaze_path, 'LME/') + '%s_%s.png' % (e, var_names[idx]))
            plt.close()

            del fig, ax, var

        gc.collect()



def target_timeseries(analysis_df):

    ####
    #
    # Plots timeseries of target fixation between NoStim/Stim trials
    #
    ####

    ####
    #
    # Split by trial type
    shift_df = analysis_df[(analysis_df['Shift'] == 1) & (analysis_df['Correct'] == 1) & (analysis_df['Intent'] == 1) & (analysis_df['pi'] == 0)]

    # Plot target gaze timeseries
    fig, ax = plt.subplots(1, 1)

    NoStim = shift_df[shift_df['Stim'] == 0]
    Stim = shift_df[shift_df['Stim'] == 1]

    ns_ts = np.vstack(NoStim['ontarget'].values)
    s_ts = np.vstack(Stim['ontarget'].values)

    # Smooth TS
    ns_ts = gaussian_filter(ns_ts, sigma=[0, 3])
    s_ts = gaussian_filter(s_ts, sigma=[0, 3])

    ci_NoStim = sms.DescrStatsW(ns_ts).tconfint_mean()
    ci_Stim = sms.DescrStatsW(s_ts).tconfint_mean()

    ax.plot(np.mean(s_ts, axis=0)[60:] - np.mean(ns_ts, axis=0)[60:], color='cornflowerblue')
    ax.fill_between(np.arange(0, 60, 1), ci_Stim[0][60:] - ci_NoStim[0][60:], ci_Stim[1][60:] - ci_NoStim[1][60:], color='cornflowerblue', alpha=0.1)

    _, clusters, p, _ = permutation_cluster_test([ns_ts[:, 60:], s_ts[:, 60:]], out_type='mask', threshold=3.2)

    # Plot significant clusters
    for i_c, c in enumerate(clusters):
        
        c = c[0]
        
        if p[i_c] < 0.05:
            
            ax.plot(np.arange(c.start, c.stop-1, 1), (np.mean(s_ts, axis=0)[60:] - 
                    np.mean(ns_ts, axis=0)[60:])[c.start:c.stop-1], 
                    color='cornflowerblue', linewidth=5)

    fig.savefig(os.path.join(glme_path, 'STIM_target_timeseries.png'), dpi=300)
    plt.close()



def run_pupil_lme(analysis_data, pd):

    if not pd.any(): return [0, 1, 0]

    analysis_data['pd'] = pd

    try:

        lme = Lmer("pd ~ Stim + Age + trial + (1|subject)", data=analysis_data).fit()

        event_cor = lme['Estimate'][1]
        event_p = lme['P-val'][1]
        event_Z = lme['T-stat'][1]
        event_lci = lme['2.5_ci'][1]
        event_uci = lme['97.5_ci'][1]

    except:

        event_cor = 0
        event_p = 1
        event_Z = 0
        event_lci = 0
        event_uci = 0

    event_Z = 0 if ((event_Z == '') | (event_Z == 'nan')) else event_Z
    event_p = 1.0 if ((event_p == '') | (event_p == 'nan')) else event_p


    lme_res = [event_cor, event_p, event_Z, event_lci, event_uci]

    return lme_res
    


def pupil_analysis(analysis_df):

    '''

    Plot pupillary diameters related to Stims

    '''

    shift_data = analysis_df[(analysis_df['Shift'] == 1) & (analysis_df['Correct'] == 1) & (analysis_df['Intent'] == 1)].reset_index()

    pd_shift = np.stack([x for x in shift_data['PD']], axis=0)

    del_idxs = []
    for i, p in enumerate(pd_shift):

        pd_shift[i] = np.where(np.isnan(p), np.nanmean(p), p)

        if np.isnan(pd_shift[i]).any():
            del_idxs.append(i)

    # Remove trials with all NaNs
    pd_shift = np.delete(pd_shift, del_idxs, axis=0)
    shift_data = shift_data.drop(del_idxs, axis=0).reset_index()

    #####
    # Plot

    tsfig, tsax = plt.subplots()
    shift_ts = []

    for stim_cond in [True, False]:

        shift_idxs = shift_data.index[shift_data['Stim'] == 0].tolist() if not stim_cond else shift_data.index[shift_data['Stim'] == 1].tolist()
        shift_pd = pd_shift[shift_idxs]

        shift_s = shift_pd
        shift_ts.append(shift_s)

    # Compute permutation cluster test for post-stimulus period
    _, clusters, p, _ = permutation_cluster_test([shift_ts[0][:, 60:], shift_ts[1][:, 60:]], out_type='mask')
    
    # Now smooth the timeseries for plotting
    shift_ts[0] = gaussian_filter(shift_ts[0], sigma=[0, 3])
    shift_ts[1] = gaussian_filter(shift_ts[1], sigma=[0, 3])

    # Get the CIs for each condition
    shift_cis = []
    for s in shift_ts:
            
        shift_cis.append(sms.DescrStatsW(s).tconfint_mean())

    # Now plot the difference between the two conditions
    tsax.plot(np.mean(shift_ts[0], axis=0) - np.mean(shift_ts[1], axis=0), color='cornflowerblue')
    tsax.fill_between(np.arange(0, 120, 1), shift_cis[0][0] - shift_cis[1][0], shift_cis[0][1] - shift_cis[1][1], color='cornflowerblue', alpha=0.1)

    # Plot significant clusters
    for i_c, c in enumerate(clusters):

        c = c[0]

        if p[i_c] < 0.05:

            tsax.plot(np.arange(c.start + 60, c.stop + 59, 1), (np.mean(shift_ts[0][:, 60:], axis=0) - np.mean(shift_ts[1][:, 60:], axis=0))[c.start:c.stop-1], color='cornflowerblue', linewidth=5)

    # Save timecourse data
    tsfig.savefig(pupil_path + 'pup_timecourse.png', dpi=300)
    plt.close()

    # Analyze effect of Stim on PD
    lme_pd = []

    for tidx in range(pd_shift.shape[1]):

        res = run_pupil_lme(shift_data.drop(columns=['Pos', 'Fix', 'Sacc', 'PD', 'Sex', 'ontarget']), pd_shift[:, tidx])
        lme_pd.append(res)

    lme_pd = np.stack(lme_pd, axis=0)

    #####
    # Plot

    fig, axs = plt.subplots(1, 1)
    axs.plot(gaussian_filter(lme_pd[:, 0], 3), color='grey', alpha=0.5)

    # Find Sig Regions
    sig, _ = statutil.permutation_cluster_correct1D(lme_pd[:, 2], alpha=0.05)

    sig_pd = np.where(sig != 0, gaussian_filter(lme_pd[:, 0], 3), np.nan)
    axs.plot(sig_pd, color='cornflowerblue', linewidth=10, alpha=0.5)

    # Save timecourse data
    fig.savefig(pupil_path + 'pup_LME.png', dpi=300)
    plt.close()



def pupil_rt(analysis_df):

    '''

    Plot relationship of RT with pupillary diameter

    '''
    
    shift_data = analysis_df[(analysis_df['Shift'] == 1) & (analysis_df['Correct'] == 1) & (analysis_df['pi'] == 0) & (analysis_df['Stim'] == 0)].reset_index()

    pd_shift = np.stack([x for x in shift_data['PD']], axis=0)

    del_idxs = []
    for i, p in enumerate(pd_shift):

        pd_shift[i] = np.where(np.isnan(p), np.nanmean(p), p)
        
        if np.isnan(pd_shift[i]).any():
            del_idxs.append(i)

    # Remove trials with all NaNs
    pd_shift = np.delete(pd_shift, del_idxs, axis=0)
    shift_data = shift_data.drop(del_idxs, axis=0).reset_index()

    # Smooth with Gaussian fitler
    pd_shift = gaussian_filter(pd_shift, [0, 3])

    # Analyze effect of Stim on PD
    lme_pd = []
    analysis_data = shift_data.copy().drop(columns=['Pos', 'Fix', 'Sacc', 'PD', 'Sex', 'ontarget'])

    for tidx in range(pd_shift.shape[1]):

        analysis_data['pd'] = pd_shift[:, tidx]

        try:

            lme = Lmer("RT ~ pd + Age + trial + (1|subject)", data=analysis_data).fit()

            event_cor = lme['Estimate'][1]
            event_p = lme['P-val'][1]
            event_Z = lme['T-stat'][1]
            event_lci = lme['2.5_ci'][1]
            event_uci = lme['97.5_ci'][1]

        except:

            event_cor = 0
            event_p = 1
            event_Z = 0
            event_lci = 0
            event_uci = 0

        event_Z = 0 if ((event_Z == '') | (event_Z == 'nan')) else event_Z
        event_p = 1.0 if ((event_p == '') | (event_p == 'nan')) else event_p


        lme_res = [event_cor, event_p, event_Z, event_lci, event_uci]
        lme_pd.append(lme_res)

    lme_pd = np.stack(lme_pd, axis=0)

    #####
    # Plot

    fig, axs = plt.subplots(1, 1)
    axs.plot(lme_pd[:, 0] * -1, color='grey', alpha=0.5)

    # Find Sig Regions
    sig, _ = statutil.permutation_cluster_correct1D(np.abs(lme_pd[:, 2]), alpha=0.05)

    sig_pd = np.where(sig != 0, lme_pd[:, 0] * -1, np.nan)
    axs.plot(sig_pd, color='cornflowerblue', linewidth=10, alpha=0.5)

    # Save timecourse data
    fig.savefig(pupil_path + 'pup_RT.png')
    plt.close()






# Runs the script
if __name__ == "__main__":


    subjects = [ ''' subject list de-identified for public sharing ''']

    print()
    print('#' * 50)
    print('#' * 50)
    print('\nWelcome!')
    print('\nThis script will analyze tandem eye tracking and CLeS data!')
    print('\nEnjoy!\n')
    print('#' * 50)
    print('#' * 50)

    print('\nLoading subjects: %s\n' % subjects)

    # Instantiate analysis dataframe
    analysis_df = []

    for id, subj in enumerate(subjects):

        pat_path = os.path.join(load_path, subj, 'ET/processed_ET')

        # Load demographic data
        with open((load_path + '/patient_dems.csv'), newline='\n') as csvfile:
            dem_data = pd.read_csv(csvfile, delimiter=',')
            age = float(dem_data[dem_data['ID'] == subj]['Age'])
            gai = float(dem_data[dem_data['ID'] == subj]['GAI'])
            sex = tuple(dem_data[dem_data['ID'] == subj]['Sex'].values)[0]

        # Load pre-processed subject data
        dataset = h5py.File(os.path.join(pat_path, "processed_data.h5"), 'r')
        respTimes, shifts, cidxs = dataset.get('Y')[()], dataset.get('shifts')[()], dataset.get('cidxs')[()]
        eyepos, eyefix, eyesacc, pupdiam = dataset.get('eyepos')[()], dataset.get('eyefix')[()], dataset.get('eyesacc')[()], dataset.get('pupdiam')[()]
        choice, L_val, R_val, gazeT, gazeB, gazeOT, ttf, tot = dataset.get('choice')[()], dataset.get('L_val')[()], dataset.get('R_val')[()], dataset.get('gazeT')[()], dataset.get('gazeB')[()], dataset.get('gazeOT')[()], dataset.get('ttf')[()], dataset.get('tot')[()]
        intent, stim = dataset.get('intent')[()], dataset.get('stim')[()]
        contacts, aal_regions, yeo_networks = dataset.attrs['contacts'], dataset.attrs['regions'], dataset.attrs['YEO_networks']
        coords = dataset.attrs['coords']
        dataset.close()

        shifts = np.array(shifts, dtype=bool)

        thresh_s_l = np.percentile(respTimes, 33)
        thresh_s_h = np.percentile(respTimes, 66)

        thresh_ns_l = np.percentile(respTimes[~shifts], 33)
        thresh_ns_h = np.percentile(respTimes[~shifts], 66)

        n_s_trl = 0

        for resp_idx in range(len(respTimes)):
            
            analysis_df.append(
                
                {
                    'subject': id,
                    'Age': age,
                    'Sex': sex,
                    'GAI': gai,
                    
                    'trial': resp_idx,
                    'choice': choice[resp_idx],
                    'Shift': shifts[resp_idx],
                    'Correct': 1 if resp_idx in cidxs else 0,
                    'pi': 0 if resp_idx == 0 else (0 if (resp_idx - 1) in cidxs else 1), 
                    'RT': respTimes[resp_idx],
                    'nRT': (respTimes[resp_idx] - np.mean(respTimes[shifts])) if shifts[resp_idx] else (respTimes[resp_idx] - np.mean(respTimes[~shifts])),
                    'lRT': np.log(respTimes[resp_idx]),

                    'Intent': intent[n_s_trl] if shifts[resp_idx] else -1,
                    'Stim': stim[n_s_trl] if shifts[resp_idx] else -1,

                    'match_side': 0 if L_val[resp_idx] else 1,
                    'item_value_0': L_val[resp_idx],
                    'item_value_1': R_val[resp_idx],
                    'gaze_target': gazeT[resp_idx],
                    'gaze_bottom': gazeB[resp_idx],
                    'gaze_off_target': gazeOT[resp_idx],
                    'time_to_first': ttf[resp_idx] if ttf[resp_idx] > 0 else 0,
                    'ontarget': tot[resp_idx],
                    
                    'Pos': eyepos[resp_idx],
                    'Fix': eyefix[resp_idx],
                    'Sacc': eyesacc[resp_idx],
                    'PD': pupdiam[resp_idx] - np.nanmean(pupdiam[resp_idx])    

                }
            )

            if shifts[resp_idx]: n_s_trl += 1

    analysis_df = pd.DataFrame(analysis_df)

    print('Running gaze LME...\n')
    gaze_analysis(analysis_df)

    print('Plotting on-target gaze timeseries...')
    target_timeseries(analysis_df)

    print('Analyzing gaze data...\n')
    gaze_plots(analysis_df)
    heatmap_lme(analysis_df)
    plot_lme()

    print('Analyzing pupil data...\n')
    pupil_analysis(analysis_df)
    pupil_rt(analysis_df)

    print('Done!\n')