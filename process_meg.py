
'''

MEG processing and analysis script for 
set-shifting data in normative paediatric sample

Nebras M. Warsi, Ibrahim Lab
Apr 2023

'''


##################
# Data processing
import os
import h5py
import numpy as np
import pandas as pd

# DSP
from scipy.signal import welch
from scipy.integrate import simpson

# Stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import roc_utils as ru


# Define the basepath
basepath = '/d/gmi/1/nebraswarsi/MEG/'

# Output paths
outpath = os.path.join(basepath, 'analysis')
psdpath = os.path.join(outpath, 'psd')
tdc_path = os.path.join(outpath, 'tdc')

# If output path doesn't exist, create it
if not os.path.exists(outpath):
    os.makedirs(outpath)
if not os.path.exists(psdpath):
    os.makedirs(psdpath)
if not os.path.exists(tdc_path):
    os.makedirs(tdc_path)

regenerate_data = False

# Select the same ROIs as epilepsy cohort 
ROIs = np.array([34, 35, 0, 1, 
                 60, 61, 2, 3, 
                 32, 33, 4, 5,
                 80, 81]) # Custom regions


##################
# Data loader
#

def load_data(subj):

    bp = '/d/gmi/1/simeon/setshifting_meg/data/'

    ####################
    #
    # First load the trial info from the cfg.mat file
    #
    ########

    trl_file = h5py.File(os.path.join(bp, subj, 'preprocessing', 'setshifting', 'ft_trials_cfg_noartifactreject.mat'), 'r')
    trl = trl_file['trl'][:].T
    
    # We only want shift trials, which are marked with a 1.0
    shift_idxs = np.where(trl[:, 3] == 1.0)[0]

    ####################
    #
    # Now for each participant, we will load the beamformed MEG data and compute the PSD
    #
    ########

    bf_file_path = os.path.join(bp, subj, 'beamforming', 'setshifting-noartfctreject', '%s_VS_meancentered.hdf5' % subj)

    # Load the HDF5 file
    bf_file = h5py.File(bf_file_path, 'r')

    # Load the data, fs should be 600 for the MEG
    # Shape of the epoched data is (n_channels, n_trials, n_samples)
    data, fs = np.squeeze(bf_file['vs_ortho'][:]), bf_file['fsample'][:]

    # Reshape the data to (n_trials, n_channels, n_samples)
    data = np.transpose(data, (1, 0, 2))

    # Print the subject number
    print('Processing subject %s' % subj)

    # Now compute the pre- and post-stimulus PSDs
    # Same formula as the sEEG children
    freqs, prePSD = welch(data[shift_idxs, :, :int(2*fs)], fs=fs, nperseg=fs, nfft=fs)
    freq_range = np.where((freqs >= 4) & (freqs <= 43))[0]
    prePSD = prePSD[:, :, freq_range] / simpson(prePSD[:, :, freq_range], freqs[freq_range])[:, :, np.newaxis] * 100
    prePSD = prePSD[:, ROIs, :]

    ps = []
    for i in range(prePSD.shape[1]):
        if i %2 == 0: ps.append(np.mean(prePSD[:, i:i+1, :], axis=1))

    prePSD = np.stack(ps, axis=1)
    
    freqs, postPSD = welch(data[shift_idxs, :, int(2*fs):int(3*fs)], fs=fs, nperseg=fs, nfft=fs)
    postPSD = postPSD[:, :, freq_range] / simpson(postPSD[:, :, freq_range], freqs[freq_range])[:, :, np.newaxis] * 100
    postPSD = postPSD[:, ROIs, :]

    ps = []
    for i in range(postPSD.shape[1]):
        if i %2 == 0: ps.append(np.mean(postPSD[:, i:i+1, :], axis=1))

    postPSD = np.stack(ps, axis=1)

    # Get indices of trials with NaNs in them
    nan_idxs = np.where(np.isnan(prePSD).any(axis=(1, 2)))[0]

    # Now remove the trials with NaNs
    prePSD = np.delete(prePSD, nan_idxs, axis=0)
    postPSD = np.delete(postPSD, nan_idxs, axis=0)


    ####################
    #
    # Next we want the RT data from the CSV file
    #
    ########

    behav_df = pd.read_csv(os.path.join(bp, subj, 'preprocessing', 'setshifting', 'behav', 'trl_rt.csv'))          
    rt = behav_df['RT (sp)'].values
    
    # Convert to ms and get the data for the shift trials only
    rt = rt[shift_idxs] / fs * 1000
    
    # Remove the NaN trials
    rt = np.delete(rt, nan_idxs, axis=0)

    # Get the indices of the slow trials based on median RT
    slow_idxs = np.where(rt > np.median(rt))[0]
    fast_idxs = np.where(rt <= np.median(rt))[0]

    ####################
    #
    # Plot the PSDs up to 60Hz for a visual check 
    # between fast and slow trials (similar to CLeS)
    #
    ########
                
    slowPSD = np.mean(prePSD[slow_idxs, :, :], axis=0)
    fastPSD = np.mean(prePSD[fast_idxs, :, :], axis=0)

    # Plot the PSDs
    plt.figure(figsize=(10, 5))
    plt.plot(freqs[freq_range], slowPSD.T, color='blue', alpha=0.1)
    plt.plot(freqs[freq_range], fastPSD.T, color='red', alpha=0.1)
    plt.plot(freqs[freq_range], slowPSD.T, color='blue', label='Slow')
    plt.plot(freqs[freq_range], fastPSD.T, color='red', label='Fast')
    plt.xlim([0, 40])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (a.u.)')
    plt.legend()
    plt.savefig(os.path.join(psdpath, '%s_PSD.png' % subj))
    plt.close()

    # Now plot the difference between slow and fast using imshow
    plt.figure(figsize=(10, 5))
    plt.imshow(slowPSD - fastPSD, aspect='auto', cmap='RdBu_r',
                        vmin=-np.max(np.abs(slowPSD - fastPSD)), 
                        vmax=np.max(np.abs(slowPSD - fastPSD)))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Channel')
    plt.colorbar()
    plt.savefig(os.path.join(psdpath, '%s_PSD_diff.png' % subj))
    plt.close()

    return rt, prePSD, postPSD, freqs



##################
# Analytic functions
#

def make_wireframe(xx, yy, z, color='#0066FF'):
    
    line_marker = dict(color=color)
    lines = []
    for idx, (i, j, k) in enumerate(zip(xx, yy, z)): # Add every third line for visibility
        if idx % 3 == 0: lines.append(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker))
            
    return lines



def generate_shere(x, y, z, radius, n_sd, resolution=20):
    """Return the coordinates for plotting a sphere centered at (x,y,z)"""
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    X = n_sd * radius[0] * np.cos(u)*np.sin(v) + x
    Y = n_sd * radius[1] * np.sin(u)*np.sin(v) + y
    Z = n_sd * radius[2] * np.cos(v) + z
    return X, Y, Z



def pca_analysis(analysis_data, X_tdc):

    #####################
    #
    # Runs PCA analysis of MEG power for RT in the TDC group
    #
    #############

    print('Running PCA...')
    print('*'*50)

    # Set the data
    op = tdc_path; data = analysis_data


    ##############
    #
    # Analyze

    print('\nAnalyzing TDC group...')
    print('*'*50)

    # Get theta and beta power
    X_theta = np.mean(X_tdc[:, :, 0:4], axis=-1)
    X_beta = np.mean(X_tdc[:, :, 10:26], axis=-1)
    
    # Concatenate the data
    X_tdc = np.concatenate((X_theta, X_beta), axis=-1)

    # Now ravel for the PCA
    X = np.array(([x.ravel() for x in X_tdc]))

    Y = []
    for rt in data['RT'].values:
        
        if rt > np.percentile(data['RT'], 75): Y.append(1)
        elif rt <= np.percentile(data['RT'], 25): Y.append(0)
        else: Y.append(-1)

    Y = np.array(Y)

    # Remove the Y = -1 trials from the data
    X = X[Y != -1, :]
    Y = Y[Y != -1]

    X_scaled = StandardScaler().fit_transform(X)

    # Now run the PCA
    pca = PCA()
    pca.fit(X_scaled)

    # Transform the data
    pca_res = pca.transform(X)

    # Print eigenvalues for all PCs 
    eigenvalues = pca.explained_variance_ratio_
    cov_matrix = np.dot(X.T, X) / len(X)

    print('\nEigenvalues')
    print('-----------')
    print()

    for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):    
        print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
        print(eigenvalue)

    # Plot the eigenvalues
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(pca.explained_variance_ratio_, marker='o')
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Variation explained (%)')
    ax.set_title('Scree plot')

    # Save
    plt.savefig(os.path.join(op, 'RT_eigenvalues.png'), dpi=300)
    plt.close()

    # Now save the loading data as a table
    loadings = pd.DataFrame(pca.components_, columns=np.arange(1, 15))
    loadings.to_csv(os.path.join(op, 'RT_loadings.csv'))

    # Now get the relevant Y data for labelling
    # Now make the color array (red for a 1, blue for a zero, and white for -1)
    colors, symbols = [], []

    for y in Y:

        if y == 0: 
            colors.append('blue')
            symbols.append('circle')

        elif y == 1: 
            colors.append('red')
            symbols.append('circle')
            

    # Perform K-means clustering on the projected data
    kmeans = KMeans(n_clusters=2, random_state=0).fit(pca_res[:, :7])
    labels = kmeans.labels_

    # Get the cluster centroids
    centers = kmeans.cluster_centers_

    # For each cluster, draw an oval around the centroid with a radius equal to the standard deviation
    x, y, z = generate_shere(centers[0, 0], centers[0, 1], centers[0, 2], np.std(pca_res[labels==0, 0:3], axis=0), 2)
    sphere1 = make_wireframe(x, y, z, color='blue')

    x2, y2, z2 = generate_shere(centers[1, 0], centers[1, 1], centers[1, 2], np.std(pca_res[labels==1, 0:3], axis=0), 2)
    sphere2 = make_wireframe(x2, y2, z2, color='red')

    # Plot all together as a mesh with Plotly
    fig = go.Figure()
    
    # Add the wireframes
    for sphere in sphere1: fig.add_trace(sphere)
    for sphere in sphere2: fig.add_trace(sphere)

    # Adjust the width and opacity of the lines
    fig.update_traces(line_width=5)

    # Add the spheres again as a transparent surface, with the first one red and the second one blue
    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.1, colorscale='Blues', cmin=-1000, cmax=-4))
    fig.add_trace(go.Surface(x=x2, y=y2, z=z2, opacity=0.1, colorscale='Reds', cmin=-1000, cmax=-4))

    # Add the scatter plot data points
    fig.add_trace(go.Scatter3d(x=pca_res[:, 0], y=pca_res[:, 1], z=pca_res[:, 2], mode='markers',
                            marker=dict(color=colors, size=5, opacity=0.8, symbol=symbols)))

    # Set layout and show the plot
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                    margin=dict(l=0, r=0, b=0, t=0))
    
    # Make the background transparent
    fig.update_layout(scene = dict(
                    xaxis = dict(
                        backgroundcolor="white",
                        gridcolor="white",
                        showbackground=True),
                    yaxis = dict(
                        backgroundcolor="white",
                        gridcolor="white",
                        showbackground=True),
                    zaxis = dict(
                        backgroundcolor="white",
                        gridcolor="white",
                        showbackground=True),),
                    width=1000, height=1000, showlegend=False)
    
    # Save
    fig.write_html(os.path.join(op, 'RT_pca.html'))

    # Now calculate the AUC for the KMeans clustering in the TDCs only
    fpr, tpr, _ = roc_curve(Y, labels)
    roc_auc = auc(fpr, tpr)

    # Print the AUC
    print('\nAUC for KMeans TDC fast vs. slow: %f' % roc_auc)

    # Now plot the bootstrapped AUC curve using roc_utils
    fig, ax = plt.subplots(figsize=(10, 6))
    rocs = ru.compute_roc_bootstrap(Y, labels, pos_label=1, n_bootstrap=1000, return_mean=False)
    ru.plot_mean_roc(rocs, ax=ax, show_all=True, show_ci=True, show_ti=False, color='maroon', lw=2)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')

    # Save
    plt.savefig(os.path.join(op, 'RT_pca_auc.png'), dpi=300)
    plt.close()







# Run the script
if __name__ == "__main__":

    print()
    print("*"*50)
    print("*"*50)
    print("\nNormative MEG set-shifting script\n")
    print("*"*50)
    print("*"*50)
    print('\nLoading and concatenating subject data ...\n')

    # Load the channel names (these are the AAL regions that were beamformed)
    # These are located in a CSV file (we only take the first 90 channels)
    channel_names = pd.read_csv(os.path.join(basepath, 'MEG_centroids.csv'))['label'].values[:90]

    # Next we load participant demographic info from another CSV file
    pheno_data = pd.read_csv(os.path.join(basepath, 'participants', 'pheno.csv'))

    if regenerate_data:

        # Now load the data for each subject
        subj_id, trl_id, ages, RTs, lRTs, RSs, tdcPSDs, tdc_post = [], [], [], [], [], [], [], []
        
        # Loop over directories in the basepath
        patients = [x for x in os.listdir(os.path.join(basepath, 'participants/'))]

        for si, subj in enumerate(patients):

            try: rt, prePSD, postPSD, freqs = load_data(subj)
            except: continue

            s_df = pheno_data[pheno_data['subj_id'] == subj]
            age = s_df['age'].values[0]


            subj_id.append(np.repeat(si, len(rt)))
            ages.append(np.repeat(age, len(rt)))
            trl_id.append(np.arange(len(rt)))
            RTs.append(rt)
            lRTs.append(np.log10(rt))
            RSs.append([x > np.median(rt) for x in rt])

            tdcPSDs.append(prePSD); tdc_post.append(postPSD)

        # Concatenate the data
        subj_id = np.concatenate(subj_id)
        ages = np.concatenate(ages)
        trl_id = np.concatenate(trl_id)
        RTs = np.concatenate(RTs)
        lRTs = np.concatenate(lRTs)
        RSs = np.concatenate(RSs)

        # Get the PSDs for the TDC MEG children
        X_tdc = np.concatenate(tdcPSDs)
        tdc_post = np.concatenate(tdc_post)

        # Now we want to create a dataframe with the data for all subjects
        analysis_data = pd.DataFrame()
        analysis_data['Subject'] = subj_id
        analysis_data['Age'] = ages
        analysis_data['Trial_ID'] = trl_id
        analysis_data['RT'] = RTs
        analysis_data['lRT'] = lRTs
        analysis_data['RS'] = RSs

        # Save the dataframe and X arrays to disk
        analysis_data.to_csv(os.path.join(outpath, 'analysis_data.csv'), index=False)
        np.save(os.path.join(outpath, 'X_tdc.npy'), X_tdc)
        np.save(os.path.join(outpath, 'tdc_post.npy'), tdc_post)

    else:

        # Load the data from disk
        analysis_data = pd.read_csv(os.path.join(outpath, 'analysis_data.csv'))
        X_tdc = np.load(os.path.join(outpath, 'X_tdc.npy'))
        tdc_post = np.load(os.path.join(outpath, 'tdc_post.npy'))


    ########################################
    #
    # Analytic code
    #
    ########################################

    # Run the PCA analysis for TDC fast/slow
    pca_analysis(analysis_data, X_tdc)