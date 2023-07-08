'''

Post-hoc analysis script for Closed-loop intracranial stimulation (CLeS) to assess 
behavioural response in terms of reaction time

@author: Nebras M. Warsi
PhD student, Ibrahim Lab
June 2021


'''

################
#
# Import block
#

# General imports
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# Stats
from scipy.stats import shapiro
from scipy.stats.stats import ttest_ind, f_oneway
from pymer4 import Lmer
from statsmodels.miscmodels.ordinal_model import OrderedModel
from pingouin import rm_anova # friedman can also be used (non-parametric)

# Options
pd.options.mode.chained_assignment = None  # default='warn'
matplotlib.use('Agg')
print_subj_rts = False

# Specify data paths and subject IDs
path = ""
save_path = ''
if not os.path.exists(save_path): os.makedirs(save_path)

patients = [

# Study identifiers removed for public access
# The list of participants to be analyzed is included in 
# this array
    
]

corr_comparison = ['Corr vs. Incorr.']

class CLeSSubject():


    '''

    CLeSSubject class. Class to load and analyze subject data 
    for CLeS experiments.

    '''


    def __init__(self, patient, path):


        '''
        Requires patient ID number to load relevant data

        '''

        self.id = patient
        self.pat_path = os.path.join(path, self.id, 'stim_data')
        self.Fs = 2048 if self.id == 'XXX' else 2000

        # Load demographic data
        with open((path + '/patient_dems.csv'), newline='\n') as csvfile:
            dem_data = pd.read_csv(csvfile, delimiter=',')
            self.age = float(dem_data[dem_data['ID'] == self.id]['Age'])

        print('Loading %s...' % self.id)


    def get_correct(self):


        '''
        Loads correct / incorrect trials

        '''

        with open(os.path.join(self.pat_path, 'trigs.txt'), newline='\n') as txtfile:
            data = pd.read_csv(txtfile, delimiter=',')
            
        trigs = data['Trig'].values
        times = data['Time'].values

        corrects = []

        for idx in range(len(trigs)):

            if trigs[idx] == 128:

                st = times[idx] + 0.8*self.Fs # Account for timing of intent wrt stimulus

                for resp_idx in np.arange(idx, len(trigs), step=1):

                    if ((trigs[resp_idx] == 1) | (trigs[resp_idx] == 2)):

                        rt = (times[resp_idx] - st)/self.Fs

                        if rt <= 5: corrects.append((trigs[resp_idx] - 1)) # Trial was not a "miss"

                        break

        return np.asarray(corrects)


    def get_RT(self):


        '''
        Loads RT data

        '''

        with open(os.path.join(self.pat_path, 'trigs.txt'), newline='\n') as txtfile:
            data = pd.read_csv(txtfile, delimiter=',')
            
        trigs = data['Trig'].values
        times = data['Time'].values

        RTs, miss_idx = [], []

        trl = 0
        for idx in range(len(trigs)):

            if trigs[idx] == 128:

                st = times[idx] + 0.8*self.Fs # Account for timing of intent wrt stimulus

                for resp_idx in np.arange(idx, len(trigs), step=1):

                    if ((trigs[resp_idx] == 1) | (trigs[resp_idx] == 2)):

                        rt = (times[resp_idx] - st)/self.Fs
                        
                        if rt <= 5:  # Makes sure trial was not a miss as this is the max RT
                            
                            RTs.append(rt)
                        
                        else:

                            miss_idx.append(trl)

                        trl += 1
                        break

        return np.asarray(RTs), np.asarray(miss_idx)


    def get_stims(self):


        '''
        Loads stimulation data from experiment

        '''

        with open(os.path.join(self.pat_path, 'stims.txt'), newline='\n') as txtfile:
            data = pd.read_csv(txtfile, delimiter=',')
        
        # Reads the newer file version
        if 'SUPPRESSED' in data['Stim'].values: # We are reading from the new files

            stims = []

            for s in data['Stim'].values:

                if s == 'SUPPRESSED':

                    stims.append(0)

                elif s == 'STIM':

                    stims.append(1)

                else:

                    stims.append(0)

            data['Stim'] = stims

        return data['Intent'].values, data['Stim'].values


def lme(data):

    lme_obj = Lmer(formula="RT ~ Stim + Age + Trial + (1|Subject)", data=data)
    res = lme_obj.fit()

    # Plot model summary
    coeff_plot = lme_obj.plot_summary(plot_intercept=False)

    return res, coeff_plot


def lme_nostim(data):

    lme_obj = Lmer(formula="RT ~ Intent + Age + Trial + (1|Subject)", data=data)
    res = lme_obj.fit()

    # Plot model summary
    coeff_plot = lme_obj.plot_summary(plot_intercept=False)

    return res, coeff_plot


def ord_reg(data):

    mod = OrderedModel(data['Quartile'], data['Stim'], distr='probit')
    res = mod.fit()

    return res


def comp_rmanova(data): # Not used 

    aov = rm_anova(dv='lRT', within='Stim', subject='Subject', data=data)
    
    return aov


def analyze(subjs):

    '''
    
    Analyzes and outputs CLeS data from individual subjects 
    or across multiple subjects for aggregate analysis

    '''

    # Loads individual subject data and combines for statistical analysis
    

    intent, stim, RT, \
        corrects, trl = [], [],\
                                [], [], []

    for subj in subjs:

        pt_RT, mi = subj.get_RT()
        RT.append(pt_RT)
        corrects.append(subj.get_correct())
        intent_subj, stim_subj = subj.get_stims()
        
        if len(mi): # Skip miss trials

            intent_subj = np.delete(intent_subj, mi)
            stim_subj = np.delete(stim_subj, mi)
                    
        intent.append(intent_subj)
        stim.append(stim_subj)
        
        trl.append([x for x in range(len(pt_RT))])

        assert len(stim_subj) == len(pt_RT), 'Error. RT and Trigger data lengths do not match for %s.\nPlease check.' % subj.id


    n_trials = [len(RT[x]) for x, _ in enumerate(subjs)]


    # DFs for analysis
    cles_data = pd.DataFrame()

    cles_data['Subject'] =  np.concatenate([[subj.id] * n_trials[x] for x, subj in enumerate(subjs)])
    cles_data['Age'] =   np.concatenate([[subj.age] * n_trials[x] for x, subj in enumerate(subjs)])
    cles_data['Trial'] = np.concatenate(trl)

    cles_data['Intent'] =  np.concatenate(intent)
    cles_data['Stim'] =  np.concatenate(stim)
    cles_data['IntentStim'] = cles_data['Intent'].values + cles_data['Stim'].values

    cles_data['Correct'] = np.concatenate(corrects)
    
    cles_data['RT'] = np.concatenate(RT)
    cles_data['rRT'] = np.concatenate([cles_data[cles_data['Subject'] == subj.id]['RT'].values 
                            - np.mean(cles_data[(cles_data['Subject'] == subj.id) & (cles_data['Stim'] == 0) & (cles_data['Correct'] == 1)]['RT'].values)
                            for subj in subjs])
    cles_data['lRT'] = np.concatenate([np.log(cles_data[cles_data['Subject'] == subj.id]['RT'].values) 
                            for subj in subjs])
    cles_data['rlRT'] = np.concatenate([np.log(cles_data[cles_data['Subject'] == subj.id]['RT'].values)
                            - np.mean(np.log(cles_data[(cles_data['Subject'] == subj.id) & (cles_data['Stim'] == 0) & (cles_data['Correct'] == 1)]['RT'].values))
                            for subj in subjs])

    ##########
    #
    # Output CLeS analysis 
    #

    for analysis in ['all']:


        print('\nAnalyzing...\n')

        data = cles_data if analysis == 'all' else cles_data[cles_data['Subject'] == analysis.id]
        out_path = save_path if analysis == 'all' else analysis.out_path

        # Plot proportion of correct/incorrect
        sns.pointplot(data=data, x='IntentStim', y='Correct', 
                        n_boot=200, ci=68)
        plt.ylabel("Proportion Correct"); plt.xlabel("Stimulation Condition")
        plt.xticks([0,1,2], ['No Intent', 'No Stim', 'Stim'])
        plt.savefig(os.path.join(out_path, 'stim_corr_plot.png'), dpi=300)
        plt.close()

        # Effects of stim on correct/incorrect
        print('\n************************************')
        print('Stim effects on correct/incorrect:')
        ftest = f_oneway(data[data['IntentStim'] == 0]['Correct'], data[data['IntentStim'] == 1]['Correct'], data[data['IntentStim'] == 2]['Correct'])
        print('Anova F-Test of Corr/Incorr: Statistic: %.3f P_value: %.3f' % (ftest[0], ftest[1]))

        # Split RT into quartiles
        quarts = []
        q1 = np.percentile(data['rRT'].values, 25)
        q2 = np.percentile(data['rRT'].values, 50)
        q3 = np.percentile(data['rRT'].values, 75)

        for rt in data['rRT'].values:

            if rt <= q1:
                quarts.append(1)
            elif rt <= q2:
                quarts.append(2)
            elif rt <= q3:
                quarts.append(3)
            else:
                quarts.append(4)
            
        # Save the quartile split of RT per patient
        data['Quartile'] = quarts

        # Split trial types into: intent, intent (no stim), and intent (stim)
        intent = data[data['Intent'] == 1]
        nostim_intent = data[data['IntentStim'] == 1]
        stim = data[data['IntentStim'] == 2]

        # Calculate and output basic descriptive stats for RT
        print('\n************************************')
        print('Crude quick look at RT:')
        print('Total Trials: %d' %  np.sum(n_trials))
        print('Stim Shapiro: %.3f' % shapiro(stim['rRT'])[1])
        print('No Stim Shapiro: %.3f' % shapiro(nostim_intent['rRT'])[1])
        ttest = ttest_ind(stim['rlRT'], nostim_intent['rlRT'], alternative="less")
        print('RT of Stim Trials: %.3f' % np.mean(stim['rRT']) + ' STDDev: %.3f' % np.std(stim['rRT']))
        print('RT of No Stim Trials: %.3f' % np.mean(nostim_intent['rRT']) + ' STDDev: %.3f' % np.std(nostim_intent['rRT']))
        print('T-Test of RT: Statistic: %.3f P_value: %.3f' % (ttest[0], ttest[1]))

        # Plot results
        print('\n' + '*****'*5)
        print('\nPlotting results...\n')
        print('*****'*5)

        # Relative RT plots by subject
        sns.pointplot(data=data, x='IntentStim', y='rRT', 
                        hue='Subject', n_boot=200,
                        ci=None, palette=sns.color_palette("crest", len(subjs)))
        plt.ylabel("RRT (S)"); plt.xlabel("Stimulation Condition"); plt.legend([])
        plt.xticks([0,1,2], ['No Intent', 'No Stim', 'Stim'])
        plt.savefig(os.path.join(out_path, 'stim_RT_plot.png'), dpi=300)
        plt.close()

        # Aggregate relative RT plot
        sns.pointplot(data=data, x='IntentStim', y='rRT', 
                        n_boot=200, ci=68)
        plt.ylabel("RRT (S)"); plt.xlabel("Stimulation Condition")
        plt.xticks([0,1,2], ['No Intent', 'No Stim', 'Stim'])
        plt.savefig(os.path.join(out_path, 'stim_RT_plot_agg.png'), dpi=300)
        plt.close()

        if print_subj_rts:

            # Print to screen each subject and the difference in between between stim and 
            # no stim trials
            print('\n' + '*****'*5)
            print('\nStim vs No Stim RT by subject:\n')
            print('*****'*5)
            for subj in subjs:

                stim_rt = np.mean(data[(data['Subject'] == subj.id) & (data['IntentStim'] == 2)]['RT'].values) * 1000
                nostim_rt = np.mean(data[(data['Subject'] == subj.id) & (data['IntentStim'] == 1)]['RT'].values) * 1000
                mean_rt = (np.mean(data[(data['Subject'] == subj.id) & (data['IntentStim'] != 2)]['RT'].values) * 1000)
                sd_rt = (np.std(data[(data['Subject'] == subj.id) & (data['IntentStim'] != 2)]['RT'].values) * 1000)
                diff = stim_rt - nostim_rt
                diff_bl = diff / mean_rt * 100

                print('Subject: %s' % subj.id)
                print('Stim RT: %.3f' % stim_rt)
                print('No Stim RT: %.3f' % nostim_rt)
                print('Baseline RT: %.3f +/- %.3f' % (mean_rt, sd_rt))
                print('Difference: %.3f' % diff)
                print('Difference (BL): %.3f' % diff_bl)
                print('')
        
        # Print total number of trials in each condition
        print('\n' + '*****'*5)
        print('\nNumber of trials in each condition:\n')
        print('*****'*5)
        print('Stim Trials: %d' %  len(stim))
        print('Withheld Trials: %d' %  len(nostim_intent))
        print('No Stim Trials: %d' %  len(data[data['IntentStim'] == 0]))

        # Stacked percentile plot of RT
        ax = pd.crosstab(data['IntentStim'], data['Quartile']).apply(lambda r: r/r.sum()*100, axis=1)
        ax.plot.bar(figsize=(10,10), stacked=True, rot=0, colormap='RdBu_r')
        plt.ylabel("Trial Proportion"); plt.xlabel("Stimulation Condition"); plt.legend([])
        plt.xticks([0,1,2], ['No Intent', 'No Stim', 'Stim'])
        plt.savefig(out_path + '/stim_quartile_CLeS.png', dpi=300)
        plt.close()
        
        # Agrregate statistical modelling
        if analysis == 'all':


            print('\nComputing aggregate LME...\n')
            res, coeff_plot = lme(intent)
            print(res)
            res.to_csv(os.path.join(out_path, 'RT_LME.csv'))
            coeff_plot.figure.savefig(os.path.join(out_path, 'RT_LME_plot.png'))
            plt.close()

            print('\nComputing aggregate LME of No Stim condition...\n')
            res, coeff_plot = lme_nostim(data[data['Stim'] == 0])
            print(res)
            res.to_csv(os.path.join(out_path, 'RT_NOSTIM_LME.csv'))

            print("\n\nDone!\n\n")


if __name__ == "__main__":

    print()
    print('*' * 50)
    print('\nWelcome to the CLeS posthoc analysis script')
    print('*' * 50)
    print()

    # Load individual subject objects
    subjs = [CLeSSubject(pat, path) for pat in patients]

    # Run aggregate and per-subject analysis
    analyze(subjs)