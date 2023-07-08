## EEGNet based classifier script for EEG:RT
##
##
## Nebras M. Warsi
## Ibrahim Lab
## June 2021

## General imports
import sys
import os
import h5py
import csv
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
from tqdm import tqdm
from datafuncs import *
## ML imports
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model 
from EEGModels import EEGNet_PSD
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
## HP tuning libraries
import ray
from ray import tune
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest import ConcurrencyLimiter

######################################################################
########
#######
#
# Initialize hyper-parameter search and relevant training parameters  
#
#######
########

config = {

    'n_chans': tune.choice([4, 8, 12, 16])

}


# Training Parameters
k = 5
cv = StratifiedKFold(n_splits=k, shuffle=True)
epochs = 250
batch_size = 32
opt = tf.keras.optimizers.Adam()
n_samples = 40
# Data Path
path = '' # Path for data loading

# For Channel Selection
Q = Queue()

sel_mode = 'LR' # Channel selection function (EEGNet or LR)

######
#
# Structure of network based on ARL EEGNet architechture
#
######

def ieeg_net(n_chans, n_samples, mode='multi_channel'):    

    # Initialize model and the input shape required
    model = EEGNet_PSD(Samples=n_samples, Chans=n_chans, mode=mode)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC')])
    return model

#######
#
# Identifies the most predicitive N contacts 
# for inclusion into EEGNet multi-contact classifier
#
######

def select_chans(X, Y, contacts, coords, tt):

    print('*' * 50)
    print('\nRanking channels!...\n')
    print('*' * 50)

    scores, auc_scores = [], []

    if sel_mode == 'EEGNet':

        # EEGNet Version
        print('\nRanking channels based on EEGNet Algorithm...\n')

        # Updated channel selection algo using DL
        for cidx, _ in enumerate(tqdm(contacts)):

            # Set up model
            model = ieeg_net(n_chans=1, n_samples=n_samples, mode='single_channel')

            if np.isnan(X[:, cidx, :]).any():
            
                scores.append(0)
                continue

            sel_accs, sel_aucs = [], []

            for _, (train_indices, val_indices) in enumerate(cv.split(X,Y)):  

                # Split into train and val data
                x_train, x_val = X[train_indices][:,cidx,:], X[val_indices][:,cidx,:]
                y_train, y_val = Y[train_indices], Y[val_indices]
                # Expand dims for EEGNet analysis: [Trials, Contacts, Timepoints, and then Kernels=1]
                x_train, x_val = np.expand_dims(np.expand_dims(x_train, axis=-1), axis=1), np.expand_dims(np.expand_dims(x_val, axis=-1), axis=1)

                model.fit(x_train, y_train, batch_size=batch_size, 
                        validation_data=(x_val, y_val), epochs=epochs,
                        callbacks=[EarlyStopping(monitor='val_accuracy', patience=10)],
                        verbose=0)

                y_hat = model.predict(x_val)
                y_pred = model.predict(x_val) > 0.5

                # Metric calculation
                trial_auc = roc_auc_score(y_val, y_hat)
                trial_acc = accuracy_score(y_val, y_pred)

                # Save to trial
                sel_accs.append(trial_acc)
                sel_aucs.append(trial_auc)

            scores.append(np.mean(sel_accs))
            auc_scores.append(np.mean(sel_aucs))

    else:

        # LR Version
        print('\nRanking channels based on LR Algorithm...\n')

        for cidx, _ in enumerate(contacts):

            X_select = X[:, cidx, :]

            if np.isnan(X_select).any():
            
                scores.append(0)
                continue

            X_select = StandardScaler().fit_transform(X_select)
            X_select = SelectKBest(k=10).fit_transform(X_select, Y)

            sel = cross_val_score(estimator=LogisticRegression(C=2e-4), X=X_select, y=Y, cv=cv, scoring='accuracy', n_jobs=-1)
            sel_auc = cross_val_score(estimator=LogisticRegression(C=2e-4), X=X_select, y=Y, cv=cv, scoring='roc_auc', n_jobs=-1)

            # Find feature importance for all channels to rank
            scores.append(np.mean(sel))
            auc_scores.append(np.mean(sel_auc))

    # Save single-channel metrics
    full_save_path = os.path.join(save_path, "%s_single_channel_AUC.csv" % tt)
    with open(full_save_path, 'a', newline='') as f:
        
        writer = csv.writer(f)

        if f.tell() == 0:
            # First time writing to file. Write header row.
            writer.writerow(['Contact', 'AUC', 'MNI_Coord'])

        for i in range(len(contacts)):
            writer.writerow([contacts[i], auc_scores[i], coords[i]])

    Q.put(auc_scores)
     
######
#
# HP Tuning function
#
#####

def tune_net(config, x=None, y=None, scores=None, checkpoint_dir=None):
    
    # Allows us to test models using either all contacts together
    # or a subset of the best based on screening
    n_chans = config['n_chans']

    top_conts = np.sort(np.argpartition(scores, -n_chans)[-n_chans:])

    # Optimization result variables and predictions
    val_accs = []
    val_aucs = []
    val_losses = []

    # k-fold CV
    for fold, (train_indices, val_indices) in enumerate(cv.split(x,y)):  

        # Set up model
        model = ieeg_net(n_chans=n_chans, n_samples=n_samples)

        # Split into train and val data
        x_train, x_val = x[train_indices][:,top_conts,:], x[val_indices][:,top_conts,:]
        y_train, y_val = y[train_indices], y[val_indices]

        # Expand dims for EEGNet analysis: [Trials, Contacts, Timepoints, and then Kernels=1]
        x_train, x_val = np.expand_dims(x_train, axis=-1), np.expand_dims(x_val, axis=-1)

        tune_model = os.path.join(val_models, '%s_val_%i.json' % (tune.get_trial_id(), fold + 1))
        history = model.fit(x_train, y_train, batch_size=batch_size, 
                validation_data=(x_val, y_val), epochs=epochs,
                 callbacks=[ModelCheckpoint(tune_model, monitor='val_accuracy', 
                                                    save_best_only=True, save_weights_only=True)])
        model.load_weights(tune_model)

        y_hat = model.predict(x_val)
        y_pred = model.predict(x_val) > 0.5

        # Metric calculation
        trial_auc = roc_auc_score(y_val, y_hat)
        trial_acc = accuracy_score(y_val, y_pred)
        trial_loss = history.history['val_loss'][-1]

        # Save to trial
        val_accs.append(trial_acc)
        val_aucs.append(trial_auc)
        val_losses.append(trial_loss)

        tune.report(score=np.mean(val_accs), Auc=np.mean(val_aucs), Acc=np.mean(val_accs))

    tune.report(score=np.mean(val_accs), Auc=np.mean(val_aucs), Acc=np.mean(val_accs))


######################################################################
##########
########
#
#
# Main Section
#
#
#######
########

if __name__ == "__main__":

    try:
        patient = sys.argv[1]
        tt = sys.argv[2]

    except:
        print("Error. Please specifiy participant file and trial type!")
        print("Usage: python EEGNet_psd.py PARTICIPANT_ID [shift/nonshift]")
        exit()

    # Loads data paths and then gets dataset from the H5PY file
    path_to_load = os.path.join(path, patient, 'processed') # FIRST SESSION (TRAIN)
    day_two_path = os.path.join(path, patient, 'processed_day_two') # SECOND SESSION (TEST)

    scratch_space = os.getcwd()

    print()
    print('#' * 50)
    print('#' * 50)
    print('\nWelcome!')
    print('\nThis script will train ML classifiers to predict attentional performance.')
    print('\nLoading data for %s (%s trials)\n' % (patient, tt))
    print('#' * 50)
    print('#' * 50)

    # # Toggle test code to see model size/params
    # model = ieeg_net(n_chans=8, n_samples=n_samples)
    # print(model.summary())
    # exit()

    #########
    #
    # Load data
    #
    #

    dataset = h5py.File(os.path.join(path_to_load, "processed_data.h5"), 'r')
    X_d1 = dataset.get('psd_shift')[()] if tt == 'shift' else dataset.get('psd_nonshift')[()]
    Y_d1 = dataset.get('Y_class_s')[()] if tt == 'shift' else dataset.get('Y_class_ns')[()]
    contacts, coords = dataset.attrs['contacts'], dataset.attrs['coords']
    dataset.close()

    # Load additional session data if available
    if os.path.isdir(day_two_path):

        dataset = h5py.File(os.path.join(day_two_path, "processed_data.h5"), 'r')
        X_d2 = dataset.get('psd_shift')[()] if tt == 'shift' else dataset.get('psd_nonshift')[()]
        Y_d2 =  dataset.get('Y_class_s')[()] if tt == 'shift' else dataset.get('Y_class_ns')[()]
        contacts_test = dataset.attrs['contacts']
        dataset.close()

        # Add into main dataset to see how many trials total
        X_all = np.concatenate((X_d1, X_d2), axis=0)
        Y_all = np.concatenate((Y_d1, Y_d2), axis=0)

    # Confirm valid data shapes
    print('\nData loaded successfully!')
    print('EEG Dataset: %s' % str(X_all.shape))
    print('%s Trials: %s' % (tt.capitalize(), str(Y_all.shape)))
    print('Contacts: %s' % str(contacts.shape))

    # Define and create save paths
    save_path = os.path.join(path, patient, 'EEGNet_%s' % tt)
    # Add an incremental number to the end of the save path if it already exists
    if not os.path.isdir(save_path): os.makedirs(save_path)
    else:
        i = 1
        while os.path.isdir(save_path + '_' + str(i)):
            i += 1
        save_path = save_path + '_' + str(i)
        os.makedirs(save_path)

    val_models = os.path.join(save_path, 'val_models/')
    clinical_models = os.path.join(save_path, 'clinical_model/')
    if not os.path.isdir(val_models): os.makedirs(val_models)
    if not os.path.isdir(clinical_models): os.makedirs(clinical_models)

    #######
    #
    # Train/test split data
    #

    # One day is the train set and other the test
    X_train, Y_train = X_d1, Y_d1
    X_test, Y_test = X_d2, Y_d2

    # Now we will get the top electrodes for the prediction 
    topn_process = Process(target=select_chans, args=(X_train, Y_train, contacts, coords, tt))
    topn_process.start()
    topn_process.join()
    scores = Q.get()

    ######
    #
    # Hyperparameter tuning 
    #
    #######

    ray_dir = os.path.join(scratch_space, tt)
    if not os.path.isdir(ray_dir): os.makedirs(ray_dir)
                
    ray.init()
    algo = TuneBOHB(metric='score', mode='max')
    algo = ConcurrencyLimiter(algo, max_concurrent=2)
    bohb = HyperBandForBOHB(
        time_attr='training_iteration',
        metric='score',
        mode='max'
    )
    exp_name = '%s_tune_%s' % (patient, tt)
    analysis = tune.run(tune.with_parameters(tune_net, x=X_train, y=Y_train, scores=scores, checkpoint_dir=ray_dir), 
                            name=exp_name, config=config, search_alg=algo, scheduler=bohb, num_samples=3, 
                            local_dir=os.path.join(scratch_space, 'ray_results/'))

    best_hps = analysis.get_best_config(metric='score', mode='max')
    best_results = analysis.get_best_trial(metric='score', mode='max').last_result
    ray.shutdown()

    # Best HPs
    n_chans =  best_hps.get('n_chans')
    top_conts = np.sort(np.argpartition(scores, -n_chans)[-n_chans:])

    ############################################
    #####
    #
    # Test result
    #
    ####

    # If more than one session has been loaded, ensure contacts are in same order
    if os.path.isdir(day_two_path): assert np.asarray([contacts[x] == contacts_test[x] for x in range(len(contacts))]).all(), "Contacts list not equivalent! Please check."

    # Select channels and expand dims for EEGNet analysis: [Trials, Contacts, Timepoints, and then Kernels=1]
    X_train = np.expand_dims(X_train[:,top_conts,:], axis=-1)            
    X_test = np.expand_dims(X_test[:,top_conts,:], axis=-1)

    clin_model_path = os.path.join(clinical_models, '%s_validated.h5' % tt)

    # Compile and train model
    monitor = 'val_accuracy'
    model = ieeg_net(n_chans, n_samples)
    clin_history = model.fit(X_train, Y_train, 
                batch_size=batch_size, epochs=1000,
                validation_data=(X_test, Y_test),
                callbacks=[ModelCheckpoint(clin_model_path, monitor=monitor, 
                                save_best_only=True, mode='max')])

    model = load_model(clin_model_path)

    # Plots Loss
    plt.plot(clin_history.history['loss'])
    plt.plot(clin_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(save_path, 'clin_loss.png'))
    plt.close()

    # Plots AUC
    plt.plot(clin_history.history['auc'])
    plt.plot(clin_history.history[monitor])
    plt.title('model AUC')
    plt.ylabel('AUC')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(save_path, 'clin_auc.png'))
    plt.close()

    # Plots Accuracy
    plt.plot(clin_history.history['accuracy'])
    plt.plot(clin_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(save_path, 'clin_acc.png'))
    plt.close()

    Y_hat = model.predict(X_test)
    Y_pred = model.predict(X_test) > 0.5

    acc = accuracy_score(Y_test, Y_pred)
    prec = precision_score(Y_test, Y_pred, average='weighted')
    rec = recall_score(Y_test, Y_pred, average='weighted')
    f1 = f1_score(Y_test, Y_pred, average='weighted')

    # AUC calculation
    auc = roc_auc_score(Y_test, Y_pred)
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
    plt.plot(fpr, tpr, marker='.', color='orange', label="AUC: %.2f" % auc)
    plt.legend(loc=4)
    plt.savefig(os.path.join(save_path, '%s_clinical_AUC.png' % tt))
    plt.close()
    
    clinical_cm = confusion_matrix(Y_test, Y_pred)
    plot_confusion_matrix(cm=clinical_cm, target_names = ['fast', 'slow'], path = save_path,
        mode='%s_clinical' % tt, title='Confusion matrix', normalize=False)
    
    # Save test metrics
    full_save_path = os.path.join(save_path, "%s_EEGNet_clinical.csv" % tt)
    with open(full_save_path, 'a', newline='') as f:
        writer = csv.writer(f)

        if f.tell() == 0:
            # First time writing to file. Write header row.
            writer.writerow(['Contacts', 'Coords', 'AUC', 'Acc', 
                                    'Precision', 'Recall', 'F1'])

        data = [contacts[top_conts], coords[top_conts], auc, acc, prec, rec, f1]
        writer.writerow(data)

    ###### FINAL MODEL SAVING
    # Compile and save model trained on all available data to deploy
    n_epochs = np.argmax(clin_history.history['val_accuracy'])

    X_final = np.concatenate((X_train, X_test), axis=0)
    Y_final = np.concatenate((Y_train, Y_test), axis=0)

    cles_model = os.path.join(clinical_models, '%s_CLeS' % tt)

    model = ieeg_net(n_chans, n_samples)
    model.fit(X_final, Y_final, batch_size=batch_size, epochs=n_epochs)

    model.save(cles_model)