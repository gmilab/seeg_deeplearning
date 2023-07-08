# seeg_deeplearning

Complete repository of deep learning and analysis scripts for 
real-time prediction and closed-loop control of lapses in 
attention. 

Description

Machine learning and EEG processing:
------------------------------------
-- EEGModels.py: Model definition for the EEGNET_PSD classifier
-- EEGNet_psd.py: Complete data loading and machine learning script
-- process_cles.py: Pre-processing script to convert raw data to stimulus-locked PSD
-- datafuncs.py: Collection of back-end processing functions

Eye tracking:
------------------------------------
-- cles_et.py: Analysis of stimulation effects on eye-tracking
-- process_et.py: Eye tracking pre-processing script

Effects of closed-loop stimulation on task RT
------------------------------------
-- cles_rt.py

MEG analysis
------------------------------------
-- process_meg.py

TMS-EEG analysis
------------------------------------
-- TMS.py
