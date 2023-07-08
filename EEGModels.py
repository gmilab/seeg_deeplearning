'''

EEGNet_PSD model capable of predicting attentional lapses in real-time 
based on the original EEGNet architecture (Lawhern et al., J Neural Eng, 2018). 
Credit for origianl model design is attributable to the authors of that study.

This script runs an updated model based on EEGNet, but adapted to learn from 
power spectral density (PSD) rather than broadband EEG signal. This approach 
1) mitigates susceptbility to EEG spectral drifts (if the signal is appropriately 
baselined) and 2) facilitates model explanations based on changes in signal power.

Nebras M. Warsi,
Ibrahim Lab
Hospital for Sick Children


'''


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

    
def EEGNet_PSD(Chans, Samples, dropoutRate = 0.50, 
                        F1 = 4, D = 2, mode = 'multi_channel',):
    
    input1   = Input(shape = (Chans, Samples, 1))

    ##################################################################

    block1       = Conv2D(F1, (1, 10), use_bias = False, padding = 'same')(input1)
    block1       = BatchNormalization()(block1)

    if mode == 'multi_channel':
        
        block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                    depth_multiplier = D,
                                    depthwise_constraint = max_norm(1.))(block1)
        block1       = BatchNormalization()(block1)

    block1       = Activation('relu')(block1)
    block1       = AveragePooling2D((1, 2), padding = 'valid')(block1) # 8 is also good
    block1       = Dropout(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F1*D, (1, 4), use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('relu')(block2)
    block2       = AveragePooling2D((1, 2), padding = 'valid')(block2) # Can be used
    block2       = Dropout(dropoutRate)(block2)
    
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(1, name = 'dense')(flatten)
    out      = Activation('sigmoid', name = 'sigmoid')(dense)
    
    return Model(inputs=input1, outputs=out)
