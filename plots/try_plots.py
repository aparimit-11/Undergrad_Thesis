# Author: Laura Kulowski
from plotting import *
from model_dyn import *
from utils_dyn import *

import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse



'''

Example of using a LSTM encoder-decoder to model a synthetic time series 

'''

import numpy as np
import matplotlib
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from importlib import reload
import sys

matplotlib.rcParams.update({'font.size': 17})

parser = argparse.ArgumentParser()
args=parser.parse_args()

DATA_FILE = "/Users/aparimitkasliwal/Desktop/2201_IITD/BTP/data/RESIN_MH_001.csv"

# window dataset

# set size of input/output windows 
iw = 90
ow = 10
s = 1
train_fraction = 0.8

train_df, val_df, test_df = train_val_test_split(DATA_FILE,0.8)

traindataset = CustomDataset(train_df,iw,ow,s)
valdataset = CustomDataset(val_df,iw,ow,s)
testdataset = CustomDataset(test_df,iw,ow,s)

# generate windowed training/test datasets

Xtrain = traindataset.X 
Ytrain = traindataset.Y
Xtest = testdataset.X
Ytest = testdataset.Y
# Xval = valdataset.X
# Yval = valdataset.Y

# plot example of windowed data  
# idx = 0
# plt.figure(figsize = (10, 6)) 
# plt.plot(np.arange(0, iw), Xtrain[:, idx, 0], 'k', linewidth = 2.2, label = 'Input')
# plt.plot(np.arange(iw - 1, iw + ow), np.concatenate([Xtrain[-1:, idx, 0], Ytrain[:, idx, 0]]),
#          color = (0.2, 0.42, 0.72), linewidth = 2.2, label = 'Target')
# plt.xlim([0, iw + ow - 1])
# plt.xlabel(r'$t$')  
# plt.ylabel(r'$y$')
# plt.title('Example of Windowed Training Data')
# plt.legend(bbox_to_anchor=(1.3, 1))
# plt.tight_layout() 
# plt.savefig('plots/windowed_data.png')


#----------------------------------------------------------------------------------------------------------------
# LSTM encoder-decoder

#convert windowed data from np.array to PyTorch tensor
#X_train, Y_train, X_test, Y_test = numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest)

model = torch.load("/Users/aparimitkasliwal/Desktop/2201_IITD/BTP/code_files/dyn_teacher_forcing/snapshots/mixed_teacher_forcing0.8_90_10_1_50/epoch_40")
model.eval()

model1 = torch.load("/Users/aparimitkasliwal/Desktop/2201_IITD/BTP/code_files/dyn_teacher_forcing/snapshots/mixed_teacher_forcing0.8_60_7_1_50/epoch_60")
model1.eval()

model2 = torch.load("/Users/aparimitkasliwal/Desktop/2201_IITD/BTP/code_files/dyn_teacher_forcing/snapshots/mixed_teacher_forcing0.8_100_7_1_50/epoch_60")
model2.eval()

model3 = torch.load("/Users/aparimitkasliwal/Desktop/2201_IITD/BTP/code_files/dyn_teacher_forcing/snapshots/mixed_teacher_forcing0.8_50_7_1_50/epoch_80")
model3.eval()

# specify model parameters and train
#model_1 = lstm_seq2seq(input_size = X_train.shape[2], hidden_size = 15)
#loss = model_1.train_model(X_train, Y_train, n_epochs = 50, target_len = ow, batch_size = 5, training_prediction = 'mixed_teacher_forcing', teacher_forcing_ratio = 0.6, learning_rate = 0.01, dynamic_tf = False)

# plot predictions on train/test data
plot_train_test_results(model, Xtrain, Ytrain, Xtest, Ytest)
plot_train_test_results(model1, Xtrain, Ytrain, Xtest, Ytest)
plot_train_test_results(model2, Xtrain, Ytrain, Xtest, Ytest)
plot_train_test_results(model3, Xtrain, Ytrain, Xtest, Ytest)

plt.close('all')






