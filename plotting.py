import argparse
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import math
from utils_dyn import *

def plot_train_test_results(model, Xtrain, Ytrain, Xtest, Ytest, num_rows = 4):
  '''
  plot examples of the lstm encoder-decoder evaluated on the training/test data
  
  : param lstm_model:     trained lstm encoder-decoder
  : param Xtrain:         np.array of windowed training input data
  : param Ytrain:         np.array of windowed training target data
  : param Xtest:          np.array of windowed test input data
  : param Ytest:          np.array of windowed test target data 
  : param num_rows:       number of training/test examples to plot
  : return:               num_rows x 2 plots; first column is training data predictions,
  :                       second column is test data predictions
  '''

  # input window size
  iw = Xtrain.shape[0]
  ow = Ytest.shape[0]
  s = 1

  # figure setup 
  num_cols = 2
  num_plots = num_rows * num_cols

  fig, ax = plt.subplots(num_rows, num_cols, figsize = (13, 15))
  
  parser = argparse.ArgumentParser() 
  args=parser.parse_args()

  BATCH_SIZE = 1
  DATA_FILE = "/Users/aparimitkasliwal/Desktop/2201_IITD/BTP/data/RESIN_MH_001.csv"
  train_df, val_df, test_df = train_val_test_split(DATA_FILE,0.8)
  traindataset = CustomDataset(train_df,iw,ow,s)
  trainloader = DataLoader(traindataset, batch_size=BATCH_SIZE, shuffle=True)
  testdataset = CustomDataset(test_df,iw,ow,s)
  testloader = DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)


  Y_train_pred = predict(model, trainloader, ow)
  Y_test_pred = predict(model, testloader, ow)
  #print(Xtrain.shape)

  #Y_test_pred = model.predict(torch.from_numpy(X_test_plt).type(torch.Tensor), target_len = ow)

  # plot training/test predictions
  for ii in range(num_rows):
      
      ax[ii, 0].plot(np.arange(0, iw), Xtrain[:, ii, 0], 'k', linewidth = 2, label = 'Input')
      ax[ii, 0].plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtrain[-1, ii, 0]], Ytrain[:, ii, 0]]),
                     color = (0.2, 0.42, 0.72), linewidth = 2, label = 'Target')
    #   print("Hello1")
    #   print(Y_train_pred[ii,:,0,0].shape)
    #   print("Hello2")
    #   print(Xtrain[:, ii, 0].shape)
    #   print("Hello3")
    #   print(Ytrain[:, ii, 0].shape)
      ax[ii, 0].plot(np.arange(iw - 1, iw + ow),np.concatenate([[Xtrain[-1, ii, 0]],Y_train_pred[ii,:,0,0]]),
                     color = (0.76, 0.01, 0.01), linewidth = 2, label = 'Prediction')
      ax[ii, 0].set_xlim([0, iw + ow - 1])
      ax[ii, 0].set_xlabel('$t$')
      ax[ii, 0].set_ylabel('$y$')

      # test set
      X_test_plt = Xtest[:, ii, :]
      
    #   print([Xtest[-1, ii, 0]])
    #   print("Fuck you")
    #   print(Y_test_pred[ii,:,0])

      ax[ii, 1].plot(np.arange(0, iw), Xtest[:, ii, 0], 'k', linewidth = 2, label = 'Input')
      ax[ii, 1].plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtest[-1, ii, 0]], Ytest[:, ii, 0]]),
                     color = (0.2, 0.42, 0.72), linewidth = 2, label = 'Target')
      ax[ii, 1].plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtest[-1, ii, 0]], Y_test_pred[ii,:,0,0]]),
                     color = (0.76, 0.01, 0.01), linewidth = 2, label = 'Prediction')
      ax[ii, 1].set_xlim([0, iw + ow - 1])
      ax[ii, 1].set_xlabel('$t$')
      ax[ii, 1].set_ylabel('$y$')

      if ii == 0:
        ax[ii, 0].set_title('Train')
        
        ax[ii, 1].legend(bbox_to_anchor=(1, 1))
        ax[ii, 1].set_title('Test')

  #plt.show()
  plt.suptitle('LSTM Encoder-Decoder Predictions', x = 0.445, y = 1.)
  plt.tight_layout()
  plt.subplots_adjust(top = 0.95)
  plt.savefig('plots/predictions.png')
  plt.close() 
  
  return 

#git_check

