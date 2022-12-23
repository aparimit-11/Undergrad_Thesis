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
import argparse

# parser = argparse.ArgumentParser() 
# args=parser.parse_args()

# TARGET_LEN = args.ow

class CustomDataset(Dataset) : 
    def __init__(self,df,input_window,output_window,stride=1) : 
        self.df = df
        self.input_window = input_window
        self.output_window = output_window
        self.stride = stride
        self.num_features = len(df.columns)  ##ignore date in df
        self.windowed_data()

    def __len__(self) : 
        return self.num_samples

    def __getitem__(self,idx) :
        return torch.tensor(self.X[:,idx,:]),torch.tensor(self.Y[:,idx,:])

    def windowed_data(self) : 
        L = self.df.shape[0]
        self.num_samples = (L - self.input_window - self.output_window) // (self.stride + 1)

        self.X = np.zeros([self.input_window,self.num_samples,self.num_features])
        self.Y = np.zeros([self.output_window,self.num_samples,self.num_features])    
        
        z = self.df.to_numpy()

        for ff in tqdm(np.arange(self.num_features),desc='Constructing Windowed Data'):
            for ii in np.arange(self.num_samples):
                start_x = self.stride * ii
                end_x = start_x + self.input_window
                self.X[:,ii,ff] = z[start_x:end_x, ff]

                start_y = self.stride * ii + self.input_window
                end_y = start_y + self.output_window 
                self.Y[:,ii,ff] = z[start_y:end_y, ff]


def train_val_test_split(data_file,train_fraction=0.8) : 

    column_name = ["dt"]
    df = pd.read_csv('/Users/aparimitkasliwal/Desktop/2201_IITD/BTP/data/RESIN_MH_001.csv', sep='|', names = column_name)
    df = df.drop([df.index[0], df.index[1], df.index[2]])
    df['dt'].str.split(',', expand=True)
    df[['DATE', 'LEVEL', 'STORAGE']] = df['dt'].str.split(',', expand=True)
    df = df.drop('dt', axis=1)
    df = df.drop([df.index[0]])
    df['LEVEL'] = df['LEVEL'].astype(float)
    df['STORAGE'] = df['STORAGE'].astype(float)
    df = df.reset_index()
    df = df.drop('index',axis=1)
    df = df.drop('DATE',axis=1)

    training_size=int(len(df)*train_fraction)
    val_size=int(0.5*(len(df)-training_size))
    return df.iloc[0:training_size],df.iloc[training_size:training_size+val_size],df.iloc[training_size+val_size:]


def evaluate(model, dataloader, lossfunction, target_len) : 

    model.eval()
    avg_batch_loss = 0.0
    n_batches = 0

    for (input_batch,target_batch) in dataloader : 
        
        input_batch = input_batch.view(input_batch.shape[1],input_batch.shape[0],input_batch.shape[2])
        target_batch = target_batch.view(target_batch.shape[1],target_batch.shape[0],target_batch.shape[2])

        outputs = model.predict(input_batch.float(),target_len)
        # compute the loss 
        loss = lossfunction(outputs, target_batch.float())
        avg_batch_loss += loss.item()

        n_batches+=1

        # loss for epoch 
        avg_batch_loss /= n_batches 

    return avg_batch_loss


def predict(model, dataloader, target_len) : 
    output = []
    model.eval()

    for (input_batch,target_batch) in dataloader : 
        
        input_batch = input_batch.view(input_batch.shape[1],input_batch.shape[0],input_batch.shape[2])
        target_batch = target_batch.view(target_batch.shape[1],target_batch.shape[0],target_batch.shape[2])
        
        
        outputs = torch.zeros(target_len, input_batch.shape[1], input_batch.shape[2])

        # initialize hidden state
        encoder_hidden = model.encoder.init_hidden(input_batch.shape[1])

        # encoder outputs
        encoder_output, encoder_hidden = model.encoder(input_batch.float())

        # decoder with teacher forcing
        decoder_input = input_batch[-1, :, :].float()   # shape: (batch_size, input_size)
        decoder_hidden = encoder_hidden

        for t in range(target_len): 
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            decoder_input = decoder_output

        output.append(outputs.detach().numpy())


    return  np.array(output)

def set_deterministic():
    np.random.seed(1729)
    torch.manual_seed(1729)
    random.seed(1729)
    torch.backends.cudnn.benchmark = False

def numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest):
    '''
    convert numpy array to PyTorch tensor
    
    : param Xtrain:                           windowed training input data (input window size, # examples, # features); np.array
    : param Ytrain:                           windowed training target data (output window size, # examples, # features); np.array
    : param Xtest:                            windowed test input data (input window size, # examples, # features); np.array
    : param Ytest:                            windowed test target data (output window size, # examples, # features); np.array
    : return X_train_torch, Y_train_torch,
    :        X_test_torch, Y_test_torch:      all input np.arrays converted to PyTorch tensors 
    '''
    
    X_train_torch = torch.from_numpy(Xtrain).type(torch.Tensor)
    Y_train_torch = torch.from_numpy(Ytrain).type(torch.Tensor)

    X_test_torch = torch.from_numpy(Xtest).type(torch.Tensor)
    Y_test_torch = torch.from_numpy(Ytest).type(torch.Tensor)
    
    return X_train_torch, Y_train_torch, X_test_torch, Y_test_torch
