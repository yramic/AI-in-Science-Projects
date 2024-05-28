import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import pandas as pd

from fno import FNO1d
from spectralconv import SpectralConv1d

torch.manual_seed(0)
np.random.seed(0)


class FNO_Trainer:
    def __init__(self, modes, width, windowsize, model,
                 data_train = 'Data/TrainingData.txt',
                 data_test = 'Data/TestingData.txt'
                 ):
        
        self.data_train = data_train
        self.data_predict = data_test
        self.windowsize = windowsize
        self.model = model
        
        self.fno = FNO1d(modes, width) # model for the fluid or solid

        # Train and Test data
        self.t_train, self.Tf0, self.Ts0, self.t_test = self.load_data()

        # Values for normalization and scaling:
        self.T_final = self.t_test[-1]

        # The difference between max(Tf) and max(Ts) is so small that it's alright to normalize with the biggest value
        self.max_val = torch.max(torch.cat((self.Tf0.reshape(-1,1), self.Ts0.reshape(-1,1)), 1))

        (self.training_set, self.training_set_Tf, self.training_set_Ts, 
         self.testing_set, self.testing_set_Tf, self.testing_set_Ts, self.dataset) = self.assemble_training_data()


    def load_data(self):
        current_directory = os.getcwd()
        train_file = os.path.join(current_directory, self.data_train)
        predict_file = os.path.join(current_directory, self.data_predict)

        table_train = np.loadtxt(train_file, delimiter=',', dtype=np.float64, skiprows=1)
        table_predict = np.loadtxt(predict_file, delimiter=',', dtype=np.float64, skiprows=1)

        # Training file:
        time_train = torch.from_numpy(table_train[:,0]).float() # First row
        Tf0_data = torch.from_numpy(table_train[:,1]).float() # Second row
        Ts0_data = torch.from_numpy(table_train[:,2]).float() # Thrid row

        # Testing file:
        time_predict = torch.from_numpy(table_predict[:]).float()

        return time_train, Tf0_data, Ts0_data, time_predict
    
    @staticmethod
    def normalize(data, max):
        return data / max

    @staticmethod
    def scale(data, max):
        return data * max

    @staticmethod
    def shifting_window(inp_tensor, windowsize):
        # This function returns the dataset in batches with a stride of 1 by default!
        """
        Further explanation:
        timeseries data with 100 points can be divided into patches with a window size of 10:
        1. patch: [0, 0:9, 2]
        2. patch: [1, 1:10, 2]
        ...
        n. patch: [n, 90:99, 2]
        with n = 100 - 10 = 90
        Total size of the input dataset will thus be: [n+1, windowsize, input dimension]
        This will result in a mapping from a set of 10 datapoints to another set of 10 datapoints
        """
        # import ipdb; ipdb.set_trace()
        assert inp_tensor.dim() == 3, "The input tensor must have 3 dimensions (bs = 1, data, input dimension)"
        # Extract information from the dataset:
        inp_dim = inp_tensor.shape[2] # input dimensions
        n_data = inp_tensor.shape[1] # Total number of datapoints
        assert inp_tensor.shape[0] == 1, "The initial batchsize for the time series data should be 1!"
        bs = n_data - windowsize + 1
        tensor = torch.zeros((bs,windowsize,inp_dim),dtype=torch.float32)

        # Now the task is to fill up the tensor and return the results:
        for batch_i in range(bs):
            start_idx = batch_i
            end_idx = batch_i + windowsize 
            tensor[batch_i,:,:] = inp_tensor[:,start_idx:end_idx,:]

        return tensor
    

    def assemble_training_data(self):
        # N x 2 input and N x 1 output vector where N = 209
        # Define an Input and Output function:
        inp_Tf = torch.zeros((self.t_train.reshape(-1,1).shape[0] - 1, 2))
        inp_Ts = torch.zeros((self.t_train.reshape(-1,1).shape[0] - 1, 2))
        # Note that the input is for the previous time step thus starting from 0!
        inp_Tf[:,0] = self.normalize(self.t_train[:-1], self.T_final) 
        inp_Tf[:,1] = self.normalize(self.Tf0[:-1], self.max_val)
        inp_Ts[:,0] = self.normalize(self.t_train[:-1], self.T_final) 
        inp_Ts[:,1] = self.normalize(self.Ts0[:-1], self.max_val)

        out_Tf = torch.zeros((inp_Tf.shape[0], 1))
        out_Ts = torch.zeros((inp_Ts.shape[0], 1))
        # Note that the output is for the predicted next step thus starting from 1!
        # output_function[:,0] = train_data_obj.t[1:]
        out_Tf[:,0] = self.normalize(self.Tf0[1:], self.max_val)
        out_Ts[:,0] = self.normalize(self.Ts0[1:], self.max_val)

        inp_Tf = inp_Tf.expand(1, -1, -1) # Batchsize: 1 otherwise this will be a problem!
        input_function_Tf = self.shifting_window(inp_Tf, self.windowsize)

        inp_Ts = inp_Ts.expand(1, -1, -1) # Batchsize: 1 otherwise this will be a problem!
        input_function_Ts = self.shifting_window(inp_Ts, self.windowsize)

        out_Tf = out_Tf.expand(1, -1, -1) # Batchsize: 1 otherwise this will be a problem!
        output_function_Tf = self.shifting_window(out_Tf, self.windowsize)

        out_Ts = out_Ts.expand(1, -1, -1) # Batchsize: 1 otherwise this will be a problem!
        output_function_Ts = self.shifting_window(out_Ts, self.windowsize)

        # Create DataLoader
        batch_size = 11 # otherwise this will become more difficult!
        # No shuffling, since we are dealing with Time Series Data!
        training_set_Tf = DataLoader(TensorDataset(input_function_Tf, output_function_Tf), batch_size=batch_size, shuffle=False)
        training_set_Ts = DataLoader(TensorDataset(input_function_Ts, output_function_Ts), batch_size=batch_size, shuffle=False)

        # Testing DataLoader
        testing_set_Tf = DataLoader(TensorDataset(input_function_Tf, output_function_Tf), batch_size=batch_size, shuffle=False)
        testing_set_Ts = DataLoader(TensorDataset(input_function_Ts, output_function_Ts), batch_size=batch_size, shuffle=False)

        if self.model == 'fluid':
            training_set = DataLoader(TensorDataset(input_function_Tf, output_function_Tf), batch_size=batch_size, shuffle=False)
            testing_set = DataLoader(TensorDataset(input_function_Tf, output_function_Tf), batch_size=batch_size, shuffle=False)

        elif self.model == 'solid':
            training_set = DataLoader(TensorDataset(input_function_Ts, output_function_Ts), batch_size=batch_size, shuffle=False)
            testing_set = DataLoader(TensorDataset(input_function_Ts, output_function_Ts), batch_size=batch_size, shuffle=False)

        else:
            raise ValueError("Invalid model specified. Expected 'fluid' or 'solid'.")

        # Dataset for Plotting:
        dataset = DataLoader(TensorDataset(inp_Tf, out_Tf, inp_Ts, out_Ts), batch_size=batch_size, shuffle=False)

        return training_set, training_set_Tf, training_set_Ts, testing_set, testing_set_Tf, testing_set_Ts, dataset
    

    def plot_data(self):
        
        training_data = self.dataset.dataset
        input_function_Tf = training_data[0][0]
        output_function_Tf = training_data[0][1]
        input_function_Ts = training_data[0][2]
        output_function_Ts = training_data[0][3]

        t = input_function_Tf[:,0].reshape(-1,1)

        plt.plot(self.scale(t[:,0], self.T_final), self.scale(output_function_Tf, self.max_val).reshape(-1,), label = "$T_{f0}$ : $x = 0$")
        plt.plot(self.scale(t[:,0], self.T_final), self.scale(output_function_Ts, self.max_val).reshape(-1,), label = "$T_{s0}$ : $x = 0$")
        plt.grid(True, which="both", ls=":")
        plt.xlabel("Time t in s")
        plt.xlim([self.scale(t[0,0], self.T_final) -10000, self.scale(t[-1,0], self.T_final) + 10000])
        plt.ylim([self.scale(torch.min(torch.cat((output_function_Tf,output_function_Ts), 1)), self.max_val) - 10, 
                  self.scale(torch.max(torch.cat((output_function_Tf, output_function_Ts), 1)), self.max_val) + 10])
        plt.ylabel("Temperature in K")
        plt.title("Visual Representation of the Noisefree Data")
        plt.legend()
        plt.show()


    def train(self, epochs, learning_rate, step_size, gamma):

        optimizer = Adam(self.fno.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        l_T = torch.nn.MSELoss()
        freq_print = 1
        for epoch in range(epochs):
            train_mse = 0.0
            for step, (input_batch, output_batch) in enumerate(self.training_set):
                optimizer.zero_grad()
                pred_output_batch = self.fno(input_batch).squeeze(2)
                output_batch = output_batch.squeeze(2)
                # Another Idea is to separate the two loss functions!
                loss_f = l_T(pred_output_batch, output_batch)
                loss_f.backward()
                optimizer.step()
                train_mse += loss_f.item()
            train_mse /= len(self.training_set)
            scheduler.step()

            with torch.no_grad():
                self.fno.eval()
                test_relative_l2 = 0.0
                for step, (input_batch, output_batch) in enumerate(self.testing_set):
                    pred_output_batch = self.fno(input_batch).squeeze(2)
                    output_batch = output_batch.squeeze(2)
                    loss_f = (torch.mean((pred_output_batch - output_batch) ** 2) / torch.mean(output_batch ** 2)) ** 0.5 * 100
                    test_relative_l2 += loss_f.item()
                test_relative_l2 /= len(self.testing_set)

            if epoch % freq_print == 0: print("######### Epoch:", epoch, " ######### Train Loss:", train_mse, 
                                            " ######### Relative L2 Test Norm:", test_relative_l2)

         
    def predict(self):
        """
        Make predictions into the future for the dataset provided (self.t_test is the prediction dataset!)

        First we have to create our prediction dataset, since our model expects 2 inputs, time and a Tf0 or Ts0 value
        The overall approach will be iterative because we still map from a set to another set and gain predictions for those 
        results!
        """
        if self.model == 'fluid':
            T0 = self.Tf0
        elif self.model == 'solid':
            T0 = self.Ts0
        else:
            raise ValueError("Invalid model specified. Expected 'fluid' or 'solid'.")
        
        t_predict = self.normalize(self.t_test, self.T_final) # Noramlization of future timesteps
        t_old = self.normalize(self.t_train[1:], self.T_final) # Normalize previous results
        T0_old = self.normalize(T0[1:], self.max_val)

        out_predict = torch.zeros((t_predict.shape[0],1)) # noramlized results for Tf0 and Ts0
        out_predict_scaled = torch.zeros((t_predict.shape[0],1)) # scaled results for Tf0 or Ts0

        inp_predict = torch.zeros((self.windowsize, 2))
        inp_predict[:,:] = torch.cat((t_old[-self.windowsize:].reshape(-1,1), 
                                      T0_old[-self.windowsize:].reshape(-1,1)), 1)
        
        # Reshape the inputs into the size (1, windowsize, 2)
        inp_predict = inp_predict.expand(1, -1, -1)
        
        for i in range(t_predict.shape[0]):
            pred_T0 = self.fno(inp_predict).reshape(-1,)

            out_predict[i,0] = pred_T0[-1]
            out_predict_scaled[i,0] = self.scale(pred_T0[-1], self.max_val)

            new_input_T0 = torch.cat((t_predict[i].reshape(-1,1), pred_T0[-1].detach().reshape(-1,1)), 1)
            # update new input!
            inp_predict = torch.cat((inp_predict[0,1:,:], new_input_T0), 0).expand(1, -1, -1)

        # for plotting we want to have the last point as well included for the predictions!
        out_predict_T0 = torch.zeros((out_predict_scaled.shape[0]+1,))
        out_predict_T0[0] = T0[-1]
        out_predict_T0[1:] = out_predict_scaled[:,0]

        out_predict_t = torch.zeros((out_predict_scaled.shape[0]+1,))
        out_predict_t[0] = self.t_train[-1]
        out_predict_t[1:] = self.t_test

        return out_predict_t, out_predict_T0
    