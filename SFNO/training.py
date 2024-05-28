import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from neuralop.datasets import load_spherical_swe
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam
# MinMaxScaler is sensitive to outliers, thus the StandardScaler was chosen!
from sklearn.preprocessing import StandardScaler # Scaling instead of Normalizing

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Here I pass the relevant model SFNO or FNO as an argument!
class Trainer:
    def __init__(self, model):

        self.model = model.to(device)
        self.history = []

        # Data has 3 channels thus for normalization I need to store the values for both training and testdata!
        self.scaler_in = []
        self.scaler_out = []

        self.test_res = [(32, 64), (64, 128)]
        self.train_set, self.test_sets, self.traindata_total, self.testsets_total, self.testsets_total_st = \
            self.assemble_data()

    @staticmethod
    def fit_scaler(scaler, tensor):
        # fit the scaler to standardize and rescale a tensor!
        n_channel = tensor.shape[1]
        for c in range(n_channel):
            reshaped_tensor = tensor[:,c,:,:].reshape(-1,1)
            scaler[c].fit(reshaped_tensor)

    @staticmethod
    def standardize(scaler, tensor):
        # Standardize the tensor and the underlying channel
        standardized_tensor = torch.zeros_like(tensor, dtype=torch.float32)
        n_channel = tensor.shape[1]
        for c in range(n_channel):
            reshaped_tensor = tensor[:,c,:,:].reshape(-1,1)
            standardized_c = scaler[c].transform(reshaped_tensor).reshape(standardized_tensor[:,c,:,:].shape)
            standardized_tensor[:,c,:,:] = torch.from_numpy(standardized_c)
        # return newly standardized tensor over all channels!
        return standardized_tensor

    @staticmethod
    def inv_standardize(scaler, tensor):
        # Scale the tensor back (inverse operation of standardizing)
        inv_tensor = torch.zeros_like(tensor, dtype=torch.float32)
        n_channel = tensor.shape[1]
        for c in range(n_channel):
            reshaped_tensor = tensor[:,c,:,:].reshape(-1,1)
            inv_c = scaler[c].inverse_transform(reshaped_tensor).reshape(inv_tensor[:,c,:,:].shape)
            inv_tensor[:,c,:,:] = torch.from_numpy(inv_c)
        return inv_tensor
        
    
    """
    FNO in 2D requires the following shape, thus some data manipulation is necessary!
    input: the solution of the coefficient function and locations (a(x, y), x, y)
    input shape: (batchsize, x=s, y=s, c=3)
    """
    def assemble_data(self):
        # Load the Navier Stokes dataset with a resolution of 128 x 128
        # Since I need to rearange everything again, I can choose a batch_size of 1!
        train_loader, test_loaders = load_spherical_swe(n_train=200, 
                                                        batch_size=1, 
                                                        train_resolution=(32, 64),
                                                        test_resolutions=self.test_res, 
                                                        n_tests=[50, 50], 
                                                        test_batch_sizes=[1, 1],)

        """
        TRAIN_LOADER
        In general the input shape is: (bs, channels, res_x res_y)
        train_loader shape: (1, 3, 32, 64)
        n_train: 200 training samples
        bs: 4 batch_size
        len(train_loader): 200 / 4 = 50
        """
        train_in = torch.zeros((200, 3, 32, 64), dtype=torch.float32)
        train_out = torch.zeros((200, 3, 32, 64), dtype=torch.float32)

        for batch_idx, batch in enumerate(train_loader):
            train_in[batch_idx,:,:,:] = batch["x"]
            train_out[batch_idx,:,:,:] = batch["y"]

        train_shape = train_in.shape
        # train_out_shape = train_out.shape

        # Fill up the scaler for the input and output for each channel!
        n_channels = train_shape[1]
        for _ in range(n_channels):
            self.scaler_in.append(StandardScaler())
            self.scaler_out.append(StandardScaler())

        # First fit the scaler:
        self.fit_scaler(self.scaler_in, train_in)
        self.fit_scaler(self.scaler_out, train_out)

        # Now standardize the input data!
        train_in_standardized = self.standardize(self.scaler_in, train_in)
        train_out_standardized = self.standardize(self.scaler_out, train_out)
        
        training_data = {"input": train_in.permute(0,2,3,1), "output" : train_out.permute(0,2,3,1)}

        # Create the dataloader:
        training_set = DataLoader(TensorDataset(train_in_standardized.permute(0,2,3,1), train_out_standardized.permute(0,2,3,1)), 
                                  batch_size=4, shuffle=False)
        """
        TEST_LOADERS
        len(test_loader): 2 # resolution_1, resolution_2
        resolution_1: (32, 64)
        resolution_2: (64, 128)
        n_train: 50 training samples
        bs: 10 batch_size
        len(train_loader[(32, 64)]): 50 / 10 = 5
        len(train_loader[(64, 128)]): 50 / 10 = 5
        shape: (10, 3, resolution_i[0], resolution_i[1])
        """
        # Dict. to store dataloader for each test resolution:
        test_in_res1 = torch.zeros((50, 3, self.test_res[0][0], self.test_res[0][1]), dtype=torch.float32)
        test_in_res2 = torch.zeros((50, 3, self.test_res[1][0], self.test_res[1][1]), dtype=torch.float32)

        test_out_res1 = torch.zeros_like(test_in_res1, dtype=torch.float32)
        test_out_res2 = torch.zeros_like(test_in_res2, dtype=torch.float32)

        test_dict_in = {self.test_res[0] : test_in_res1, self.test_res[1] : test_in_res2}
        test_dict_out = {self.test_res[0] : test_out_res1, self.test_res[1] : test_out_res2}

        for res in self.test_res:
            for batch_idx, batch in enumerate(test_loaders[res]):
                test_dict_in[res][batch_idx,:,:,:] = batch["x"]
                test_dict_out[res][batch_idx,:,:,:] = batch["y"]

        # Standardize the testing data as well!
        test_dict_in_st = {self.test_res[0] : self.standardize(self.scaler_in, test_dict_in[self.test_res[0]]), 
                                     self.test_res[1] : self.standardize(self.scaler_in, test_dict_in[self.test_res[1]])}
        
        test_dict_out_st = {self.test_res[0] : self.standardize(self.scaler_out, test_dict_out[self.test_res[0]]), 
                                     self.test_res[1] : self.standardize(self.scaler_out, test_dict_out[self.test_res[1]])}

        testing_data = {"input" : test_dict_in, "output" : test_dict_out}
        testing_data_st = {"input" : test_dict_in_st, "output" : test_dict_out_st}

        testing_sets =  {
                self.test_res[0] : DataLoader(TensorDataset(test_dict_in_st[self.test_res[0]].permute(0,2,3,1), 
                                                    test_dict_out_st[self.test_res[0]].permute(0,2,3,1)), 
                                                    batch_size=10, shuffle=False), 
                self.test_res[1] : DataLoader(TensorDataset(test_dict_in_st[self.test_res[1]].permute(0,2,3,1), 
                                                    test_dict_out_st[self.test_res[1]].permute(0,2,3,1)), 
                                                    batch_size=10, shuffle=False)
                        }
        return training_set, testing_sets, training_data, testing_data, testing_data_st
    
    # Plot the input data to understand it better!
    def plot_training_data(self, batch_idx = 0):
        train_inp = self.traindata_total['input'].numpy()
        train_out = self.traindata_total['output'].numpy()
        
        # channels: u, v, h 
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle('Visualization of the Training Dataset')
        images = []
        for c in range(3):
            im = axs[0,c].imshow(train_inp[batch_idx,:,:,c], cmap='viridis')
            # fig.colorbar(im, ax=ax)
            images.append(im)
            axs[0,c].label_outer()
            axs[0,c].set_title(f'Input: Batch {batch_idx}, Channel {c}')
            # axs[0,c].set_xlabel('X-axis')
            # axs[0,c].set_ylabel('Y-axis')

            im = axs[1,c].imshow(train_out[batch_idx,:,:,c], cmap='viridis')
            images.append(im)
            axs[1,c].label_outer()
            axs[1,c].set_title(f'Output: Batch {batch_idx}, Channel {c}')
        
        fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=0.05, pad=0.05)
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()
    
    def train(self, epochs, learning_rate, step_size, gamma):
        
        optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        l = torch.nn.MSELoss()
        freq_print = 1

        for epoch in range(epochs):
            train_mse = 0.0
            for step, (input_batch, output_batch) in enumerate(self.train_set):
                optimizer.zero_grad()
                pred_output = self.model(input_batch)
                gt_output = output_batch
                loss_f = l(pred_output, gt_output)
                loss_f.backward()
                optimizer.step()
                train_mse += loss_f.item()
                # add to history to plot the training loss
                self.history.append(loss_f.item())
            train_mse /= len(self.train_set)
            scheduler.step()

            with torch.no_grad():
                self.model.eval()
                test_relative_l2 = 0.0
                for res in self.test_res:
                    test_relative_res = 0.0
                    for step, (input_batch, output_batch) in enumerate(self.test_sets[res]):
                        pred_output = self.model(input_batch)
                        gt_output = output_batch
                        loss_f = (torch.mean((pred_output - gt_output) ** 2) / torch.mean(gt_output ** 2)) ** 0.5 * 100
                        test_relative_res += loss_f.item()
                    test_relative_res /= len(self.test_sets[res])
                    test_relative_l2 += test_relative_res

            if epoch % freq_print == 0: print("######### Epoch:", epoch, " ######### Train Loss:", train_mse, 
                                              " ######### Relative L2 Test Norm:", test_relative_l2)
    
    def predict(self):
        # permute the results to (bs, x, y, c)
        test_in_res1 = self.testsets_total_st["input"][self.test_res[0]].permute(0,2,3,1)
        test_in_res2 = self.testsets_total_st["input"][self.test_res[1]].permute(0,2,3,1)

        # gt_out_res1 = self.testsets_total["output"][self.test_res[0]].permute(0,2,3,1)
        # gt_out_res2 = self.testsets_total["output"][self.test_res[1]].permute(0,2,3,1)

        pred_res1 = self.model(test_in_res1).detach() # in training distribution prediction
        pred_res2 = self.model(test_in_res2).detach() # out of distribution prediction
        
        # scale and permute the predictions to (bs, c, x, y)
        pred_res1 = self.inv_standardize(self.scaler_out, pred_res1.permute(0,3,1,2))
        pred_res2 = self.inv_standardize(self.scaler_out, pred_res2.permute(0,3,1,2))

        # now permute the predictions back to (bs, x, y, c)
        pred_res1 = pred_res1.permute(0,2,3,1)
        pred_res2 = pred_res2.permute(0,2,3,1)
        
        return pred_res1.numpy(), pred_res2.numpy()
    
    def plot_trainloss(self):
        plt.plot(np.linspace(0,len(self.history)-1,len(self.history),dtype=int), self.history)
        plt.xlabel('Iteration')
        plt.ylabel('Training Loss')
        plt.grid(True)
        plt.show()
    
    def plot_fno_sfno(self, pred_fno, pred_sfno, batch_idx=0, res="in-dist"):
        # First get the groundtruth (gt) for both resolutions:
        # Shape (bs, x, y, c)
        gt_out_res1 = self.testsets_total["output"][self.test_res[0]].permute(0,2,3,1).numpy()
        gt_out_res2 = self.testsets_total["output"][self.test_res[1]].permute(0,2,3,1).numpy()
        if res == "in-dist":
            gt_out = gt_out_res1
            resolution = self.test_res[0]
        else:
            gt_out = gt_out_res2
            resolution = self.test_res[1]

        # channels: u, v, h 
        fig, axs = plt.subplots(3, 3, figsize=(12, 8))
        fig.suptitle(f"Prediction Results with Resolution {resolution}")
        images = []

        for c in range(3):
            im = axs[c,0].imshow(gt_out_res1[batch_idx,:,:,c], cmap='viridis')
            # fig.colorbar(im, ax=ax)
            images.append(im)
            axs[c,0].label_outer()
            axs[c,0].set_title(f'Groundtruth: Channel {c}')
            # axs[0,c].set_xlabel('X-axis')
            # axs[0,c].set_ylabel('Y-axis')

            im = axs[c,1].imshow(pred_fno[batch_idx,:,:,c], cmap='viridis')
            images.append(im)
            axs[c,1].label_outer()
            axs[c,1].set_title(f'FNO: Channel {c}')

            im = axs[c,2].imshow(pred_sfno[batch_idx,:,:,c], cmap='viridis')
            images.append(im)
            axs[c,2].label_outer()
            axs[c,2].set_title(f'SFNO: Channel {c}')

        
        fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=0.05, pad=0.05)
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()


    def errplot_fno_sfno(self, pred_fno, pred_sfno, batch_idx=0, res="in-dist"):
        # First get the groundtruth (gt) for both resolutions:
        # Shape (bs, x, y, c)
        gt_out_res1 = self.testsets_total["output"][self.test_res[0]].permute(0,2,3,1).numpy()
        gt_out_res2 = self.testsets_total["output"][self.test_res[1]].permute(0,2,3,1).numpy()
        if res == "in-dist":
            gt_out = gt_out_res1
            resolution = self.test_res[0]
        else:
            gt_out = gt_out_res2
            resolution = self.test_res[1]

        # channels: u, v, h 
        fig, axs = plt.subplots(3, 2, figsize=(12, 8))
        fig.suptitle(f"Error of each Model with a Resolution of {resolution}")
        images = []

        for c in range(3):
            # Compute errors for sfno and fno:
            err_fno = np.abs(gt_out[batch_idx,:,:,c] - pred_fno[batch_idx,:,:,c])
            err_sfno = np.abs(gt_out[batch_idx,:,:,c] - pred_sfno[batch_idx,:,:,c])
            im = axs[c,0].imshow(err_fno, cmap='viridis')
            # fig.colorbar(im, ax=ax)
            images.append(im)
            axs[c,0].label_outer()
            axs[c,0].set_title(f'FNO: Channel {c}')

            im = axs[c,1].imshow(err_sfno, cmap='viridis')
            images.append(im)
            axs[c,1].label_outer()
            axs[c,1].set_title(f'SFNO: Channel {c}')

        
        fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=0.05, pad=0.05)
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()

    @staticmethod
    def compute_channel_error(gt, pred):
        err_model = 0
        for c in range(3):
            err_model += np.mean(np.abs(gt - pred))
        err_model /= 3
        return err_model
            
    def scaling_study(self, pred_fno1, pred_fno2, pred_sfno1, pred_sfno2):
        # First get the groundtruth (gt) for both resolutions:
        # Shape (bs, x, y, c)
        gt_out_res1 = self.testsets_total["output"][self.test_res[0]].permute(0,2,3,1).numpy()
        gt_out_res2 = self.testsets_total["output"][self.test_res[1]].permute(0,2,3,1).numpy()

        # error calculation over all batches!
        err_fno1 = self.compute_channel_error(gt_out_res1, pred_fno1)
        err_fno2 = self.compute_channel_error(gt_out_res2, pred_fno2)
        err_sfno1 = self.compute_channel_error(gt_out_res1, pred_sfno1)
        err_sfno2 = self.compute_channel_error(gt_out_res2, pred_sfno2)
        
        # channels: u, v, h 
        plt.plot(np.linspace(1,2,2,dtype=int),[err_fno1, err_fno2],label='FNO', marker='o')
        plt.plot(np.linspace(1,2,2,dtype=int),[err_sfno1, err_sfno2],label='SFNO', marker='o')
        plt.xlabel('Resolution')
        plt.ylabel('Aggregated Error')
        plt.xticks(np.linspace(1,2,2,dtype=int), self.test_res)
        plt.legend()
        plt.grid(True)
        plt.show()
