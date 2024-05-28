import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_pred_results(t_train, Tf0, Ts0, out_predict_t, out_predict_Tf0, out_predict_Ts0):

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # First subplot: Fluid Tf0
    axs[0].plot(t_train, Tf0, label='Measurement Data')
    axs[0].plot(out_predict_t, out_predict_Tf0, label='Predictions')
    axs[0].grid(True)
    axs[0].legend()
    axs[0].set_title("Predictions Fluid $T_{f0}$")
    axs[0].set_xlabel("Time t in s")
    axs[0].set_ylabel("Temperature T in K")

    # Second subplot: Fluid Ts0
    axs[1].plot(t_train, Ts0, label='Measurement Data')
    axs[1].plot(out_predict_t, out_predict_Ts0, label='Predictions')
    axs[1].grid(True)
    axs[1].legend()
    axs[1].set_title("Predictions Solid $T_{s0}$")
    axs[1].set_xlabel("Time t in s")
    axs[1].set_ylabel("Temperature T in K")

    plt.tight_layout()
    plt.show()

def validate(fno_Tf, fno_Ts):
    # Use the testing data to evaluate the model's behavior afterwards
    val_set = fno_Tf.testing_set.dataset
    inp_Tf0 = val_set.tensors[0]
    out_Tf0 = val_set.tensors[1]

    val_set = fno_Ts.testing_set.dataset
    inp_Ts0 = val_set.tensors[0]
    out_Ts0 = val_set.tensors[1]

    n_data = inp_Tf0.shape[0] + fno_Tf.windowsize - 1 # Number of datapoints
    val_res = torch.zeros((n_data, 2), dtype=torch.float32) # store normalized results
    val_res_scaled = torch.zeros((n_data, 2), dtype=torch.float32) # store scaled results

    pred_Tf0 = fno_Tf.fno(inp_Tf0[0,:,:].expand(1, -1, -1)).reshape(-1,)
    pred_Ts0 = fno_Ts.fno(inp_Ts0[0,:,:].expand(1, -1, -1)).reshape(-1,)

    val_res[:fno_Tf.windowsize,:] = torch.cat((pred_Tf0.reshape(-1,1), pred_Ts0.reshape(-1,1)),1)
    val_res_scaled[:fno_Tf.windowsize,:] = torch.cat((fno_Tf.scale(pred_Tf0, fno_Tf.max_val).reshape(-1,1), 
                                                      fno_Tf.scale(pred_Ts0, fno_Tf.max_val).reshape(-1,1)), 1)

    for batch_i in range(inp_Tf0.shape[0]):
        # skip the first batch!
        if batch_i == 0:
            continue
        # for further predictions store the last value!
        idx_val = fno_Tf.windowsize + batch_i - 1
        pred_Tf0 = fno_Tf.fno(inp_Tf0[batch_i,:,:].expand(1, -1, -1)).reshape(-1,)
        pred_Ts0 = fno_Ts.fno(inp_Ts0[batch_i,:,:].expand(1, -1, -1)).reshape(-1,)

        val_res[idx_val,:] = torch.cat((pred_Tf0[-1].reshape(-1,1), pred_Ts0[-1].reshape(-1,1)),1)
        val_res_scaled[idx_val,:] = torch.cat((fno_Tf.scale(pred_Tf0[-1], fno_Tf.max_val).reshape(-1,1), 
                                                fno_Tf.scale(pred_Ts0[-1], fno_Tf.max_val).reshape(-1,1)), 1)
        
    # Further idea since results are slightly different for each prediction I could average them!

    # Now plot the results:
    t = fno_Tf.t_train[1:]# scaled times

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # Plot the first subplot (Tf0)
    axs[0].plot(t, val_res_scaled[:, 0].detach().numpy(), label="Prediction")
    axs[0].plot(t, fno_Tf.Tf0[1:], label="Groundtruth")
    axs[0].grid(True, which="both", ls=":")
    axs[0].set_xlabel("Time t in s")
    axs[0].set_xlim([0, t[-1] + 10000])
    axs[0].set_ylim([torch.min(val_res_scaled).detach().numpy() - 10, 
                        torch.max(val_res_scaled).detach().numpy() + 10])
    axs[0].set_ylabel("Temperature T in K")
    axs[0].set_title("Validation Results for $T_{f0}$")
    axs[0].legend()

    # Plot the second subplot (Ts0)
    axs[1].plot(t, val_res_scaled[:, 1].detach().numpy(), label="Prediction")
    axs[1].plot(t, fno_Ts.Ts0[1:], label="Groundtruth")
    axs[1].grid(True, which="both", ls=":")
    axs[1].set_xlabel("Time t in s")
    axs[1].set_xlim([0, t[-1] + 10000])
    axs[1].set_ylim([torch.min(val_res_scaled).detach().numpy() - 10, 
                        torch.max(val_res_scaled).detach().numpy() + 10])
    axs[1].set_ylabel("Temperature T in K")
    axs[1].set_title("Validation Results for $T_{s0}$")
    axs[1].legend()

    plt.tight_layout()  # Adjust subplot parameters to give specified padding
    plt.show()