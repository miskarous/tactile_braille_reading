#!/usr/bin/env python
# coding: utf-8

# ## Notebook intended to run the working networks for Braille reading

# In[1]:


import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import random
import json

from sklearn.metrics import confusion_matrix


# ### Don't forget to select the threshold you want to work with and if you want to use pre-trained weights or train from scratch

# In[2]:


# set variables
# multiple_gpus = True # set to 'True' if more than 1 GPU available
use_nni_weights = True # set to 'True' for use of weights from NNI optimization 
use_seed = False # set seed to achive reproducable results
threshold = 1 # possible values are: 1, 2, 5, 10
run = "_6" # run number for statistics
epochs = 300 # 300 # set the number of epochs you want to train the network here


# In[3]:


if torch.cuda.device_count()>1:
    gpu_sel = 1
    gpu_av = [torch.cuda.is_available() for ii in range(torch.cuda.device_count())]
    print("Detected {} GPUs. The load will be shared.".format(torch.cuda.device_count()))
    if True in gpu_av:
        if gpu_av[gpu_sel]:
            device = torch.device("cuda:"+str(gpu_sel))
        else:
            device = torch.device("cuda:"+str(gpu_av.index(True)))
        # torch.cuda.set_per_process_memory_fraction(0.25, device=device)
    else:
        device = torch.device("cpu")
else:
    if torch.cuda.is_available():
        print("Single GPU detected. Setting up the simulation there.")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("No GPU detected. Running on CPU.")


# In[4]:


if use_seed:
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Seed set to {}".format(seed))
else:
    print("Shuffle data randomly")


# In[5]:


dtype = torch.float


# In[6]:


letters = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


# In[7]:


# replace torch.tile() by torch_tile()
def torch_tile(data, reps):
    np_tile = np.tile(data.cpu().detach().numpy(),reps)
    return torch.tensor(np_tile)


# In[ ]:


def load_and_extract_augmented(params, file_name, taxels=None, letter_written=letters):
    """ From extract_data_icub_events by Alejandro"""
    
    max_time = int(54*25) #ms
    time_bin_size = int(params['time_bin_size']) # ms
    global time
    time = range(0,max_time,time_bin_size)
    ## Increase max_time to make sure no timestep is cut due to fractional amount of steps
    global time_step
    time_step = time_bin_size*0.001
    data_steps = len(time)
    
    infile = open(file_name, 'rb')
    data_dict = pickle.load(infile)
    infile.close()
    # Extract data
    data = []
    labels = []
    bins = 1000  # [ms] 1000 ms in 1 second
    nchan = len(data_dict[1]['events']) # number of channels/sensors
    for i, sample in enumerate(data_dict):
        dat = (sample['events'][:])
        events_array = np.zeros([nchan,round((max_time/time_bin_size)+0.5),2])
        for taxel in range(len(dat)):
            for event_type in range(len(dat[taxel])):
                if dat[taxel][event_type]:
                    indx = bins*(np.array(dat[taxel][event_type]))
                    indx = np.array((indx/time_bin_size).round(), dtype=int)
                    events_array[taxel,indx,event_type] = 1
        if taxels != None:
            events_array = np.reshape(np.transpose(events_array, (1,0,2))[:,taxels,:],(events_array.shape[1],-1))
            selected_chans = 2*len(taxels)
        else:
            events_array = np.reshape(np.transpose(events_array, (1,0,2)),(events_array.shape[1],-1))
            selected_chans = 2*nchan
        data.append(events_array)
        labels.append(letter_written.index(sample['letter']))
        
    # return data,labels
    data = np.array(data)
    labels = np.array(labels)

    data = torch.tensor(data, dtype=dtype )    
    labels = torch.tensor(labels,dtype=torch.long)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, shuffle=True, stratify=labels, random_state=42) # if fix seed wanted add: random_state=42

    ds_train = TensorDataset(x_train,y_train)
    ds_test = TensorDataset(x_test,y_test)
    
    return ds_train, ds_test, labels, selected_chans, data_steps


# In[ ]:


def run_snn(inputs, layers):

    bs = inputs.shape[0]
    # h1_from_input = torch.einsum("abc,cd->abd", (inputs.tile((nb_input_copies,)), layers[0]))
    h1_from_input = torch.einsum("abc,cd->abd", (torch_tile(inputs,(nb_input_copies,)).to(device), layers[0]))
    syn = torch.zeros((bs,nb_hidden), device=device, dtype=dtype)
    mem = torch.zeros((bs,nb_hidden), device=device, dtype=dtype)

    out = torch.zeros((bs, nb_hidden), device=device, dtype=dtype)

    # Here we define two lists which we use to record the membrane potentials and output spikes
    mem_rec = []
    spk_rec = []

    # Compute hidden (recurrent) layer activity
    for t in range(nb_steps):
        h1 = h1_from_input[:,t] + torch.einsum("ab,bc->ac", (out, layers[2]))
        
        # LK: leak and integrate
        new_syn = alpha*syn + h1
        new_mem = beta*mem + new_syn
        # new_mem = beta*mem + new_syn*(1 - spk_rec[-1]) if t != 0 else alpha*syn + h1

        # LK: fire
        mthr = new_mem-1.0
        out = spike_fn(mthr)
        rst = out.detach() # We do not want to backprop through the reset
        
        mem = new_mem*(1.0-rst)
        syn = new_syn

        mem_rec.append(mem)
        spk_rec.append(out)

    # Now we merge the recorded membrane potentials into a single tensor
    mem_rec = torch.stack(mem_rec,dim=1)
    spk_rec = torch.stack(spk_rec,dim=1)

    # Readout layer
    h2 = torch.einsum("abc,cd->abd", (spk_rec, layers[1]))
    flt = torch.zeros((bs,nb_outputs), device=device, dtype=dtype)
    out = torch.zeros((bs,nb_outputs), device=device, dtype=dtype)
    s_out_rec = [out] # out is initialized as zeros, so it is fine to start with this
    out_rec = [out]
    for t in range(nb_steps):        
        # LK: leak and integrate
        new_flt = alpha*flt + h2[:,t]
        new_out = beta*out + new_flt
            
        # LK: fire
        mthr_out = new_out-1.0
        s_out = spike_fn(mthr_out)
        rst_out = s_out.detach()

        flt = new_flt
        out = new_out*(1.0-rst_out)

        out_rec.append(out)
        s_out_rec.append(s_out)

    out_rec = torch.stack(out_rec,dim=1)
    s_out_rec = torch.stack(s_out_rec,dim=1)
    other_recs = [mem_rec, spk_rec, s_out_rec]
    layers_update = layers

    return out_rec, other_recs, layers_update


# In[10]:


def load_layers(file, map_location, requires_grad=True, variable=False):
    
    if variable:
        
        lays = file
        
        for ii in lays:
            ii.requires_grad = requires_grad
    
    else:
        
        lays = torch.load(file, map_location=map_location)
    
        for ii in lays:
            ii.requires_grad = requires_grad
        
    return lays


# In[11]:


### Here, this function is only used to define the global variables to be used in other functions
def build(params):
    
    global nb_input_copies
    nb_input_copies = params['nb_input_copies']  # Num of spiking neurons used to encode each channel 

    # Network parameters
    global nb_inputs
    nb_inputs  = nb_channels*nb_input_copies
    global nb_hidden
    nb_hidden  = 450
    global nb_outputs
    nb_outputs = len(np.unique(labels))+1
    global nb_steps
    nb_steps = data_steps

    tau_mem = params['tau_mem'] # ms
    tau_syn = tau_mem/params['tau_ratio']
    
    global alpha
    alpha   = float(np.exp(-time_step/tau_syn))
    global beta
    beta    = float(np.exp(-time_step/tau_mem))

    fwd_weight_scale = params['fwd_weight_scale']
    rec_weight_scale = params['weight_scale_factor']*fwd_weight_scale 
    


# In[12]:


def build_and_predict(params, x, use_nni_weights):
    
    x = x.to(device)
    
    global nb_input_copies
    nb_input_copies = params['nb_input_copies']  # Num of spiking neurons used to encode each channel 

    # Network parameters
    global nb_inputs
    nb_inputs  = nb_channels*nb_input_copies
    global nb_hidden
    nb_hidden  = 450
    global nb_outputs
    nb_outputs = len(np.unique(labels))+1
    global nb_steps
    nb_steps = data_steps

    tau_mem = params['tau_mem'] # ms
    tau_syn = tau_mem/params['tau_ratio']
    
    global alpha
    alpha   = float(np.exp(-time_step/tau_syn))
    global beta
    beta    = float(np.exp(-time_step/tau_mem))

    fwd_weight_scale = params['fwd_weight_scale']
    rec_weight_scale = params['weight_scale_factor']*fwd_weight_scale 
    
    # Spiking network
    if use_nni_weights:
        layers = load_layers('../NNI/SpyTorch_layers/best_test_'+file_type+'_thr_'+str(file_thr)+'_ref_'+str(file_ref)+'_'+optim_nni_experiment+'.pt', map_location=device)
    else:
        layers = []

        w1 = torch.empty((nb_inputs, nb_hidden),  device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(w1, mean=0.0, std=fwd_weight_scale/np.sqrt(nb_inputs))
        layers.append(w1)

        w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(w2, mean=0.0, std=fwd_weight_scale/np.sqrt(nb_hidden))
        layers.append(w2)

        v1 = torch.empty((nb_hidden, nb_hidden), device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(v1, mean=0.0, std=rec_weight_scale/np.sqrt(nb_hidden))
        layers.append(v1)

    
    # Make predictions
    output, _, _ = run_snn(x,layers)
    m = torch.sum(others[-1],1) # sum over time
    _, am = torch.max(m, 1)     # argmax over output units
    
    return letters[am.detach().cpu().numpy()[0]]


# In[13]:


def train(params, dataset, lr=0.0015, nb_epochs=300, opt_parameters=None, layers=None, dataset_test=None):
    
    ttc_hist = []
    
    if (opt_parameters != None) & (layers != None):
        parameters = opt_parameters # The paramters we want to optimize
        layers = layers
    elif (opt_parameters != None) & (layers == None): 
        parameters = opt_parameters
        layers = [w1,w2,v1]
    elif (opt_parameters == None) & (layers != None):
        parameters = [w1,w2,v1]
        layers = layers
    elif (opt_parameters == None) & (layers == None): # default from tutorial 5
        parameters = [w1,w2,v1]
        layers = [w1,w2,v1]
        
    optimizer = torch.optim.Adamax(parameters, lr=0.0015, betas=(0.9,0.995)) # params['lr']

    log_softmax_fn = nn.LogSoftmax(dim=1) # The log softmax function across output units
    loss_fn = nn.NLLLoss() # The negative log likelihood loss function

    generator = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

    # The optimization loop
    loss_hist = []
    accs_hist = [[],[]]
    for e in range(nb_epochs):
        local_loss = []
        # accs: mean training accuracies for each batch
        accs = []
        for x_local, y_local in generator:
            x_local, y_local = x_local.to(device), y_local.to(device)
            output,recs,layers_update = run_snn(x_local,layers)
            _,spks,_=recs
            # with output spikes
            m = torch.sum(recs[-1],1) # sum over time
            log_p_y = log_softmax_fn(m)
        
            # Here we can set up our regularizer loss
            reg_loss = params['reg_spikes']*torch.mean(torch.sum(spks,1)) # e.g., L1 loss on total number of spikes (original: 1e-3)
            reg_loss += params['reg_neurons']*torch.mean(torch.sum(torch.sum(spks,dim=0),dim=0)**2) # L2 loss on spikes per neuron (original: 2e-6)
        
            # Here we combine supervised loss and the regularizer
            loss_val = loss_fn(log_p_y, y_local) + reg_loss

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            local_loss.append(loss_val.item())
        
            # compare to labels
            _, am = torch.max(m, 1) # argmax over output units
            tmp = np.mean((y_local == am).detach().cpu().numpy())
            accs.append(tmp)
    
        mean_loss = np.mean(local_loss)
        loss_hist.append(mean_loss)
        
        # mean_accs: mean training accuracy of current epoch (average over all batches)
        mean_accs = np.mean(accs)
        accs_hist[0].append(mean_accs)

        # Calculate test accuracy in each epoch
        if dataset_test is not None:
            test_acc, test_ttc = compute_classification_accuracy(
                params,
                dataset_test,
                layers=layers_update,
                early=True
            )
            accs_hist[1].append(test_acc) # only safe best test
            ttc_hist.append(test_ttc)
        
        if dataset_test is None:
            # save best training
            if mean_accs >= np.max(accs_hist[0]):
                best_acc_layers = []
                for ii in layers_update:
                    best_acc_layers.append(ii.detach().clone())
        else:
            # save best test
            if np.max(test_acc) >= np.max(accs_hist[1]):
                best_acc_layers = []
                for ii in layers_update:
                    best_acc_layers.append(ii.detach().clone())
        
        print("Epoch {}/{} done. Train accuracy: {:.2f}%, Test accuracy: {:.2f}%.".format(e+1,nb_epochs, accs_hist[0][-1]*100, accs_hist[1][-1]*100))
    

    return loss_hist, accs_hist, best_acc_layers, ttc_hist


# In[14]:


def build_and_train(params, ds_train, ds_test, epochs=epochs):
    
    global nb_input_copies
    nb_input_copies = params['nb_input_copies']  # Num of spiking neurons used to encode each channel 

    # Network parameters
    global nb_inputs
    nb_inputs  = nb_channels*nb_input_copies
    global nb_hidden
    nb_hidden  = 450
    global nb_outputs
    nb_outputs = len(np.unique(labels))+1
    global nb_steps
    nb_steps = data_steps

    tau_mem = params['tau_mem'] # ms
    tau_syn = tau_mem/params['tau_ratio']
    
    global alpha
    alpha   = float(np.exp(-time_step/tau_syn))
    global beta
    beta    = float(np.exp(-time_step/tau_mem))

    fwd_weight_scale = params['fwd_weight_scale']
    rec_weight_scale = params['weight_scale_factor']*fwd_weight_scale

    # Spiking network
    layers = []
    w1 = torch.empty((nb_inputs, nb_hidden),  device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.normal_(w1, mean=0.0, std=fwd_weight_scale/np.sqrt(nb_inputs))
    layers.append(w1)

    w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.normal_(w2, mean=0.0, std=fwd_weight_scale/np.sqrt(nb_hidden))
    layers.append(w2)
    
    v1 = torch.empty((nb_hidden, nb_hidden), device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.normal_(v1, mean=0.0, std=rec_weight_scale/np.sqrt(nb_hidden))
    layers.append(v1)

    layers_init = []
    for ii in layers:
        layers_init.append(ii.detach().clone())

    opt_parameters = [w1, w2, v1]
    
    # a fixed learning rate is already defined within the train function, that's why here it is omitted
    loss_hist, accs_hist, best_layers, _ = train(params, ds_train, nb_epochs=epochs, opt_parameters=opt_parameters, layers=layers, dataset_test=ds_test)

    # best training and test at best training
    acc_best_train = np.max(accs_hist[0]) # returns max value
    acc_best_train = acc_best_train*100
    idx_best_train = np.argmax(accs_hist[0]) # returns index of max value 
    acc_test_at_best_train = accs_hist[1][idx_best_train]*100

    # best test and training at best test
    acc_best_test = np.max(accs_hist[1])
    acc_best_test = acc_best_test*100
    idx_best_test = np.argmax(accs_hist[1])
    acc_train_at_best_test = accs_hist[0][idx_best_test]*100

    print("Final results: \n")
    print("Best training accuracy: {:.2f}% and according test accuracy: {:.2f}% at epoch: {}".format(acc_best_train, acc_test_at_best_train, idx_best_train+1)) # only from training
    print("Best test accuracy: {:.2f}% and according train accuracy: {:.2f}% at epoch: {}".format(acc_best_test, acc_train_at_best_test, idx_best_test+1)) # only from training

    return loss_hist, accs_hist, best_layers


# In[15]:


def compute_classification_accuracy(params, dataset, layers=None, early=False):
    """ Computes classification accuracy on supplied data in batches. """

    generator = DataLoader(dataset, batch_size=128,
                           shuffle=False, num_workers=2)
    accs = []
    multi_accs = []
    ttc = None

    for x_local, y_local in generator:
        x_local, y_local = x_local.to(device), y_local.to(device)
        if layers == None:
            layers = [w1,w2,v1]
            output, others, _= run_snn(x_local,layers)
        else:
            output, others, _ = run_snn(x_local,layers)
        # with output spikes
        m = torch.sum(others[-1],1) # sum over time
        _, am = torch.max(m, 1)     # argmax over output units
        # compare to labels
        tmp = np.mean((y_local == am).detach().cpu().numpy())
        accs.append(tmp)

        if early:
            accs_early = []
            for t in range(output.shape[1]-1):
                # with spiking output layer
                m_early = torch.sum(others[-1][:,:t+1,:],1) # sum over time
                _, am_early = torch.max(m_early, 1)         # argmax over output units
                # compare to labels
                tmp_early = np.mean((y_local == am_early).detach().cpu().numpy())
                accs_early.append(tmp_early)
            multi_accs.append(accs_early)
    
    if early:
        mean_multi = np.mean(multi_accs, axis=0)
        if np.max(mean_multi) > mean_multi[-1]:
            if mean_multi[-2]==mean_multi[-1]:
                flattening = []
                for ii in range(len(mean_multi)-2,1,-1):
                    if mean_multi[ii] != mean_multi[ii-1]:
                        flattening.append(ii)
                # time to classify
                ttc = time[flattening[0]]
            else:
                # time to classify
                ttc = time[-1]
        else:
            # time to classify
            ttc = time[np.argmax(mean_multi)]

    return np.mean(accs), ttc


# In[16]:


def ConfusionMatrix(params, dataset, save, layers=None, labels=letters):
    
    generator = DataLoader(dataset, batch_size=128,
                           shuffle=False, num_workers=2)
    accs = []
    multi_accs = []
    trues = []
    preds = []
    for x_local, y_local in generator:
        x_local, y_local = x_local.to(device), y_local.to(device)
        if layers == None:
            layers = [w1, w2, v1]
            output, others, _= run_snn(x_local, layers)
        else:
            output, others, _= run_snn(x_local, layers)
        # with output spikes
        m = torch.sum(others[-1],1) # sum over time
        _, am = torch.max(m, 1)     # argmax over output units
        # compare to labels
        tmp = np.mean((y_local == am).detach().cpu().numpy())
        accs.append(tmp)
        trues.extend(y_local.detach().cpu().numpy())
        preds.extend(am.detach().cpu().numpy())
    
    #return trues, preds
    
    cm = confusion_matrix(trues, preds, normalize='true')
    cm_df = pd.DataFrame(cm, index=[ii for ii in labels], columns=[jj for jj in labels])
    plt.figure(figsize=(12,9))
    sn.heatmap(cm_df,
               annot=True,
               fmt='.1g',
               #linewidths=0.005,
               #linecolor='black',
               cbar=False,
               square=False,
               cmap="YlGnBu")
    plt.xlabel('\nPredicted')
    plt.ylabel('True\n')
    plt.xticks(rotation=0)
    plt.show()


# Load augmented Braille data

# In[17]:


file_dir_data = './data/'
file_type = 'data_braille_letters_th'
file_thr = str(threshold)
file_name = file_dir_data + file_type + file_thr

file_dir_params = '../parameters/'
param_filename = 'parameters_th' + str(threshold)  + '.txt'
file_name_parameters = file_dir_params + param_filename
params = {}
with open(file_name_parameters) as file:
 for line in file:
    (key, value) = line.split()
    if key == 'time_bin_size' or key == 'nb_input_copies':
        params[key] = int(value)
    else:
        params[key] = np.double(value)


# In[18]:


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements 
    the surrogate gradient. By subclassing torch.autograd.Function, 
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid 
    as this was done in Zenke & Ganguli (2018).
    """

    scale = params['scale']

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad

spike_fn  = SurrGradSpike.apply


# ### Train and test the network

# In[19]:


if not use_nni_weights:
    ds_train, ds_test, labels, nb_channels, data_steps = load_and_extract_augmented(params, file_name, letter_written=letters)

    loss_hist, acc_hist, best_layers = build_and_train(params, ds_train, ds_test, epochs=epochs)


# In[20]:


if not use_nni_weights:
    plt.plot(range(1,len(acc_hist[0])+1),100*np.array(acc_hist[0]), color='blue')
    plt.plot(range(1,len(acc_hist[1])+1),100*np.array(acc_hist[1]), color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend(["Training","Test"], loc='lower right')
    plt.show()


# ### Test the pre-trained network if you already have the pre-trained weights

# In[21]:


if use_nni_weights:
    # saved from NNI:
    layers = load_layers("weights/SpyTorch_trained_weights_rec_th" + file_thr + run + ".pt", map_location=device)

    print("Input weights matrix: {}x{}".format(len(layers[0]),len(layers[0][0])))
    print("Hidden weights matrix: {}x{}".format(len(layers[2]),len(layers[2][0])))
    print("Output weights matrix: {}x{}".format(len(layers[1]),len(layers[1][0])))


# In[22]:


if use_nni_weights:
    ds_train, ds_test, labels, nb_channels, data_steps = load_and_extract_augmented(params, file_name, letter_written=letters)

    build(params)

    test_acc = compute_classification_accuracy(params, ds_test, layers=layers, early=True)

    print("Test accuracy: {}%".format(np.round(test_acc[0]*100,2)))
    print("Test accuracy as it comes, without rounding: {}".format(test_acc[0]))


# ### Confusion matrix

# In[24]:


save = False

if use_nni_weights:
    # from SAVED layers (from NNI) corresponding to best test:
    ConfusionMatrix(params, ds_test, layers=load_layers("weights/SpyTorch_trained_weights_rec_th" + file_thr + run + ".pt", map_location=device), save=save)
else:
    # from the JUST TRAINED layers corresponding to best test:
    ConfusionMatrix(params, ds_test, layers=best_layers, save=save)


# In[25]:


#torch.save(best_layers, "weights/SpyTorch_trained_weights_rec_th" + file_thr + run + ".pt")


# In[ ]:




