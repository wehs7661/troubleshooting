import os 
import sys
import copy
import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils import data
from torch import distributions
from generator import RealNVP

class BoltzmannGenerator:
    def __init__(self, model_params=None):
        """
        Initialize a Boltzmann generator. Note that this method assumes identical number 
        of nodes for all layers in the network (except for the input/output nodes). The
        activation functions are all ReLU function in all layers, except that the the last
        activation function of the scaling network is a hyperbolic tangent function. 

        Parameters
        ----------
        model_params : dict
            A dictionary of the model parameters, including the n_blocks, dimension, 
            n_nodes, n_layers, n_iteration, batch_size, LR and prior_sigma
        """
        self.params = model_params    
        for key in self.params:
            setattr(self, key, self.params[key])

    def affine_layers(self):
        layers = []
        for i in range(self.n_layers):
            if i == 0:  # first layer
                layers.append(nn.Linear(self.dimension, self.n_nodes))
            elif i == self.n_layers - 1:  # last layer
                layers.append(nn.Linear(self.n_nodes, self.dimension))
            else:  # hidden layers
                layers.append(nn.Linear(self.n_nodes, self.n_nodes))
            
            if i != self.n_layers - 1:
                layers.append(nn.ReLU())
        
        return layers

    def build_networks(self):
        self.s_net = lambda: nn.Sequential(*self.affine_layers(), nn.Tanh())
        self.t_net = lambda: nn.Sequential(*self.affine_layers())
    
    def build(self, system):
        """
        Parameters
        ----------
        system : object
            The object of the system of interest. (For example, DoubleWellPotential)
        """
        self.build_networks()   # build the affine coupling layers
        self.mask = torch.from_numpy(np.array([[0, 1], [1, 0]] * self.n_blocks).astype(np.float32))
        self.prior = distributions.MultivariateNormal(torch.zeros(self.dimension), torch.eye(self.dimension) * self.prior_sigma) 
        model = RealNVP(self.s_net, self.t_net, self.mask, self.prior, system, (self.dimension,))
        return model

    def preprocess_data(self, samples):
        self.n_pts = len(samples)   # number of data points
        training_set = samples.astype('float32')
        subdata = data.DataLoader(dataset=training_set, batch_size=self.batch_size)
        batch = torch.from_numpy(subdata.dataset)   # note that subdata.dataset is a numpy array

        return batch


    def train(self, model, x_samples=None, optimizer=None):
        """
        Trains a Boltzmann generator.

        Parameters
        ----------
        model : objet
            The object of the model to be trained that is built by Boltzmann.build.
        w_loss : list or np.array
            The weighting coefficients of the loss functions. w_loss = [w_ML, w_KL, w_RC]
        samples : np.array
            The training data set for training the model. 
        optimizer : object
            The object of the optimizer for gradient descent method.
        """
        if optimizer is None:
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad==True], lr=self.LR)
        
        # preprocess the tradining datasets

        batch_x = self.preprocess_data(x_samples)

        # start training!
        self.loss_list = []
        for i in tqdm(range(self.n_iterations)):
            loss = model.loss_ML(batch_x)
            self.loss_list.append(loss.item())  # convert from 1-element tensor to scalar

            # backpropagation
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()   # check https://tinyurl.com/y8o2y5e7 for more info
            print("Total loss: %s" % loss.item(), end='\r')

        return model
        