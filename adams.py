Fill in my code please. I want to plot the generalisation error

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP

# generate a data set

from sklearn.datasets import make_circles
X, y = make_circles(n_samples=100, noise=0.05)

y = y*2 - 1 # make y be -1 or 1 for outer and inner circle respectively.

# visualise the data in 2D
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='Spectral')

# initialise a neural network

def initialise(n_neurons):
    """Function that initialises a 2-layer neural network with n_neruons neurons in each layer."""
    
    model = MLP(2, [n_neurons, n_neurons, 1]) # 2-layer neural network
    print(model)
    print("number of parameters", len(model.parameters()))
    
    return model
    
model = initialise(n_neurons=16)

def loss(Xdat,ydat,model):
    """Returns the total loss (C(theta) + regularisation) and the accuracy of the neural network."""
    
    inputs = [list(map(Value, xrow)) for xrow in Xdat]
    scores = list(map(model, inputs))
    
    # compute data loss
    # svm "max-margin" loss - from micrograd demo example
    losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(ydat, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    
    alpha = 1e-4
    reg_loss = alpha * sum((p*p for p in model.parameters()))

    total_loss = data_loss + reg_loss #loss function + regularisation

    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(ydat, scores)]
    
    return total_loss, accuracy

def adam(model, n_steps=100):
    '''
    A function that optimises the parameters of the neural network in n_steps steps using the Adam optimiser.
    Return a list of losses and a list of accuracies during training. 
    '''
    
    losses = []
    accuracies = []
    
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lmbda = 1.0
    eta = 5e-2
    
    Nparam = len(model.parameters())
    
    m = np.zeros(Nparam)
    v = np.zeros(Nparam)
    
    for step in range(n_steps):
        
        t = step + 1
    
        # forward
        total_loss, acc = loss(X, y, model)
        losses.append(total_loss.data)
        accuracies.append(np.mean(acc))

        # backward
        model.zero_grad()
        total_loss.backward()

        # update (adam)
        #TO DO
    
    return losses, accuracies


modeladam = initialise(n_neurons = 16)
lossadam, accadam = adam(modeladam)

visualise(modeladam)