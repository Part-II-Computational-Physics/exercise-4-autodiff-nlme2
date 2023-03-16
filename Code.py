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

def gd(model, n_steps=100):
    '''
    A function that optimises the parameters of the neural network in n_steps steps using gradient descent.
    Return a list of losses and a list of accuracies during training 
    '''
    
    losses = []
    accuracies = []

    for step in range(n_steps):

        # forward pass
        total_loss, acc = loss(X, y, model)
        losses.append(total_loss.data)
        accuracies.append(acc) #Computing the mean of the accuracy list to obtain overall accuracy
        


        # backward pass
        model.zero_grad()
        total_loss.backward()
    

        # update

        
        lr = learning_rate(step, n_steps)
              
        for p in model.parameters():
            p.data -= lr * p.grad

        if step % 10 == 0 or step == n_steps-1:
            print(f"step {step} loss {total_loss.data}, accuracy {np.mean(acc)*100}%")
    
    return losses, accuracies

def visualise(model):
    """Plots the descision boundary between the inner and outer circle."""
    
    h = 0.1
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    inputs = [list(map(Value, xrow)) for xrow in Xmesh]
    scores = list(map(model, inputs))
    Z = np.array([s.data > 0 for s in scores])
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    plt.figure(figsize=(5,5))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.axis('equal')    

    visualise(modelgd)

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
        accuracies.append(acc)

        # backward
        model.zero_grad()
        total_loss.backward()

        # update (adam)
        lr = eta * learning_rate(step, n_steps)
        
        for i, param in enumerate(model.parameters()):
            grad = param.grad
            m[i] = b1*m[i] + (1-b1)*grad
            v[i] = b2*v[i] + (1-b2)*grad*grad

            # Bias correction
            mb = m[i] / (1-b1**t)
            vb = v[i] / (1-b2**t)

            # Update parameters
            param.data -= lr * mb / (np.sqrt(vb) + eps) - lr * lmbda * param.data

        if step % 10 == 0 or step == n_steps-1:
            print(f"step {step} loss {total_loss.data}, accuracy {np.mean(acc)*100}%")
    
    return losses, accuracies

modeladam = initialise(n_neurons = 16)
lossadam, accadam = adam(modeladam)

visualise(modeladam)

def lion(model, n_steps=100):
    '''
    A function that optimizes the parameters of the neural network in n_steps steps using the Lion optimizer.
    Returns a list of losses and a list of accuracies during training. 
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
        accuracies.append(acc)

        # backward
        model.zero_grad()
        total_loss.backward()

        # update (lion)
        lr = eta * learning_rate(step, n_steps)

        for i, param in enumerate(model.parameters()):
            grad = param.grad
            m[i] = b1*m[i] + (1-b1)*grad
            v[i] = b2*v[i] + (1-b2)*grad*grad

            # EMA of gt
            mt = b2*m[i] + (1-b2)*grad
            ct = b1*m[i] + (1-b1)*grad
            gt = ct / (1-b1**t)
            mt_hat = mt / (1-b2**t)

            # update
            param.data -= lr * (np.sign(gt) + lmbda*param.data)

        if step % 10 == 0 or step == n_steps-1:
            print(f"step {step} loss {total_loss.data}, accuracy {np.mean(acc)*100}%")
    
    return losses, accuracies

### PLOT ACCURACY ###

plt.figure()
plt.plot(accgd,   label = 'gd')
plt.plot(accadam, label = 'Adam')
plt.plot(acclion, label = 'Lion')
plt.plot([0,len(accgd)-1],[1,1], '--', c='k')
plt.legend()
plt.xlabel('step')
plt.ylabel('accuracy')
plt.xlim([0,len(accgd)-1])
plt.ylim([0,1.01])

### PLOT LOSS ###

plt.figure()
plt.plot(lossgd,   label = 'gd')
plt.plot(lossadam, label = 'Adam')
plt.plot(losslion, label = 'Lion')
plt.legend()
plt.xlabel('step')
plt.ylabel('loss')
plt.xlim([0,len(lossgd)-1])

# Test set
phi = np.linspace(0, 2*np.pi, 40)
r   = np.linspace(0.7, 1.1,   10)

Xtest = np.zeros((len(phi)*len(r), 2))
i=0
for r0 in r:
    for phi0 in phi:
        Xtest[i,:] = np.array([r0*np.cos(phi0),r0*np.sin(phi0)])
        i+=1
ytest = np.hstack([(1*(radius<0.9)-1*(radius>0.9))*np.ones(len(phi), dtype=np.intp) for radius in r])

plt.figure(figsize=(5,5))
plt.scatter(Xtest[:,0], Xtest[:,1], c=ytest, s=20, cmap='Spectral')
plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='Spectral', alpha=0.5)

### GENERALISATION ERROR ###
# TO DO