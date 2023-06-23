#############################################################################################################################
################################################ Functions for Model Training ###############################################
#############################################################################################################################

import torch
import numpy as np
from engine.plot import plot_loss
import time
from datetime import timedelta
from copy import deepcopy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#############################################################################################################################
#############################################################################################################################

def train(model, loader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    for batch in loader:
        batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        epoch_loss += loss
        loss.backward()
        optimizer.step()
    return epoch_loss


#############################################################################################################################
#############################################################################################################################

def test(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    for batch in loader:
        batch.to(device)
        with torch.no_grad():
            out = model(batch)
        loss = criterion(out, batch.y)
        epoch_loss += loss
    return epoch_loss


#############################################################################################################################
#############################################################################################################################

def train_and_test(model, epochs, train_loader, test_loader, criterion, optimizer, plot=False):
    model.to(device)
    
    print('Start training...')
    print('-------------------')
    
    train_losses = []
    test_losses = []

    best_test_loss = np.inf
    best_model = None

    total_time = 0
    
    for epoch in range(epochs):
        start = time.time()

        train_loss = train(model, train_loader, criterion, optimizer)
        test_loss = test(model, test_loader, criterion)
    
        stop = time.time()

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model = deepcopy(model)

        epoch_time = stop - start
        total_time += epoch_time

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        tt = timedelta(seconds=round(total_time))    
        if epoch == 0: 
            print(f"Epoch {epoch+1:>4} finished | Train loss: {train_loss:>8.5f} | Test loss: {test_loss:>8.5f} | Epoch time: {epoch_time:>5.2f}s | Total time: {tt}")
        if epoch % 25 == 24: 
            print(f"Epoch {epoch+1:>4} finished | Train loss: {train_loss:>8.5f} | Test loss: {test_loss:>8.5f} | Epoch time: {epoch_time:>5.2f}s | Total time: {tt}")
    
    print('-------------------')
    print(f'Finished training, needed {tt}')
    print()
    
    train_losses = [loss.detach() for loss in train_losses]
    test_losses = [loss.detach() for loss in test_losses]
    
    if plot: 
        plot_loss(train_losses, test_losses)
    else:
        return best_model, train_losses, test_losses    

#############################################################################################################################
#############################################################################################################################

