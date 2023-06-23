#############################################################################################################################
############################################### Functions for Model Testing #################################################
#############################################################################################################################

import numpy as np
import torch
from torch_geometric.data import Data
from torch.nn.functional import smooth_l1_loss
from landlab.components import FlowDirectorSteepest, FlowAccumulator, StreamPowerEroder
import engine.pre_processing as pp


#############################################################################################################################
#############################################################################################################################

def simulate_erosion(grid, timesteps, method, model=None, dt=None):
    '''
    This function lets us simulate erosion on a landlab grid using either the native landlab components or a (trained) gcn
    '''
    
    assert method in ['landlab', 'gcn_delta', 'gcn']

    # save initial topography
    top_init = np.copy(grid.at_node['topographic__elevation'])

    grid.at_node['topographic__elevation'][grid.boundary_nodes] = 0
    
    predictions = [top_init]
    
    # landlab simulation
    if method == 'landlab':
        fd = FlowDirectorSteepest(grid)
        fa = FlowAccumulator(grid, flow_director=fd)
        se = StreamPowerEroder(grid)

        for _ in range(timesteps):
            fd.run_one_step()
            fa.run_one_step()
            se.run_one_step(dt)
            
            pred = np.copy(grid.at_node['topographic__elevation'])
            predictions.append(pred)
         
    # gcn_delta simulation        
    if method == 'gcn_delta':
        assert model is not None, "Need a trained model"
        model.to('cpu')
        for _ in range(timesteps):
            N = pp.node_feature_matrix(grid, normalize=False)
            A = pp.adjacency_matrix(grid)
            L = pp.link_feature_matrix(grid, normalize=True)
            data = Data(x=N, edge_index=A, edge_attr=L)
            delta = model(data).squeeze()
            delta = delta.detach().numpy()
            topography = grid.at_node['topographic__elevation']
            topography += delta
            topography[topography < 0] = 0
            prediction = np.copy(topography)
            predictions.append(prediction)
    
    # gcn simulation
    if method == 'gcn':
        assert model is not None, "Need a trained model"
        model.to('cpu')
        for _ in range(timesteps):
            N = pp.node_feature_matrix(grid, normalize=False)
            A = pp.adjacency_matrix(grid)
            L = pp.link_feature_matrix(grid, normalize=True)
            data = Data(x=N, edge_index=A, edge_attr=L)
            next_topo = model(data).squeeze()
            next_topo = next_topo.detach().numpy()
            topography = grid.at_node['topographic__elevation']
            topography = next_topo
            topography[topography < 0] = 0
            prediction = np.copy(topography)
            predictions.append(prediction)
    

    # restore initial topography
    grid.at_node['topographic__elevation'] = top_init
        
    return np.copy(predictions)


#############################################################################################################################
#############################################################################################################################

def simulate_drainage(grid, timesteps, method, model, dt):
    '''
    This function lets us simulate erosion on a landlab grid for a given number of timesteps. For each timestep we compute the drainage
    area with both the native landlab modules and a trained gcn — both are returned. 
    '''
    
    assert method in ['landlab', 'gcn']

    # save initial topography
    top_init = np.copy(grid.at_node['topographic__elevation'])

    # landlab components    
    fd = FlowDirectorSteepest(grid)
    fa = FlowAccumulator(grid, flow_director=fd)
    se = StreamPowerEroder(grid)
    
    # simulate for t timesteps
    if method == 'gcn':
        assert model is not None, "Need a trained model"
        model.to('cpu')
        drainages_model = []

        for _ in range(timesteps):
            N = pp.node_feature_matrix(grid, normalize=False)
            A = pp.adjacency_matrix(grid)
            L = pp.link_feature_matrix(grid, normalize=True)
            data = Data(x=N, edge_index=A, edge_attr=L)

            drainage_model = model(data).squeeze().detach().numpy()
            drainages_model.append(drainage_model)

            fd.run_one_step()
            fa.run_one_step()
            se.run_one_step(dt)

    if method == 'landlab':
        for _ in range(timesteps):
            drainages_landlab = []
            
            drainage_landlab = np.copy(grid.at_node['drainage_area'])
            drainages_landlab.append(drainage_landlab)

            fd.run_one_step()
            fa.run_one_step()
            se.run_one_step(dt)
    
    # restore initial topography
    grid.at_node['topographic__elevation'] = top_init

    if method == 'landlab':    
        return np.copy(drainages_landlab)
    if method == 'gcn':
        return np.copy(drainages_model)


#############################################################################################################################
#############################################################################################################################

def simulation_loss(simulation_1, simulation_2):
    '''
    This function takes as input arguments two simulations as returnd by function simulate_{erosion, drainage} and returns
    the smooth l1 loss between the two simulations averaged over all the number of frames per simulation.
    '''
    
    assert len(simulation_1) == len(simulation_2), "Simulations need same number of datapints"
    total_loss = 0
    s1 = torch.tensor(simulation_1)
    s2 = torch.tensor(simulation_2)
    for i in range(len(simulation_1)):
        loss = smooth_l1_loss(s1[i], s2[i])
        total_loss += loss
    average_loss = total_loss/len(simulation_1)
    return average_loss


#############################################################################################################################
#############################################################################################################################



