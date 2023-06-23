#############################################################################################################################
################## Functions for Data Pre-processing ######################
#############################################################################################################################

import numpy as np
import torch
import torch.nn.functional as F

#############################################################################################################################
#############################################################################################################################

# compute (normalized) node feature matrix N
def node_feature_matrix(grid, normalize):
    topographic_elevation = np.copy(grid.at_node['topographic__elevation'])
    
    N = torch.tensor([[x] for x in topographic_elevation])
    
    if normalize:
        return F.normalize(N, dim=0).float()
    else:
        return N.float()


#############################################################################################################################
#############################################################################################################################

# compute adjacency matrix A (COO format)
def adjacency_matrix(grid):
    tails1 = grid.node_at_link_tail[grid.active_links] 
    heads1 = grid.node_at_link_head[grid.active_links]
    
    tails2 = heads1
    heads2 = tails1
    
    edges1 = np.vstack((tails1, heads1))
    edges2 = np.vstack((tails2, heads2))
    
    A = torch.tensor(np.hstack((edges1, edges2)))
    return A


#############################################################################################################################
#############################################################################################################################

# compute (normalized) link feature matrix L
def link_feature_matrix(grid, normalize):
    lengths1 = grid.length_of_link[grid.active_links]
    lenghts2 = lengths1
    
    slopes1 = grid.calc_grad_at_link('topographic__elevation')[grid.active_links]
    slopes2 = slopes1 * -1
    
    lengths = np.hstack((lengths1, lenghts2))
    slopes = np.hstack((slopes1, slopes2))
    
    features = lengths * slopes

    L = torch.tensor([[f] for f in features])
    
    if normalize:
        return F.normalize(L, dim=0).float()
    else:
        return L.float() 


#############################################################################################################################
#############################################################################################################################

# compute (normalized) target matrix Y
def target_matrix(grid, type, reference, normalize):
    assert type in ['topo', 'topo_delta', 'drainage']

    if type == 'topo':
        topographic_elevation = np.copy(grid.at_node['topographic__elevation'])
        Y = torch.tensor([[t] for t in topographic_elevation])

    if type == 'topo_delta':
        topographic_elevation = np.copy(grid.at_node['topographic__elevation'])
        topographic_elevation_delta = topographic_elevation - reference
        Y = torch.tensor([[t] for t in topographic_elevation_delta])

    if type == 'drainage':
        drainage_area = np.copy(grid.at_node['drainage_area'])
        Y = torch.tensor([[d] for d in drainage_area])

    if normalize:
        return F.normalize(Y, dim=0).float()
    else:
        return Y.float()

#############################################################################################################################
#############################################################################################################################