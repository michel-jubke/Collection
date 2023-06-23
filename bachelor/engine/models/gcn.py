#############################################################################################################################
############################################### Pytorch Geometric GCN Models ####################################################
#############################################################################################################################

from torch.nn import Module, Sequential
from torch_geometric.nn import Linear, GCNConv
import torch.nn.functional as F

#############################################################################################################################

class GCN8(Module):
    
    def __init__(self, in_channels, out_channels, use_edge_weights):
        super().__init__()

        self.name = 'GCN8'
        self.out = out_channels
        self.use_edge_weights = use_edge_weights

        self.convs = Sequential(
            GCNConv(in_channels, 256),
            GCNConv(256, 256),
            
            GCNConv(256, 128),
            GCNConv(128, 128),
            
            GCNConv(128, 64),
            GCNConv(64, 64),
            
            GCNConv(64, 32),
            GCNConv(32, 32))
        
        self.regression = Linear(32, out_channels)
        
    
    def forward(self, data):

        x, edge_index = data.x, data.edge_index 
        
        if self.use_edge_weights: 
            edge_weights = data.edge_attr
        else:
            edge_weights = None

        for conv in self.convs:
            x = conv(x, edge_index, edge_weights)
            x = F.relu(x)

        return self.regression(x)


#############################################################################################################################

class GCN16(Module):
    
    def __init__(self, in_channels, out_channels, use_edge_weights):
        super().__init__()

        self.name = 'GCN16'
        self.out = out_channels
        self.use_edge_weights = use_edge_weights

        self.convs = Sequential(
            GCNConv(in_channels, 256),
            GCNConv(256, 256),
            GCNConv(256, 256),
            
            GCNConv(256, 128),
            GCNConv(128, 128),
            GCNConv(128, 128),
            
            GCNConv(128, 64),
            GCNConv(64, 64),
            GCNConv(64, 64),
            
            GCNConv(64, 32),
            GCNConv(32, 32),
            GCNConv(32, 32),
            
            GCNConv(32, 16),
            GCNConv(16, 16),
            GCNConv(16, 16),

            GCNConv(16, 8))
        
        self.regression = Linear(8, out_channels)
        
    
    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index
        
        if self.use_edge_weights: 
            edge_weights = data.edge_attr
        else:
            edge_weights = None

        for conv in self.convs:
            x = conv(x, edge_index, edge_weights)
            x = F.relu(x)

        return self.regression(x)


#############################################################################################################################

class GCN32(Module):
    
    def __init__(self, in_channels, out_channels, use_edge_weights):
        super().__init__()

        self.name = 'GCN32'
        self.out = out_channels
        self.use_edge_weights = use_edge_weights

        self.convs  = Sequential(
            GCNConv(in_channels, 256),
            GCNConv(256, 256),
            GCNConv(256, 256),
            GCNConv(256, 256),
            GCNConv(256, 256),

            GCNConv(256, 128),
            GCNConv(128, 128),
            GCNConv(128, 128),
            GCNConv(128, 128),
            GCNConv(128, 128),
            
            GCNConv(128, 64),
            GCNConv(64, 64),
            GCNConv(64, 64),
            GCNConv(64, 64),
            GCNConv(64, 64),
            
            GCNConv(64, 32),
            GCNConv(32, 32),
            GCNConv(32, 32),
            GCNConv(32, 32),
            GCNConv(32, 32),
            
            GCNConv(32, 16),
            GCNConv(16, 16),
            GCNConv(16, 16),
            GCNConv(16, 16),
            GCNConv(16, 16),
            
            GCNConv(16, 8),
            GCNConv(8, 8),
            GCNConv(8, 8),
            GCNConv(8, 8),
            GCNConv(8, 8),
            
            GCNConv(8, 4),
            GCNConv(4, 4))
        
        self.regression = Linear(4, out_channels)
        
    
    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index 
        
        if self.use_edge_weights: 
            edge_weights = data.edge_attr
        else:
            edge_weights = None

        for conv in self.convs:
            x = conv(x, edge_index, edge_weights)
            x = F.relu(x)
        
        return self.regression(x)

