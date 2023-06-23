#############################################################################################################################
############################################# Pytorch Geometric SAGE Models #################################################
#############################################################################################################################

from torch.nn import Module, Sequential
from torch_geometric.nn import Linear, SAGEConv
import torch.nn.functional as F

#############################################################################################################################

class SAGE8(Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.name = 'SAGE8'
        self.out = out_channels
        
        self.convs = Sequential(
            SAGEConv(in_channels, 256),
            SAGEConv(256, 256),
            
            SAGEConv(256, 128),
            SAGEConv(128, 128),
            
            SAGEConv(128, 64),
            SAGEConv(64, 64),
            
            SAGEConv(64, 32),
            SAGEConv(32, 32))
        
        self.regression = Linear(32, out_channels)
        
    
    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index 

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        return self.regression(x)


#############################################################################################################################

class SAGE16(Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.name = 'SAGE16'
        self.out = out_channels
        
        self.convs = Sequential(
            SAGEConv(in_channels, 256),
            SAGEConv(256, 256),
            SAGEConv(256, 256),
            
            SAGEConv(256, 128),
            SAGEConv(128, 128),
            SAGEConv(128, 128),
            
            SAGEConv(128, 64),
            SAGEConv(64, 64),
            SAGEConv(64, 64),
            
            SAGEConv(64, 32),
            SAGEConv(32, 32),
            SAGEConv(32, 32),
            
            SAGEConv(32, 16),
            SAGEConv(16, 16),
            SAGEConv(16, 16),
            
            SAGEConv(16, 8))
        
        self.regression = Linear(8, out_channels)
        
    
    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index 

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        return self.regression(x)


#############################################################################################################################

class SAGE32(Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.name = 'SAGE32'
        self.out = out_channels
        
        self.convs = Sequential(
            SAGEConv(in_channels, 256),
            SAGEConv(256, 256),
            SAGEConv(256, 256),
            SAGEConv(256, 256),
            SAGEConv(256, 256),
            
            SAGEConv(256, 128),
            SAGEConv(128, 128),
            SAGEConv(128, 128),
            SAGEConv(128, 128),
            SAGEConv(128, 128),
            
            SAGEConv(128, 64),
            SAGEConv(64, 64),
            SAGEConv(64, 64),
            SAGEConv(64, 64),
            SAGEConv(64, 64),
            
            SAGEConv(64, 32),
            SAGEConv(32, 32),
            SAGEConv(32, 32),
            SAGEConv(32, 32),
            SAGEConv(32, 32),
            
            SAGEConv(32, 16),
            SAGEConv(16, 16),
            SAGEConv(16, 16),
            SAGEConv(16, 16),
            SAGEConv(16, 16),
            
            SAGEConv(16, 8),
            SAGEConv(8, 8),
            SAGEConv(8, 8),
            SAGEConv(8, 8),
            SAGEConv(8, 8),

            SAGEConv(8, 4),
            SAGEConv(4, 4))
        
        self.regression = Linear(4, out_channels)
        
    
    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index 

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        return self.regression(x)


#############################################################################################################################

class SAGE48(Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.name = 'SAGE48'
        self.out = out_channels
        
        self.convs = Sequential(
            SAGEConv(in_channels, 256),
            SAGEConv(256, 256),
            SAGEConv(256, 256),
            SAGEConv(256, 256),
            SAGEConv(256, 256),
            SAGEConv(256, 256),
            SAGEConv(256, 256),

            SAGEConv(256, 128),
            SAGEConv(128, 128),
            SAGEConv(128, 128),
            SAGEConv(128, 128),
            SAGEConv(128, 128),
            SAGEConv(128, 128),
            SAGEConv(128, 128),
            
            SAGEConv(128, 64),
            SAGEConv(64, 64),
            SAGEConv(64, 64),
            SAGEConv(64, 64),
            SAGEConv(64, 64),
            SAGEConv(64, 64),
            SAGEConv(64, 64),
            
            SAGEConv(64, 32),
            SAGEConv(32, 32),
            SAGEConv(32, 32),
            SAGEConv(32, 32),
            SAGEConv(32, 32),
            SAGEConv(32, 32),
            SAGEConv(32, 32),
            
            SAGEConv(32, 16),
            SAGEConv(16, 16),
            SAGEConv(16, 16),
            SAGEConv(16, 16),
            SAGEConv(16, 16),
            SAGEConv(16, 16),
            SAGEConv(16, 16),
            
            SAGEConv(16, 8),
            SAGEConv(8, 8),
            SAGEConv(8, 8),
            SAGEConv(8, 8),
            SAGEConv(8, 8),
            SAGEConv(8, 8),
            SAGEConv(8, 8),

            SAGEConv(8, 4),
            SAGEConv(4, 4),
            SAGEConv(4, 4),
            SAGEConv(4, 4),
            SAGEConv(4, 4),
            SAGEConv(4, 4))
        
        self.regression = Linear(4, out_channels)
        
    
    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index 

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        return self.regression(x)


#############################################################################################################################

class SAGE80(Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.name = 'SAGE80'
        self.out = out_channels
        
        self.convs = Sequential(
            SAGEConv(in_channels, 256),
            SAGEConv(256, 256),
            SAGEConv(256, 256),
            SAGEConv(256, 256),
            SAGEConv(256, 256),
            SAGEConv(256, 256),
            SAGEConv(256, 256),
            SAGEConv(256, 256),
            SAGEConv(256, 256),
            SAGEConv(256, 256),
            SAGEConv(256, 256),
            SAGEConv(256, 256),

            SAGEConv(256, 128),
            SAGEConv(128, 128),
            SAGEConv(128, 128),
            SAGEConv(128, 128),
            SAGEConv(128, 128),
            SAGEConv(128, 128),
            SAGEConv(128, 128),
            SAGEConv(128, 128),
            SAGEConv(128, 128),
            SAGEConv(128, 128),
            SAGEConv(128, 128),
            SAGEConv(128, 128),
            
            SAGEConv(128, 64),
            SAGEConv(64, 64),
            SAGEConv(64, 64),
            SAGEConv(64, 64),
            SAGEConv(64, 64),
            SAGEConv(64, 64),
            SAGEConv(64, 64),
            SAGEConv(64, 64),
            SAGEConv(64, 64),
            SAGEConv(64, 64),
            SAGEConv(64, 64),
            SAGEConv(64, 64),
            
            SAGEConv(64, 32),
            SAGEConv(32, 32),
            SAGEConv(32, 32),
            SAGEConv(32, 32),
            SAGEConv(32, 32),
            SAGEConv(32, 32),
            SAGEConv(32, 32),
            SAGEConv(32, 32),
            SAGEConv(32, 32),
            SAGEConv(32, 32),
            SAGEConv(32, 32),
            SAGEConv(32, 32),
            
            SAGEConv(32, 16),
            SAGEConv(16, 16),
            SAGEConv(16, 16),
            SAGEConv(16, 16),
            SAGEConv(16, 16),
            SAGEConv(16, 16),
            SAGEConv(16, 16),
            SAGEConv(16, 16),
            SAGEConv(16, 16),
            SAGEConv(16, 16),
            SAGEConv(16, 16),
            SAGEConv(16, 16),
            
            SAGEConv(16, 8),
            SAGEConv(8, 8),
            SAGEConv(8, 8),
            SAGEConv(8, 8),
            SAGEConv(8, 8),
            SAGEConv(8, 8),
            SAGEConv(8, 8),
            SAGEConv(8, 8),
            SAGEConv(8, 8),
            SAGEConv(8, 8),
            SAGEConv(8, 8),
            SAGEConv(8, 8),

            SAGEConv(8, 4),
            SAGEConv(4, 4),
            SAGEConv(4, 4),
            SAGEConv(4, 4),
            SAGEConv(4, 4),
            SAGEConv(4, 4),
            SAGEConv(4, 4),
            SAGEConv(4, 4))
        
        self.regression = Linear(4, out_channels)
        
    
    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index 

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        return self.regression(x)
