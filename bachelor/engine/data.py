#############################################################################################################################
################################################ Classes for Data Generation ################################################
#############################################################################################################################

from landlab import RasterModelGrid, VoronoiDelaunayGrid
from landlab.components import FlowDirectorSteepest, FlowAccumulator, StreamPowerEroder
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import engine.pre_processing as pp

import engine.plot as pl

import numpy as np
np.random.seed(1312)

#############################################################################################################################
#############################################################################################################################

class RasterGrid():
    '''
    This class allows us to easily generate and check landlab raster grids that can be passed to LandlabData.add() later on.
    '''
    
    def __init__(self, rows, columns, spacing, topo, uplift, close_r=True, close_t=True, close_l=True, close_b=False):        

        # grid init
        self.grid = RasterModelGrid((rows, columns), spacing)
        
        # boundary control
        right = close_r
        top = close_t
        left = close_l
        bottom = close_b
        self.grid.set_closed_boundaries_at_grid_edges(right, top, left, bottom)
        
        # topography init
        topo = np.copy(topo)
        topo += uplift
        topo[self.grid.boundary_nodes] = 0
        self.grid.add_field('topographic__elevation', topo, at='node', units='m', clobber=True)

    
    def plot_init_2D(self):
        pl.plot_topo_raster_2D(self.grid)

    
    def plot_init_3D(self):
        pl.plot_topo_raster_3D(self.grid)


    def plot_erosion_2D(self, timesteps, dt):
        pl.plot_erosion_raster_2D(self.grid, timesteps, dt)
        

    def plot_erosion_3D(self, timesteps, dt):
        pl.plot_erosion_raster_3D(self.grid, timesteps, dt)

#############################################################################################################################
#############################################################################################################################

class VoronoiGrid():
    '''
    This class allows us to easily generate and check landlab voronoi grids that can be passed to LandlabData.add() later on.
    '''
    
    def __init__(self, xs, ys, topo, uplift):        

        # grid init
        self.grid = VoronoiDelaunayGrid(xs, ys)
        
        # boundary control
        pass
        
        # topography init
        topo = np.copy(topo)
        topo += uplift
        topo[self.grid.boundary_nodes] = 0
        self.grid.add_field('topographic__elevation', topo, at='node', units='m', clobber=True)

    
    def plot_init_2D(self):
        pl.plot_topo_voronoi_2D(self.grid)

    
    def plot_init_3D(self):
        pl.plot_topo_voronoi_3D(self.grid)


    def plot_erosion_2D(self, timesteps, dt):
        pl.plot_erosion_voronoi_2D(self.grid, timesteps, dt)
        

    def plot_erosion_3D(self, timesteps, dt):
        pl.plot_erosion_voronoi_3D(self.grid, timesteps, dt)

#############################################################################################################################
#############################################################################################################################

class LandlabData():
    '''
    This class allows us, to generate as much data as we want and to export it as a ready to use 
    pytorch_geometric dataloader. All this, with a very easy to use interface:

        g1 = RasterGrid(.....)
        g2 = RasterGrid(.....)

        data = LandlabData()

        data.add(g1)
        data.add(g2)
        ...

        loader = data.make_dataloader(batch_size, mask)

    Happy sampling!    
    '''

    def __init__(self):    
        self.graphs = dict()
        self.offset = 0


    def __len__(self):
        return len(self.graphs)


    def add(self, grid, timesteps, dt, target_type, nf_norm=False, lf_norm=True, t_norm=False):
        '''
        This function takes as input a `landlab` grid with an initial topography and the timesteps for which to simulate erosion.
        It then generates and saves one graph per timestep that holds all the information we need (target included) to the 
        global container. 
        '''
        
        assert target_type in ['topo', 'topo_delta', 'drainage'], "Choose valid target option"

        # save initial topography
        top_init = np.copy(grid.grid.at_node['topographic__elevation'])

        # init landlab components
        fd = FlowDirectorSteepest(grid.grid)
        fa = FlowAccumulator(grid.grid, flow_director=fd)
        se = StreamPowerEroder(grid.grid)

        # and now: run!
        for _ in range(timesteps):

            N = pp.node_feature_matrix(grid.grid, nf_norm)
            A = pp.adjacency_matrix(grid.grid)
            L = pp.link_feature_matrix(grid.grid, lf_norm)
            
            if target_type == 'topo_delta':
                reference = np.copy(grid.grid.at_node['topographic__elevation']) 
            else:
                reference = None

            fa.run_one_step()
            se.run_one_step(dt) 

            Y = pp.target_matrix(grid.grid, target_type, reference, t_norm)
            
            self.graphs[self.offset] = Data(N, A, L, Y)
            self.offset += 1

        # restore initial topography
        grid.grid.at_node['topographic__elevation'] = top_init
            

    def make_dataloader(self, batch_size):
        ''' 
        This function returns a ready to use `pytorch_geometric` DataLoader instance 
        from all the graphs added so far.
        '''
        
        loader = DataLoader(self.graphs, batch_size=batch_size, shuffle=True)
        return loader

#############################################################################################################################
#############################################################################################################################

