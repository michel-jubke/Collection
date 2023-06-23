#############################################################################################################################
############################################### Functions for Model Testing #################################################
#############################################################################################################################

import numpy as np

from landlab.plot.imshow import imshow_grid

from engine.simulate import simulate_erosion, simulation_loss

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
from matplotlib import cm
import ipywidgets as widgets


#############################################################################################################################
#############################################################################################################################

def plot_topo_raster_2D(grid):
    '''
    This function takes as argument a landlab raster grid and plots its topography in 2D
    '''

    rows = grid.number_of_node_rows
    columns = grid.number_of_node_columns
    topo = grid.at_node['topographic__elevation']
    topo = topo.reshape((rows, columns))
    topo = np.flip(topo, axis=0)
    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(topo, cmap=cm.copper)
    fig.colorbar(im, aspect=10, fraction=.08)
    plt.show()


#############################################################################################################################
#############################################################################################################################

def plot_topo_voronoi_2D(grid):
    '''
    This function takes as argument a landlab voronoi raster grid and plots its topography in 2D
    '''

    imshow_grid(grid, 'topographic__elevation')
    plt.show()


#############################################################################################################################
#############################################################################################################################

def plot_topo_raster_3D(grid):
    '''
    This function takes as argument a landlab grid and plots its topography in 3D
    '''
    
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')

    xs = grid.x_of_node.reshape(grid.shape)
    ys = grid.y_of_node.reshape(grid.shape)
    zs = grid.at_node['topographic__elevation'].reshape(grid.shape)

    surf1 = ax.plot_surface(xs, ys, zs, cmap=cm.copper)
    ax.set_axis_off()
    fig.colorbar(surf1, ax=ax, aspect=8, fraction=.08)
    plt.show()


#############################################################################################################################
#############################################################################################################################

def plot_topo_voronoi_3D(grid):
    raise NotImplementedError


#############################################################################################################################
#############################################################################################################################

def plot_erosion_raster_2D(grid, timesteps, dt, interactive):
    if interactive:
        raise NotImplementedError
    else:
        raise NotImplementedError


#############################################################################################################################
#############################################################################################################################

def plot_erosion_voronoi_2D(grid, timesteps, dt):
    raise NotImplementedError


#############################################################################################################################
#############################################################################################################################

def plot_erosion_raster_3D(grid, timesteps, dt, interactive=True):

    rows = grid.number_of_node_rows
    columns = grid.number_of_node_columns

    # plot landscape evolution
    x_vals = grid.x_of_node.reshape((rows, columns))
    y_vals = grid.y_of_node.reshape((rows, columns))
    erosion = simulate_erosion(grid, timesteps, method='landlab', dt=dt)

    if interactive:
        # define surface plot
        def plot_func(t):
            fig = plt.figure(figsize=(7,15))
            ax = plt.axes(projection='3d')
            ax.set_axis_off()
            surf = ax.plot_surface(x_vals, y_vals, erosion[t].reshape((rows, columns)), cmap=cm.copper)
            fig.colorbar(surf, aspect=8, fraction=.08)
            plt.show()

        # make it interactive
        widgets.interact(plot_func, t=widgets.IntSlider(value=0, 
                                                        min=0, 
                                                        max=timesteps, 
                                                        step=1,
                                                        description='Timesteps:'))

    
    else:
        raise NotImplementedError

#############################################################################################################################
#############################################################################################################################

def plot_erosion_voronoi_3D(grid, timesteps, dt):
    raise NotImplementedError


#############################################################################################################################
#############################################################################################################################

def compare_erosion_raster_2D(erosion1, erosion2, gnn_name, choice=[0, 14, 29, 49], interactive=False, square=True, rows=None, columns=None):
    '''
    This function takes two erosion simulations as returned by simulate.simulate_erosion() and plots a 
    2D topographic comparison either interactively or statically at given timepoints
    '''
    from landlab import RasterModelGrid

    assert len(erosion1) == len(erosion2)
    
    # define dimensions
    if square:
        dim = int(np.sqrt(len(erosion1[0])))
        rows, columns = dim, dim
    else:
        assert rows is not None, "Need number of rows if not square"
        assert columns is not None, "Need number of columns if not square"

    # interactive plot
    if interactive:

        def plot_func(t):
            # define layout
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,8))

            # plot 1
            ll = np.flip(erosion1[t].reshape((rows, columns)), axis=0)
            im1 = ax1.imshow(ll, cmap=cm.copper)
            ax1.set_axis_off()
            fig.colorbar(im1, ax=ax1, aspect=10, fraction=.08)
            ax1.set_title('Landlab')

            # plot 2
            gcn = np.flip(erosion2[t].reshape((rows, columns)), axis=0)
            im2 = ax2.imshow(gcn, cmap=cm.copper)
            ax2.set_axis_off()
            fig.colorbar(im2, ax=ax2, aspect=10, fraction=.08)
            ax2.set_title('Graph Conv Net')
            
            plt.show()
        
        # make it interactive
        t_max = len(erosion1) - 1
        widgets.interact(plot_func, t=widgets.IntSlider(value=0, 
                                                        min=0, 
                                                        max=t_max, 
                                                        step=1,
                                                        description='Timesteps:'))

    # static plot
    else:
        assert choice is not None, "Need argument 'choice' if not interactive."
        
        # define layout
        l = len(choice)
        fig, axs = plt.subplots(nrows=2, ncols=l, figsize=(3*l, 6))
        
        # make plots
        for i, p in enumerate(choice):    
            
            ax0 = axs[0,i]
            plt.sca(ax0)
            g0 = RasterModelGrid((rows, columns), 1)
            g0.add_field('topographic__elevation', erosion1[p], at='node', clobber=True)
            imshow_grid(g0, 'topographic__elevation', shrink=0.75)
            ax0.set_xlabel('')
            if i == 0:
                ax0.set_ylabel('Landlab', labelpad=8, size='x-large')
            else:
                ax0.set_ylabel('')
            ax0.set_xticks([])
            ax0.set_yticks([])
            ax0.set_frame_on(False)
        

            # bottom row
            ax1 = axs[1,i]
            plt.sca(ax1)
            g1 = RasterModelGrid((rows, columns), 1)
            g1.add_field('topographic__elevation', erosion2[p], at='node', clobber=True)
            imshow_grid(g1, 'topographic__elevation', shrink=0.75)
            ax1.set_xlabel(f't = {p+1}', labelpad=8, size='x-large')
            if i == 0:
                ax1.set_ylabel(gnn_name, labelpad=8, size='x-large')
            else:
                ax1.set_ylabel('')
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_frame_on(False)


#############################################################################################################################
#############################################################################################################################

def compare_erosion_voronoi_2D(simulation_1, simulation_2, gnn_name, choice=[0, 14, 29, 49], xs=None, ys=None, tikz=False):
    from landlab import VoronoiDelaunayGrid
        
    # define layout
    l = len(choice)
    fig, axs = plt.subplots(nrows=2, ncols=l, figsize=(3*l, 6))
    fig.subplots_adjust(wspace=0.15, hspace=0)
    
    
    # make plots
    for i, p in enumerate(choice):    
            
        # top row
        ax0 = axs[0,i]
        plt.sca(ax0)
        g0 = VoronoiDelaunayGrid(xs, ys)
        g0.add_field('topographic__elevation', simulation_1[p], at='node', clobber=True)
        imshow_grid(g0, 'topographic__elevation', shrink=0.75)
        ax0.set_xlabel('')
        if i == 0:
            ax0.set_ylabel('Landlab', labelpad=8, size='x-large')
        else:
            ax0.set_ylabel('')
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_frame_on(False)

        # bottom row
        ax1 = axs[1,i]
        plt.sca(ax1)
        g1 = VoronoiDelaunayGrid(xs, ys)
        g1.add_field('topographic__elevation', simulation_2[p], at='node', clobber=True)
        imshow_grid(g1, 'topographic__elevation', shrink=0.75)
        ax1.set_xlabel(f't = {p+1}', labelpad=8, size='x-large')
        if i == 0:
            ax1.set_ylabel(gnn_name, labelpad=8, size='x-large')
        else:
            ax1.set_ylabel('')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_frame_on(False)
        
    if tikz:
        import tikzplotlib
        tikzplotlib.save(f'{gnn_name}.tex')
        

#############################################################################################################################
#############################################################################################################################

def compare_erosion_raster_3D(erosion1, erosion2, grid, choice=None, interactive=False, square=True, rows=None, columns=None):
    '''
    This function takes two erosion simulations as returned by tests.simulate_erosion() and plots a 
    interactive 3D topographic comparison
    '''

    assert len(erosion1) == len(erosion2)
    
    # define dimensions
    if square:
        dim = int(np.sqrt(len(erosion1[0])))
        rows, columns = dim, dim
    else:
        assert rows is not None, "Need number of rows if not square"
        assert columns is not None, "Need number of columns if not square"

    # find x and y positions
    xs = grid.x_of_node.reshape((rows, columns))
    ys = grid.y_of_node.reshape((rows, columns))
    
    # interactive plot
    if interactive:
        
        # define surface plots
        def plot_func(t):
            
            # define layout
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,10), subplot_kw=dict(projection='3d'))
            
            # plot 1
            surf1 = ax1.plot_surface(xs, ys, erosion1[t].reshape((rows, columns)), cmap=cm.copper)
            ax1.set_axis_off()
            fig.colorbar(surf1, ax=ax1, aspect=8, fraction=.08)
            ax1.set_title('Landlab')
            
            # plot 2
            surf2 = ax2.plot_surface(xs, ys, erosion2[t].reshape((rows, columns)), cmap=cm.copper)
            ax2.set_axis_off()
            fig.colorbar(surf2, ax=ax2, aspect=8, fraction=.08)
            ax2.set_title('Graph Conv Net')

            plt.show()
            
        # make it interactive
        t_max = len(erosion1) - 1
        widgets.interact(plot_func, t=widgets.IntSlider(value=0, 
                                                        min=0, 
                                                        max=t_max, 
                                                        step=1,
                                                        description='Timesteps:'))
            
    # static plot
    else:
        assert choice is not None, "Need argument 'choice' if not interactive."
        
        # define layout
        l = len(choice)
        fig, axs = plt.subplots(nrows=2, ncols=l, figsize=(3*l, 6), subplot_kw=dict(projection='3d'))
        
        # make plots
        for i, p in enumerate(choice):    
            
            # top row
            ax1 = axs[0,i] 
            ll = erosion1[p].reshape((rows, columns))
            surf1 = ax1.plot_surface(xs, ys, ll, cmap=cm.copper)
            fig.colorbar(surf1, ax=ax1, aspect=14, fraction=.05)
            ax1.set_axis_off()
            ax1.set_title(f't = {p}')

            # bottom row
            ax2 = axs[1,i]
            gcn = erosion2[p].reshape((rows, columns))
            surf2 = ax2.plot_surface(xs, ys, gcn, cmap=cm.copper)
            fig.colorbar(surf2, ax=ax2, aspect=14, fraction=.05)
            ax2.set_axis_off()

        plt.show()


#############################################################################################################################
#############################################################################################################################

def compare_drainage_raster_2D(drainage1, drainage2, choice=None, interactive=False, square=True, rows=None, columns=None):
    
    '''
    This function takes two drainage simulations as returned by tests.simulate_drainage() and plots a 
    2D comparison either interactively or statically at given timepoints
    '''

    assert len(drainage1) == len(drainage2)
    
    # define dimensions
    if square:
        dim = int(np.sqrt(len(drainage1[0])))
        rows, columns = dim, dim
    else:
        assert rows is not None, "Need number of rows if not square"
        assert columns is not None, "Need number of columns if not square"

    # interactive plot
    if interactive:

        def plot_func(t):
            # define layout
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,8))

            # plot 1
            ll = np.flip(drainage1[t].reshape((rows, columns)), axis=0)
            im1 = ax1.imshow(ll, cmap=cm.RdBu)
            ax1.set_axis_off()
            fig.colorbar(im1, ax=ax1, aspect=10, fraction=.08)
            ax1.set_title('Landlab')

            # plot 2
            gcn = np.flip(drainage2[t].reshape((rows, columns)), axis=0)
            im2 = ax2.imshow(gcn, cmap=cm.RdBu)
            ax2.set_axis_off()
            fig.colorbar(im2, ax=ax2, aspect=10, fraction=.08)
            ax2.set_title('Graph Conv Net')

            plt.show()
        
        # make it interactive
        t_max = len(drainage1) - 1
        widgets.interact(plot_func, t=widgets.IntSlider(value=0, 
                                                        min=0, 
                                                        max=t_max, 
                                                        step=1,
                                                        description='Timesteps:'))

    # static plot
    else:
        assert choice is not None, "Need argument 'choice' if not interactive."
        
        # define layout
        l = len(choice)
        fig, axs = plt.subplots(nrows=2, ncols=l, figsize=(3*l, 6))
        
        # make plots
        for i, p in enumerate(choice):    
            
            # top row
            ax1 = axs[0,i] 
            ll = np.flip(drainage1[p].reshape((rows, columns)), axis=0)
            im1 = ax1.imshow(ll, cmap=cm.RdBu)
            fig.colorbar(im1, ax=ax1, aspect=12, fraction=.07)
            ax1.set_axis_off()
            ax1.set_title(f't = {p}')

            # bottom row
            ax2 = axs[1,i]
            gcn = np.flip(drainage2[p].reshape((rows, columns)), axis=0)
            im2 = ax2.imshow(gcn, cmap=cm.RdBu)
            fig.colorbar(im2, ax=ax2, aspect=12, fraction=.07)
            ax2.set_axis_off()

        plt.show()


#############################################################################################################################
#############################################################################################################################

def compare_drainage_voronoi_2D(simulation_1, simulation_2, choice=[0, 10, 20, 30, 40, 49], xs=None, ys=None):
    from landlab import VoronoiDelaunayGrid
    print('Simulation loss:', round(float(simulation_loss(simulation_1, simulation_2)), ndigits=4))
    print()
    for i in choice:
        print(f't = {i}:')
        g = VoronoiDelaunayGrid(xs, ys)
        g.add_field('drainage', simulation_1[i], at='node', clobber=True)
        plt.figure(figsize=(4,4))
        imshow_grid(g, 'drainage')
        plt.show()
        g = VoronoiDelaunayGrid(xs, ys)
        g.add_field('drainage', simulation_2[i], at='node', clobber=True)
        plt.figure(figsize=(4,4))
        imshow_grid(g, 'drainage')
        plt.show()


#############################################################################################################################
#############################################################################################################################

def plot_loss(train_loss, test_loss, gnn_name=None, title='', tikz=False):
    assert len(train_loss) == len(test_loss)
    plt.figure(figsize=(6,2))
    x = range(len(train_loss))
    plt.plot(x, train_loss, color='yellow', label='Train loss')
    plt.plot(x, test_loss, color='blue', label='Test loss')
    plt.xlabel('Epoch', labelpad=10, size='x-large')
    plt.ylabel('Loss', labelpad=10, size='x-large')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(title)
    plt.legend(fontsize=13)
    if tikz:
        import tikzplotlib
        tikzplotlib.save(f'{gnn_name}.tex')
    else:
        plt.show()


#############################################################################################################################
#############################################################################################################################
