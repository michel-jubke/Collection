from time import time

print('Start imports and argparse...')
start = time()

from engine.data import RasterGrid, VoronoiGrid, LandlabData
from engine.models.gcn import GCN16v1, GCN16v2, GCN32v1, GCN32v2
from engine.models.sage import SAGE16v1, SAGE16v2, SAGE32v1, SAGE32v2, SAGE48v1
from engine.train import train_and_test
from engine.simulate import simulate_erosion, simulate_drainage

import torch
import torch.nn.functional as F

import numpy as np

import argparse

###############################################################################################################

# parse commandline args
parser = argparse.ArgumentParser()

# meta stuff
parser.add_argument('--gridsize',             default=30,           type=int)
parser.add_argument('--gridtype',             default='voronoi',    type=str)
parser.add_argument('--spacing',              default=1,            type=int)
parser.add_argument('--timesteps',            default=50,           type=int)
parser.add_argument('--dt',                   default=None,         type=int)
parser.add_argument('--uplift',               default=20,           type=int) # topographic uplift relative to init

# trainign stuff
parser.add_argument('--model',                default=None,         type=str)
parser.add_argument('--use_weights',          default='no',         type=str)
parser.add_argument('--target',               default='topo_delta', type=str)
parser.add_argument('--epochs',               default=5000,         type=int)

# hyperparameters
parser.add_argument('--batch_size',           default=128,          type=int)
parser.add_argument('--learning_rate',        default=0.0001,       type=float)

# talk to the os
parser.add_argument('--save_dir',             default=None,         type=str)
parser.add_argument('--save',                 default='yes',        type=str)

# parse
args = parser.parse_args()

# error handling
assert args.gridsize in [30, 50, 80],                     "Only 30×30, 50×50 and 80×80 grids are supported."
assert args.gridtype in ['raster', 'voronoi'],            "Gridtype must be 'raster' or 'voronoi'"
assert args.uplift >= 0,                                  "Its uplift, not downlift, stupid."
assert args.target in ['topo', 'topo_delta', 'drainage'], "Please choose valid target option"

assert args.save in ['yes', 'no'],                        "Please choose if results should be saved or not."
if args.save == 'yes': 
    assert args.save_dir is not None,                     "Please specify a folder within the working diretory to save output."

stop = time()
print(f'Done with imports and argparse, needed {round(stop-start, ndigits=2)} seconds.')
print()

###############################################################################################################

print(f"Start setup for {args.model}...")
start = time()
print()

print(args)
print()

# where, if and how to safe all the output
work_dir = '/mnt/qb/work/hennig/mjubke62'
save_dir = args.save_dir
save = True if args.save == 'yes' else False

# grid properties
gridsize = args.gridsize
rows, columns = gridsize, gridsize
spacing = args.spacing
gridtype = args.gridtype
uplift = args.uplift

# stepsize
if gridsize == 30: dt = 100
if gridsize == 50: dt = 120
if gridsize == 80: dt = 150
if args.dt is not None: dt = args.dt

# timesteps, default: 50
timesteps = args.timesteps

# model
model_name = args.model
target = args.target
learning_rate = args.learning_rate
batch_size = args.batch_size 

# load initial topographies
if rows == columns == 30:
    topo_init_train = np.load(f'{work_dir}/data-init/topo-init-train-{gridtype}-30.npy')
    topo_init_test = np.load(f'{work_dir}/data-init/topo-init-test-{gridtype}-30.npy')

if rows == columns == 50:
    topo_init_train = np.load(f'{work_dir}/data-init/topo-init-train-{gridtype}-50.npy')
    topo_init_test = np.load(f'{work_dir}/data-init/topo-init-test-{gridtype}-50.npy')

if rows == columns == 80:
    topo_init_train = np.load(f'{work_dir}/data-init/topo-init-train-{gridtype}-80.npy')
    topo_init_test = np.load(f'{work_dir}/data-init/topo-init-test-{gridtype}-80.npy')

# init data object
data_train = LandlabData()
data_test = LandlabData()

# fill data object with erosion timeseries
for i in range(len(topo_init_train)):
    topo = topo_init_train[i]
    if gridtype == 'raster':
        g = RasterGrid(rows, columns, spacing, topo, uplift)
    if gridtype == 'voronoi':
        np.random.seed(i)
        xs = np.random.rand(rows*columns) * rows*spacing
        ys = np.random.rand(rows*columns) * rows*spacing
        g = VoronoiGrid(xs, ys, topo, uplift)
    data_train.add(g, timesteps, dt, target_type=target)

for i in range(len(topo_init_test)):
    topo = topo_init_test[i]
    if gridtype == 'raster':
        g = RasterGrid(rows, columns, spacing, topo, uplift)
    if gridtype == 'voronoi':
        seed = i*1312
        np.random.seed(seed)
        xs = np.random.rand(rows*columns) * rows*spacing
        ys = np.random.rand(rows*columns) * rows*spacing
        g = VoronoiGrid(xs, ys, topo, uplift)
    data_test.add(g, timesteps, dt, target_type=target)


print(f"Number of datapoints in training data: {len(data_train)}")
print()

# prepare dataloaders
train_loader = data_train.make_dataloader(batch_size=batch_size)
test_loader = data_test.make_dataloader(batch_size=batch_size)

###############################################################################################################

# set number of input and output features
in_channels = 1
out_channels = 1

# instantiate model
use_edge_weights = True if args.use_weights == 'yes' else False
    
if model_name == 'GCN16v1':
    model = GCN16v1(in_channels, out_channels, use_edge_weights)
if model_name == 'GCN16v2':
    model = GCN16v2(in_channels, out_channels, use_edge_weights)
if model_name == 'GCN32v1':
    model = GCN32v1(in_channels, out_channels, use_edge_weights)
if model_name == 'GCN32v2':
    model = GCN32v2(in_channels, out_channels, use_edge_weights)

if model_name == 'SAGE16v1':
    model = SAGE16v1(in_channels, out_channels)
if model_name == 'SAGE16v2':
    model = SAGE16v2(in_channels, out_channels)
if model_name == 'SAGE32v1':
    model = SAGE32v1(in_channels, out_channels)
if model_name == 'SAGE32v2':
    model = SAGE32v2(in_channels, out_channels)
if model_name == 'SAGE48v1':
    model = SAGE48v1(in_channels, out_channels)

# print model
print('Model:')
print(model)
print()

# choose criterion and optimizer
criterion = F.smooth_l1_loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# set number of epochs
epochs = args.epochs

stop = time()
print(f'Done with setup, needed {round(stop-start, ndigits=2)} seconds.')
print()

# and now: run! 
best_model, train_loss, test_loss = train_and_test(model, epochs, train_loader, test_loader, criterion, optimizer)

# simulation
print('Start simulating...')
start = time()

best_model.eval()
simulations_landlab = np.empty((len(topo_init_test), 50, gridsize**2))
simulations_model   = np.empty((len(topo_init_test), 50, gridsize**2))

for i in range(len(topo_init_test)):
    topo = topo_init_test[i]
    if gridtype == 'raster':
        g = RasterGrid(rows, columns, spacing, topo, uplift)
    if gridtype == 'voronoi':
        seed = i*1312
        np.random.seed(seed)
        xs = np.random.rand(rows*columns) * rows*spacing
        ys = np.random.rand(rows*columns) * rows*spacing
        g = VoronoiGrid(xs, ys, topo, uplift)
    if target == 'topo_delta':
        simulations_landlab[i] = simulate_erosion(g.grid, 49, 'landlab', dt=dt)
        simulations_model[i] = simulate_erosion(g.grid, 49, 'gcn_delta', best_model)
    if target == 'drainage':
        simulations_landlab[i] = simulate_drainage(g.grid, 50, 'landlab', best_model, dt=dt)
        simulations_model[i] = simulate_drainage(g.grid, 50, 'gcn', best_model, dt=dt)

stop = time()
print(f'Done with simulations, needed {round(stop-start, ndigits=2)} seconds.')
print()

print('Start saving...')
start = time()

if save: torch.save(train_loss, f'{work_dir}/{save_dir}/losses/{model_name}-{gridsize}-{gridtype}-LR{learning_rate}-{target}-train-loss.pt')
if save: torch.save(test_loss, f'{work_dir}/{save_dir}/losses/{model_name}-{gridsize}-{gridtype}-LR{learning_rate}-{target}-test-loss.pt')
if save: torch.save(best_model, f'{work_dir}/{save_dir}/models/{model_name}-{gridsize}-{gridtype}-LR{learning_rate}-{target}-model.pt')
if save: np.save(f'{work_dir}/{save_dir}/simulations/{model_name}-{gridsize}-{gridtype}-LR{learning_rate}-{target}-simulations-landlab', simulations_landlab)
if save: np.save(f'{work_dir}/{save_dir}/simulations/{model_name}-{gridsize}-{gridtype}-LR{learning_rate}-{target}-simulations-model', simulations_model)

stop = time()
print(f'Finished saving, needed {round(stop-start, ndigits=2)} seconds.')
print()

print(f'Done with {model_name}')

