import numpy as np
import torch
import copy
import datetime
import os
import sys
import glob
import argparse
import json
from util.objmesh import ObjMesh
from util.models import Dataset, Mesh
from util.networks import NacNet
import util.loss as Loss

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data

parser = argparse.ArgumentParser(description='NAC for mesh')
parser.add_argument('-i', '--input', type=str, required=True)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--iter', type=int, default=5000)
parser.add_argument('--skip', type=bool, default=False)
FLAGS = parser.parse_args()

for k, v in vars(FLAGS).items():
    print('{:10s}: {}'.format(k, v))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file_path = FLAGS.input
gt_file = glob.glob(file_path + '/*_gt.obj')[0]
mesh_name = gt_file.split('/')[-2]
gt_mesh = Mesh(gt_file)
n1_mesh = Mesh(gt_file)
n2_mesh = Mesh(gt_file)

# node-features and edge-index
np.random.seed(314)
n1 = np.random.normal(0, 0.002, (len(n1_mesh.vs), 3))
n1_mesh.vs = gt_mesh.vs + n1
o_mesh = ObjMesh(gt_file)
o_mesh.vs = o_mesh.vertices = n1_mesh.vs
o_mesh.faces = n1_mesh.faces
o_mesh.save(os.path.join(file_path, mesh_name) + '_n1.obj')

edge_index = torch.tensor(n1_mesh.edges.T, dtype=torch.long).to(device)
edge_index = torch.cat([edge_index, edge_index[[1,0],:]], dim=1)

# create model instance
model = NacNet().to(device)
model.train()

# output experimental conditions
dt_now = datetime.datetime.now()
log_dir = "./logs/" + mesh_name + dt_now.isoformat()
writer = SummaryWriter(log_dir=log_dir)

# learning loop
optimizer = torch.optim.Adamax(model.parameters(), lr=FLAGS.lr)
criterion = torch.nn.L1Loss()

for epoch in range(1, FLAGS.iter+1):
    n2 = np.random.normal(0, 0.002, (len(n1_mesh.vs), 3))
    n2_mesh.vs = n1_mesh.vs + n2
    x = torch.from_numpy(n2_mesh.vs).float().to(device)
    y = torch.from_numpy(n1_mesh.vs).float().to(device)
    
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    writer.add_scalar("L1Loss", loss, epoch)
    
    if epoch % 10 == 0:
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
        
    if epoch % 50 == 0:
        g = torch.from_numpy(gt_mesh.vs).float().to(device)
        model.eval()
        out = model(y, edge_index)
        loss = criterion(out, g)
        writer.add_scalar("test_L1Loss", loss, epoch)
        print("test_L1Loss: ", loss)
        o_mesh = ObjMesh(gt_file)
        o_mesh.vs = o_mesh.vertices = out.to('cpu').detach().numpy().copy()
        o_mesh.faces = n1_mesh.faces
        o_mesh.save('datasets/' + mesh_name + '/nac_output/' + str(epoch) + '_output.obj')

# test
model.eval()
out = model(x, edge_index)
