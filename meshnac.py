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
from util.networks import Net
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
file_list = glob.glob(file_path + '/*.obj')
file_list = sorted(file_list)
gt_file = file_list[0]
n1_file = file_list[1]
n2_file = file_list[2]
mesh_name = gt_file.split('/')[-2]

gt_mesh = Mesh(gt_file)
n1_mesh = Mesh(n1_file)
n2_mesh = Mesh(n2_file)

# node-features and edge-index
x = torch.tensor(n2_mesh.vn, dtype=torch.float).to(device)
y = torch.tensor(n1_mesh.vn, dtype=torch.float).to(device)
gt = torch.tensor(gt_mesh.vn, dtype=torch.float).to(device)

edge_index = torch.tensor(n2_mesh.edges.T, dtype=torch.long).to(device)
edge_index = torch.cat([edge_index, edge_index[[1,0],:]], dim=1)

# create model instance
model = Net(FLAGS.skip).to(device)
model.train()

# output experimental conditions
dt_now = datetime.datetime.now()
log_dir = "./logs/" + mesh_name + dt_now.isoformat()
writer = SummaryWriter(log_dir=log_dir)
log_file = "datasets/" + mesh_name + "/condition.json"
condition = {"n2": n2_file, "n1": n1_file, "gt": gt_file, "iter": FLAGS.iter, "skip": FLAGS.skip, "lr": FLAGS.lr}

with open(log_file, mode="w") as f:
    l = json.dumps(condition, indent=2)
    f.write(l)

# learning loop
optimizer = torch.optim.Adamax(model.parameters(), lr=FLAGS.lr)
criterion = torch.nn.L1Loss()

for epoch in range(1, FLAGS.iter+1):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    #loss = Loss.mse_loss(out, y)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    writer.add_scalar("mse_loss", loss, epoch)
    
    if epoch % 10 == 0:
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
        
    if epoch % 50 == 0:
        model.eval()
        t_out = model(y, edge_index)
        t_loss = criterion(t_out, gt)
        writer.add_scalar("test_loss", t_loss, epoch)
        print("test_loss: ", t_loss)

# test
model.eval()
out = model(x, edge_index)
o_mesh = ObjMesh(n1_file)
o_mesh.vs = o_mesh.vertices = out.to('cpu').detach().numpy().copy()
o_mesh.faces = n1_mesh.faces
o_mesh.save('datasets/' + mesh_name + '/output.obj')
init_mad = Loss.mad(n1_mesh, gt_mesh)
final_mad = Loss.mad(o_mesh, gt_mesh)
print("initial_mad: ", init_mad)
print("final_mad: ", final_mad)