import numpy as np
import torch
import copy
import datetime
import os
import sys
import glob
import argparse
import json
import util.loss as Loss
import util.models as Models
from util.objmesh import ObjMesh
from util.models import Dataset, Mesh
from util.networks import PosNet, NormalNet

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data

parser = argparse.ArgumentParser(description='DMP_adv for mesh')
parser.add_argument('-i', '--input', type=str, required=True)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--iter', type=int, default=5000)
parser.add_argument('--skip', type=bool, default=False)
parser.add_argument('--lap', type=float, default=1.4)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--ntype', type=str, default='hybrid')
FLAGS = parser.parse_args()

for k, v in vars(FLAGS).items():
    print('{:10s}: {}'.format(k, v))

device = torch.device('cuda:' + str(FLAGS.gpu) if torch.cuda.is_available() else 'cpu')

file_path = FLAGS.input
gt_file = glob.glob(file_path + '/*_gt.obj')[0]
n_file = glob.glob(file_path + '/*_noise.obj')[0]
s_file = glob.glob(file_path + '/*_smooth.obj')[0]

mesh_name = gt_file.split('/')[-2]

gt_mesh = Mesh(gt_file)
n_mesh = Mesh(n_file)
o1_mesh = Mesh(n_file)
#o2_mesh = Mesh(n_file)
s_mesh = Mesh(s_file)

# node-features and edge-index
np.random.seed(314)
z1 = np.random.normal(size=(n_mesh.vs.shape[0], 16))
np.random.seed(159)
z2 = np.random.normal(size=(n_mesh.fn.shape[0], 16))
z1, z2 = torch.tensor(z1, dtype=torch.float, requires_grad=True), torch.tensor(z2, dtype=torch.float, requires_grad=True)

x_pos = torch.tensor(s_mesh.vs, dtype=torch.float).to(device)
x_norm = torch.tensor(n_mesh.fn, dtype=torch.float).to(device)

edge_index = torch.tensor(n_mesh.edges.T, dtype=torch.long).to(device)
edge_index = torch.cat([edge_index, edge_index[[1,0],:]], dim=1)

face_index = torch.from_numpy(n_mesh.f_edges).to(device)

# create data
data = Data(x=z1, z1=z1, z2=z2, x_pos=x_pos, x_norm=x_norm, edge_index=edge_index, face_index=face_index)
dataset = Dataset(data)

# create model instance
posnet = PosNet(device).to(device)
normnet = NormalNet(device).to(device)
posnet.train()
normnet.train()

# output experimental conditions
dt_now = datetime.datetime.now()
log_dir = "./logs/" + mesh_name + dt_now.isoformat()
writer = SummaryWriter(log_dir=log_dir)
log_file = "datasets/" + mesh_name + "/condition.json"
condition = {"n": n_file, "s": s_file, "gt": gt_file, "iter": FLAGS.iter, "skip": FLAGS.skip, "lr": FLAGS.lr}
os.makedirs("datasets/" + mesh_name + "/output", exist_ok=True)

with open(log_file, mode="w") as f:
    l = json.dumps(condition, indent=2)
    f.write(l)

# learning loop
min_mad = 1000
min_norm_loss = 1000
init_mad = Loss.mad(n_mesh, gt_mesh)
init_vn_loss = Loss.mse_loss(torch.tensor(n_mesh.vn, dtype=float), torch.tensor(gt_mesh.vn, dtype=float))
init_fn_loss = Loss.mse_loss(torch.tensor(n_mesh.fn, dtype=float), torch.tensor(gt_mesh.fn, dtype=float))
print("init_mad: ", init_mad, " init_vn_loss: ", float(init_vn_loss), " init_fn_loss: ", float(init_fn_loss))

optimizer_pos = torch.optim.Adam(posnet.parameters(), lr=FLAGS.lr)
optimizer_norm = torch.optim.Adam(normnet.parameters(), lr=FLAGS.lr)

for epoch in range(1, FLAGS.iter+1):
    if FLAGS.ntype == "pos":
        posnet.train()
        optimizer_pos.zero_grad()
        pos = posnet(dataset)
        loss_pos1 = Loss.mse_loss(pos, torch.tensor(n_mesh.vs, dtype=float).to(device))
        loss_pos2 = FLAGS.lap * Loss.mesh_laplacian_loss(pos, n_mesh.ve, n_mesh.edges)
        o1_mesh.vs = pos.to('cpu').detach().numpy().copy()

        norm = Models.compute_fn(pos, n_mesh.faces).float()
        """test"""
        Mesh.compute_face_normals(o1_mesh)
        Mesh.compute_vert_normals(o1_mesh)
        #loss_pos3 = Loss.norm_cos_loss(torch.tensor(gt_mesh.vn, dtype=float), torch.tensor(norm2, dtype=float))
        loss_pos3 = Loss.norm_cos_loss(torch.tensor(gt_mesh.fn, dtype=float).to(device), norm)
        #loss_pos3 = 10 * Loss.pos4norm(pos.to('cpu'), o1_mesh, torch.tensor(gt_mesh.fn, dtype=float))

        loss_pos = loss_pos1 + loss_pos2 + loss_pos3
        loss_pos.backward()
        optimizer_pos.step()
        writer.add_scalar("pos1", loss_pos1, epoch)
        writer.add_scalar("pos2", loss_pos2, epoch)
        writer.add_scalar("pos3", loss_pos3, epoch)
        writer.add_scalar("pos", loss_pos, epoch)
    
    elif FLAGS.ntype == "norm":
        normnet.train()
        optimizer_norm.zero_grad()
        norm = normnet(dataset)

        loss_norm1 = Loss.norm_cos_loss(norm, torch.tensor(n_mesh.fn, dtype=float).to(device))
        loss_norm2 = 5.0 * Loss.fn_lap_loss(norm, n_mesh.f2f_mat)
        loss_norm = loss_norm1 + loss_norm2
        loss_norm.backward()
        optimizer_norm.step()

        writer.add_scalar("norm1", loss_norm1, epoch)
        writer.add_scalar("norm2", loss_norm2, epoch)
        writer.add_scalar("norm", loss_norm, epoch)
    
    elif FLAGS.ntype == "hybrid":
        posnet.train()
        normnet.train()
        optimizer_pos.zero_grad()
        optimizer_norm.zero_grad()

        pos = posnet(dataset)
        loss_pos1 = Loss.mse_loss(pos, torch.tensor(n_mesh.vs, dtype=float).to(device))
        loss_pos2 = FLAGS.lap * Loss.mesh_laplacian_loss(pos, n_mesh.ve, n_mesh.edges)

        norm = normnet(dataset)
        """ regress vn
        loss_norm1 = Loss.norm_cos_loss(norm, torch.tensor(n_mesh.vn, dtype=float).to(device))
        loss_norm2 = 5.0 * Loss.norm_laplacian_loss(norm, n_mesh.ve, n_mesh.edges)
        """
        loss_norm1 = Loss.norm_cos_loss(norm, torch.tensor(n_mesh.fn, dtype=float).to(device))
        loss_norm2 = 5.0 * Loss.fn_lap_loss(norm, n_mesh.f2f_mat)

        fn2 = Models.compute_fn(pos, n_mesh.faces).float()
        loss_pos3 = Loss.norm_cos_loss(fn2, norm)
        #loss_pos3 = Loss.norm_cos_loss(fn2, torch.tensor(gt_mesh.fn).float().to(device))
        
        loss_pos = loss_pos1 + loss_pos2 + loss_pos3
        loss_norm = loss_norm1 + loss_norm2
        loss_pos.backward(retain_graph=True)
        loss_norm.backward()
        optimizer_pos.step()
        optimizer_norm.step()

        writer.add_scalar("pos1", loss_pos1, epoch)
        writer.add_scalar("pos2", loss_pos2, epoch)
        writer.add_scalar("pos3", loss_pos3, epoch)
        writer.add_scalar("pos", loss_pos, epoch)
        writer.add_scalar("norm1", loss_norm1, epoch)
        writer.add_scalar("norm2", loss_norm2, epoch)
        writer.add_scalar("norm", loss_norm, epoch)

    if epoch % 10 == 0:
        if FLAGS.ntype == "pos":
            print('Epoch %d || Loss_P: %.4f' % (epoch, loss_pos.item()))
        elif FLAGS.ntype == "norm":
            print('Epoch %d || Loss_N: %.4f' % (epoch, loss_norm.item()))
        else:
            print('Epoch %d || Loss_P: %.4f | Loss_N: %.4f' % (epoch, loss_pos.item(), loss_norm.item()))
        
    if epoch % 50 == 0:
        if FLAGS.ntype == "pos":
            o1_mesh.vs = pos.to('cpu').detach().numpy().copy()
            Mesh.compute_face_normals(o1_mesh)
            Mesh.compute_vert_normals(o1_mesh)
            mad_value = Loss.mad(o1_mesh, gt_mesh)
            min_mad = min(mad_value, min_mad)
            print("mad_value: ", mad_value, "min_mad: ", min_mad)
            writer.add_scalar("MAD", mad_value, epoch)
            Mesh.save(o1_mesh, "datasets/" + mesh_name + "/output/" + str(epoch) + "_pos.obj")

        elif FLAGS.ntype == "norm":
            test_norm_loss = Loss.mse_loss(norm, torch.tensor(gt_mesh.fn, dtype=float).to(device))
            min_norm_loss = min(min_norm_loss, test_norm_loss)
            print("test_norm_loss: ", float(test_norm_loss), "min_norm_loss: ", float(min_norm_loss))
            writer.add_scalar("test_norm", test_norm_loss, epoch)
        else:
            o1_mesh.vs = pos.to('cpu').detach().numpy().copy()
            Mesh.compute_face_normals(o1_mesh)
            Mesh.compute_vert_normals(o1_mesh)
            """DMP-Pos"""
            mad_value = Loss.mad(o1_mesh, gt_mesh)
            min_mad = min(mad_value, min_mad)
            print("mad_value: ", mad_value, "min_mad: ", min_mad)
            writer.add_scalar("MAD", mad_value, epoch)
            Mesh.save(o1_mesh, "datasets/" + mesh_name + "/output/" + str(epoch) + "_pos.obj")
            
            """DMP-Norm"""
            test_norm_loss = Loss.mse_loss(norm, torch.tensor(gt_mesh.fn, dtype=float).to(device))
            min_norm_loss = min(min_norm_loss, test_norm_loss)
            print("test_norm_loss: ", float(test_norm_loss), "min_norm_loss: ", float(min_norm_loss))
            writer.add_scalar("test_norm", test_norm_loss, epoch)
            """
            div = Models.build_div(n_mesh, norm.to('cpu').detach().numpy().copy())
            new_vs = Models.cg(n_mesh, div)
            o2_mesh.vs = new_vs
            Mesh.save(o2_mesh, 'datasets/' + mesh_name + '/output/' + str(epoch) + '_norm.obj')
            """
