import numpy as np
import torch
import torch.nn as nn
import copy
import datetime
import os
import sys
import glob
import argparse
import json
import util.loss as Loss
import util.models as Models
import util.datamaker as Datamaker
from util.objmesh import ObjMesh
from util.datamaker import Dataset
from util.mesh import Mesh
from util.networks import PosNet, NormalNet, SphereNet

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data

parser = argparse.ArgumentParser(description='DMP_adv for mesh')
parser.add_argument('-i', '--input', type=str, required=True)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--iter', type=int, default=5000)
parser.add_argument('--lap', type=float, default=1.4)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--ntype', type=str, default='hybrid')
FLAGS = parser.parse_args()

for k, v in vars(FLAGS).items():
    print('{:10s}: {}'.format(k, v))

""" --- create dataset --- """
mesh_dic, dataset = Datamaker.create_dataset(FLAGS.input)
gt_file, n_file, s_file, mesh_name = mesh_dic["gt_file"], mesh_dic["n_file"], mesh_dic["s_file"], mesh_dic["mesh_name"]
gt_mesh, n_mesh, o1_mesh, s_mesh = mesh_dic["gt_mesh"], mesh_dic["n_mesh"], mesh_dic["o1_mesh"], mesh_dic["s_mesh"]

""" --- create model instance --- """
device = torch.device('cuda:' + str(FLAGS.gpu) if torch.cuda.is_available() else 'cpu')
posnet = PosNet(device).to(device)
normnet = NormalNet(device).to(device)
#normnet = SphereNet(device).to(device)
optimizer_pos = torch.optim.Adam(posnet.parameters(), lr=FLAGS.lr)
optimizer_norm = torch.optim.Adam(normnet.parameters(), lr=1.0e-3)

""" --- output experimental conditions --- """
dt_now = datetime.datetime.now()
log_dir = "./logs/" + mesh_name + dt_now.isoformat()
writer = SummaryWriter(log_dir=log_dir)
log_file = log_dir + "/condition.json"
condition = {"PosNet": str(posnet).split("\n"), "NormNet": str(normnet).split("\n"), "optimizer_pos": str(optimizer_pos).split("\n"), "optimizer_norm": str(optimizer_norm).split("\n")}
with open(log_file, mode="w") as f:
    l = json.dumps(condition, indent=2)
    f.write(l)

os.makedirs("datasets/" + mesh_name + "/output", exist_ok=True)

""" --- initial condition --- """
min_mad = 1000
min_norm_loss = 1000
init_mad = Loss.mad(n_mesh.fn, gt_mesh.fn)
init_vn_loss = Loss.rmse_loss(n_mesh.vn, gt_mesh.vn)
init_fn_loss = Loss.rmse_loss(n_mesh.fn, gt_mesh.fn)
print("init_mad: ", init_mad, " init_vn_loss: ", float(init_vn_loss), " init_fn_loss: ", float(init_fn_loss))

""" --- learning loop --- """
for epoch in range(1, FLAGS.iter+1):
    if FLAGS.ntype == "pos":
        posnet.train()
        optimizer_pos.zero_grad()
        pos = posnet(dataset)
        loss_pos1 = Loss.rmse_loss(pos, n_mesh.vs)
        loss_pos2 = FLAGS.lap * Loss.mesh_laplacian_loss(pos, n_mesh)
        o1_mesh.vs = pos.to('cpu').detach().numpy().copy()
        fn2 = Models.compute_fn(pos, n_mesh.faces).float()
        
        loss_pos = loss_pos1 + loss_pos2
        loss_pos.backward()
        optimizer_pos.step()
        writer.add_scalar("pos1", loss_pos1, epoch)
        writer.add_scalar("pos2", loss_pos2, epoch)
        writer.add_scalar("pos", loss_pos, epoch)
    
    elif FLAGS.ntype == "norm":
        normnet.train()
        optimizer_norm.zero_grad()
        norm = normnet(dataset)

        loss_norm1 = Loss.norm_cos_loss(norm, n_mesh.fn)
        loss_norm2 = 0.5 * Loss.fn_bnf_loss(norm, n_mesh)
        loss_norm = loss_norm1 + loss_norm2
        loss_norm.backward()
        nn.utils.clip_grad_norm_(normnet.parameters(), 1.0)
        optimizer_norm.step()

        writer.add_scalar("norm1", loss_norm1, epoch)
        writer.add_scalar("norm2", loss_norm2, epoch)
        writer.add_scalar("norm", loss_norm, epoch)

    elif FLAGS.ntype == "sphere":
        normnet.train()
        optimizer_norm.zero_grad()
        norm = normnet(dataset)

        loss_norm1 = l1loss(norm, torch.tensor(n_mesh.fn_uv).float().to(device))
        loss_norm2 = 5.0 * Loss.fn_lap_loss(norm, n_mesh.f2f_mat)
        #loss_norm2 = 10.0 * Loss.fn_bilap_loss(norm, n_mesh.fc, n_mesh.f2f)
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
        loss_pos1 = Loss.rmse_loss(pos, n_mesh.vs)
        loss_pos2 = FLAGS.lap * Loss.mesh_laplacian_loss(pos, n_mesh)

        norm = normnet(dataset)
        loss_norm1 = Loss.norm_cos_loss(norm, n_mesh.fn)
        loss_norm2 = 0.5 * Loss.fn_bnf_loss(norm, n_mesh)

        fn2 = Models.compute_fn(pos, n_mesh.faces).float()
        loss_pos3 = Loss.norm_cos_loss(fn2, norm)
        
        loss_pos = loss_pos1 + loss_pos2 + loss_pos3
        loss_norm = loss_norm1 + loss_norm2
        loss_pos.backward(retain_graph=True)
        loss_norm.backward()
        nn.utils.clip_grad_norm_(normnet.parameters(), 1.0)
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
        elif FLAGS.ntype == "sphere":
            print('Epoch %d || Loss_N: %.4f' % (epoch, loss_norm.item()))
        else:
            print('Epoch %d || Loss_P: %.4f | Loss_N: %.4f' % (epoch, loss_pos.item(), loss_norm.item()))
        
    if epoch % 50 == 0:
        if FLAGS.ntype == "pos":
            o1_mesh.vs = pos.to('cpu').detach().numpy().copy()
            Mesh.compute_face_normals(o1_mesh)
            Mesh.compute_vert_normals(o1_mesh)
            mad_value = Loss.mad(o1_mesh.fn, gt_mesh.fn)
            min_mad = min(mad_value, min_mad)
            print("mad_value: ", mad_value, "min_mad: ", min_mad)
            writer.add_scalar("MAD", mad_value, epoch)
            Mesh.save(o1_mesh, "datasets/" + mesh_name + "/output/" + str(epoch) + "_pos.obj")

        elif FLAGS.ntype == "norm":
            test_norm_loss = Loss.rmse_loss(norm, gt_mesh.fn)
            min_norm_loss = min(min_norm_loss, test_norm_loss)
            print("test_norm_loss: ", float(test_norm_loss), "min_norm_loss: ", float(min_norm_loss))
            writer.add_scalar("test_norm", test_norm_loss, epoch)

        elif FLAGS.ntype == "sphere":
            normal = Models.uv2xyz(norm)
            test_norm_loss = Loss.rmse_loss(normal, gt_mesh.fn)
            min_norm_loss = min(min_norm_loss, test_norm_loss)
            print("test_norm_loss: ", float(test_norm_loss), "min_norm_loss: ", float(min_norm_loss))
            writer.add_scalar("test_norm", test_norm_loss, epoch)

        else:
            o1_mesh.vs = pos.to('cpu').detach().numpy().copy()
            Mesh.compute_face_normals(o1_mesh)
            Mesh.compute_vert_normals(o1_mesh)
            """ DMP-Pos """
            mad_value = Loss.mad(o1_mesh.fn, gt_mesh.fn)
            min_mad = min(mad_value, min_mad)
            print("mad_value: ", mad_value, "min_mad: ", min_mad)
            writer.add_scalar("MAD", mad_value, epoch)
            Mesh.save(o1_mesh, "datasets/" + mesh_name + "/output/" + str(epoch) + "_hybrid.obj")
            
            """ DMP-Norm """
            test_norm_loss = Loss.rmse_loss(norm, gt_mesh.fn)
            min_norm_loss = min(min_norm_loss, test_norm_loss)
            print("test_norm_loss: ", float(test_norm_loss), "min_norm_loss: ", float(min_norm_loss))
            writer.add_scalar("test_norm", test_norm_loss, epoch)