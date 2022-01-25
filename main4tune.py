import numpy as np
import torch
import torch.nn as nn
import copy
import datetime
import os
import sys
import random
import glob
import argparse
import json
import wandb
import util.loss as Loss
import util.models as Models
import util.datamaker as Datamaker
import pymeshlab as ml
from util.objmesh import ObjMesh
from util.datamaker import Dataset
from util.mesh import Mesh
from util.networks import PosNet, NormalNet, LightNormalNet, BigNormalNet

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data

def set_random_seed(seed=12345):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return

parser = argparse.ArgumentParser(description='DMP_adv for mesh')
parser.add_argument('-i', '--input', type=str, required=True)
parser.add_argument('--pos_lr', type=float, default=0.01)
parser.add_argument('--norm_lr', type=float, default=0.01)
parser.add_argument('--iter', type=int, default=1000)
parser.add_argument('--k1', type=float, default=3)
parser.add_argument('--k2', type=float, default=4)
parser.add_argument('--k3', type=float, default=4)
parser.add_argument('--k4', type=float, default=4)
parser.add_argument('--k5', type=float, default=2)
parser.add_argument('--grad_crip', type=float, default=0.8)
parser.add_argument('--bnfloop', type=int, default=5)
parser.add_argument('--gpu', type=int, default=0)
FLAGS = parser.parse_args()

for k, v in vars(FLAGS).items():
    print('{:12s}: {}'.format(k, v))

""" --- create dataset --- """
mesh_dic, dataset = Datamaker.create_dataset(FLAGS.input)
g_file, n_file, s_file, mesh_name = mesh_dic["gt_file"], mesh_dic["n_file"], mesh_dic["s_file"], mesh_dic["mesh_name"]
g_mesh, n_mesh, o1_mesh, s_mesh = mesh_dic["gt_mesh"], mesh_dic["n_mesh"], mesh_dic["o1_mesh"], mesh_dic["s_mesh"]
dt_now = datetime.datetime.now()

""" --- create model instance --- """
device = torch.device('cuda:' + str(FLAGS.gpu) if torch.cuda.is_available() else 'cpu')
set_random_seed()
posnet = PosNet(device).to(device)
set_random_seed()
normnet = NormalNet(device).to(device)
optimizer_pos = torch.optim.Adam(posnet.parameters(), lr=FLAGS.pos_lr)
optimizer_norm = torch.optim.Adam(normnet.parameters(), lr=FLAGS.norm_lr)

""" --- output experimental conditions --- """
log_dir = "./logs/" + mesh_name + dt_now.isoformat()
writer = SummaryWriter(log_dir=log_dir)
log_file = log_dir + "/condition.json"
condition = {"PosNet": str(posnet).split("\n"), "NormNet": str(normnet).split("\n"), "optimizer_pos": str(optimizer_pos).split("\n"), "optimizer_norm": str(optimizer_norm).split("\n")}

with open(log_file, mode="w") as f:
    l = json.dumps(condition, indent=2)
    f.write(l)

os.makedirs("datasets/" + mesh_name + "/tune", exist_ok=True)

""" parameter settings """
para_grid = np.arange(0.0, 5.5, 0.5)
default_para = np.array([FLAGS.k1, FLAGS.k2, FLAGS.k3, FLAGS.k4, FLAGS.k5]).astype(np.float)
para_list = np.tile(default_para, (5, 11, 1))
para_list[0, :, 0] = para_list[1, :, 1] = para_list[2, :, 2] = para_list[3, :, 3] = para_list[4, :, 4] = para_grid
mad_list = np.zeros([5, 11]).astype(np.float)

for i, para_i in enumerate(para_list):
    for j, para in enumerate(para_i):
        """ --- learning loop --- """
        for epoch in range(1, FLAGS.iter+1):
            posnet.train()
            normnet.train()
            optimizer_pos.zero_grad()
            optimizer_norm.zero_grad()

            pos = posnet(dataset)
            loss_pos1 = Loss.pos_rec_loss(pos, n_mesh.vs)
            loss_pos2 = Loss.mesh_laplacian_loss(pos, n_mesh)

            norm = normnet(dataset)
            loss_norm1 = Loss.norm_rec_loss(norm, n_mesh.fn)
            loss_norm2, new_fn = Loss.fn_bnf_loss(pos, norm, n_mesh, loop=FLAGS.bnfloop)
            
            if epoch <= 100:
                loss_norm2 = loss_norm2 * 0.0


            loss_pos3 = Loss.pos_norm_loss(pos, norm, n_mesh)
            
            loss = para[0] * loss_pos1 + para[1] * loss_pos2 + para[2] * loss_norm1 + para[3] * loss_norm2 + para[4] * loss_pos3
            loss.backward()
            nn.utils.clip_grad_norm_(normnet.parameters(), FLAGS.grad_crip)
            optimizer_pos.step()
            optimizer_norm.step()

            writer.add_scalar("pos1", loss_pos1, epoch)
            writer.add_scalar("pos2", loss_pos2, epoch)
            writer.add_scalar("pos3", loss_pos3, epoch)
            writer.add_scalar("norm1", loss_norm1, epoch)
            writer.add_scalar("norm2", loss_norm2, epoch)

        o1_mesh.vs = pos.to('cpu').detach().numpy().copy()
        Mesh.compute_face_normals(o1_mesh)
        mad = Loss.mad(o1_mesh.fn, g_mesh.fn)
        mad = np.round(mad, decimals=4)
        mad_list[i, j] = mad
        para_str = "-".join(map(str, (para*10).astype(np.long).tolist()))
        o_path = "datasets/" + mesh_name + "/tune/" + para_str + "_" + str(mad) + ".obj"
        Mesh.save(o1_mesh, o_path)
        print(para_str, mad)
    print(mad_list[i])
print(mad_list)