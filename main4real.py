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
parser.add_argument('--norm_lr', type=float, default=0.001)
parser.add_argument('--norm_optim', type=str, default='Adam')
parser.add_argument('--iter', type=int, default=5000)
parser.add_argument('--pos_lambda', type=float, default=1.4)
parser.add_argument('--norm_lambda', type=float, default=0.5)
parser.add_argument('--pn_lambda', type=float, default=1.0)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--ntype', type=str, default='hybrid')
FLAGS = parser.parse_args()

for k, v in vars(FLAGS).items():
    print('{:12s}: {}'.format(k, v))

""" --- create dataset --- """
mesh_dic, dataset = Datamaker.create_dataset(FLAGS.input)
_, n_file, s_file, mesh_name = mesh_dic["gt_file"], mesh_dic["n_file"], mesh_dic["s_file"], mesh_dic["mesh_name"]
_, n_mesh, o1_mesh, s_mesh = mesh_dic["gt_mesh"], mesh_dic["n_mesh"], mesh_dic["o1_mesh"], mesh_dic["s_mesh"]
dt_now = datetime.datetime.now()

""" --- hyper parameters --- """
wandb.init(project="dmp-adv", group=mesh_name, job_type=FLAGS.ntype, name=dt_now.isoformat(),
           config={
               "pos_lr": FLAGS.pos_lr,
               "norm_lr": FLAGS.norm_lr,
               "grad_crip": 0.8,
               "pos_lambda": FLAGS.pos_lambda,
               "norm_lambda": FLAGS.norm_lambda,
               "pn_lambda": FLAGS.pn_lambda,
               "norm_optim": FLAGS.norm_optim
           })
config = wandb.config

""" --- create model instance --- """
device = torch.device('cuda:' + str(FLAGS.gpu) if torch.cuda.is_available() else 'cpu')
set_random_seed()
posnet = PosNet(device).to(device)
#normnet = NormalNet(device).to(device)
set_random_seed()
normnet = BigNormalNet(device).to(device)
optimizer_pos = torch.optim.Adam(posnet.parameters(), lr=config.pos_lr)

norm_optimizers = {}
norm_optimizers["SGD"] = torch.optim.SGD(normnet.parameters(), lr=config.norm_lr)
norm_optimizers["Adam"] = torch.optim.Adam(normnet.parameters(), lr=config.norm_lr)
norm_optimizers["RMSprop"] = torch.optim.RMSprop(normnet.parameters(), lr=config.norm_lr)
norm_optimizers["Adadelta"] = torch.optim.Adadelta(normnet.parameters(), lr=config.norm_lr)
norm_optimizers["AdamW"] = torch.optim.AdamW(normnet.parameters(), lr=config.norm_lr)

optimizer_norm = norm_optimizers[FLAGS.norm_optim]
scheduler_pos = torch.optim.lr_scheduler.StepLR(optimizer_pos, step_size=500, gamma=1.0)
scheduler_norm = torch.optim.lr_scheduler.StepLR(optimizer_norm, step_size=500, gamma=1.0)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_norm, T_max=50)

""" --- output experimental conditions --- """
log_dir = "./logs/" + mesh_name + dt_now.isoformat()
writer = SummaryWriter(log_dir=log_dir)
log_file = log_dir + "/condition.json"
condition = {"PosNet": str(posnet).split("\n"), "NormNet": str(normnet).split("\n"), "optimizer_pos": str(optimizer_pos).split("\n"), "optimizer_norm": str(optimizer_norm).split("\n")}

with open(log_file, mode="w") as f:
    l = json.dumps(condition, indent=2)
    f.write(l)

os.makedirs("datasets/" + mesh_name + "/output", exist_ok=True)

""" --- learning loop --- """
for epoch in range(1, FLAGS.iter+1):
    posnet.train()
    normnet.train()
    optimizer_pos.zero_grad()
    optimizer_norm.zero_grad()

    pos = posnet(dataset)
    loss_pos1 = Loss.rmse_loss(pos, n_mesh.vs)
    loss_pos2 = config.pos_lambda * Loss.mesh_laplacian_loss(pos, n_mesh)

    norm = normnet(dataset)
    loss_norm1 = Loss.norm_cos_loss(norm, n_mesh.fn)
    loss_norm2, new_fn = Loss.fn_bnf_loss(norm, n_mesh)
    loss_norm2 = config.norm_lambda * loss_norm2

    fn2 = Models.compute_fn(pos, n_mesh.faces).float()

    loss_pos3 = config.pn_lambda * Loss.pos_norm_loss(pos, norm, n_mesh)
    
    loss_pos = loss_pos1 + loss_pos2 + loss_pos3
    loss_norm = loss_norm1 + loss_norm2
    loss_pos.backward(retain_graph=True)
    loss_norm.backward()
    nn.utils.clip_grad_norm_(normnet.parameters(), config.grad_crip)
    optimizer_pos.step()
    optimizer_norm.step()
    scheduler_pos.step()
    scheduler_norm.step()

    writer.add_scalar("pos1", loss_pos1, epoch)
    writer.add_scalar("pos2", loss_pos2, epoch)
    writer.add_scalar("pos3", loss_pos3, epoch)
    writer.add_scalar("pos", loss_pos, epoch)
    writer.add_scalar("norm1", loss_norm1, epoch)
    writer.add_scalar("norm2", loss_norm2, epoch)
    writer.add_scalar("norm", loss_norm, epoch)
    wandb.log({"pos": loss_pos, "pos1": loss_pos1, "pos2": loss_pos2, "pos3": loss_pos3, "norm": loss_norm, "norm1": loss_norm1, "norm2": loss_norm2})

    if epoch % 10 == 0:
        print('Epoch %d || Loss_P: %.4f | Loss_N: %.4f' % (epoch, loss_pos.item(), loss_norm.item()))
        
    if epoch % 50 == 0:
        o1_mesh.vs = pos.to('cpu').detach().numpy().copy()
        Mesh.compute_face_normals(o1_mesh)
        Mesh.compute_vert_normals(o1_mesh)
        Mesh.save(o1_mesh, "datasets/" + mesh_name + "/output/" + str(epoch) + "_hybrid.obj")

    wandb.save(log_dir + "/model.h5")