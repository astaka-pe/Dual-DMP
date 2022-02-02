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
import warnings
from util.objmesh import ObjMesh
from util.datamaker import Dataset
from util.mesh import Mesh
from util.networks import PosNet, NormalNet, LightNormalNet, BigNormalNet

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data

warnings.simplefilter("ignore")

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
parser.add_argument('--norm_optim', type=str, default='Adam')
parser.add_argument('--iter', type=int, default=1000)
parser.add_argument('--k1', type=float, default=1.0)
parser.add_argument('--k2', type=float, default=1.4)
parser.add_argument('--k3', type=float, default=1.0)
parser.add_argument('--k4', type=float, default=0.5)
parser.add_argument('--k5', type=float, default=1.0)
parser.add_argument('--grad_crip', type=float, default=0.8)
parser.add_argument('--bnfloop', type=int, default=5)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--ntype', type=str, default='hybrid')
FLAGS = parser.parse_args()

for k, v in vars(FLAGS).items():
    print('{:12s}: {}'.format(k, v))

""" --- create dataset --- """
mesh_dic, dataset = Datamaker.create_dataset(FLAGS.input)
gt_file, n_file, s_file, mesh_name = mesh_dic["gt_file"], mesh_dic["n_file"], mesh_dic["s_file"], mesh_dic["mesh_name"]
gt_mesh, n_mesh, o1_mesh, s_mesh = mesh_dic["gt_mesh"], mesh_dic["n_mesh"], mesh_dic["o1_mesh"], mesh_dic["s_mesh"]
dt_now = datetime.datetime.now()

""" --- hyper parameters --- """
wandb.init(project="dmp-adv", group=mesh_name, job_type=FLAGS.ntype, name=dt_now.isoformat(),
           config={
               "pos_lr": FLAGS.pos_lr,
               "norm_lr": FLAGS.norm_lr,
               "grad_crip": FLAGS.grad_crip,
               "k1":FLAGS.k1,
               "k2":FLAGS.k2,
               "k3":FLAGS.k3,
               "k4":FLAGS.k4,
               "k5":FLAGS.k5,
               "norm_optim": FLAGS.norm_optim,
           })
config = wandb.config

""" --- create model instance --- """
device = torch.device('cuda:' + str(FLAGS.gpu) if torch.cuda.is_available() else 'cpu')
set_random_seed()
posnet = PosNet(device).to(device)
set_random_seed()
normnet = NormalNet(device).to(device)
#normnet = BigNormalNet(device).to(device)
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

""" --- output experimental conditions --- """
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
min_dfrm = 1000
min_rmse_norm = 1000
min_rmse_pos = 1000
init_mad = Loss.mad(n_mesh.fn, gt_mesh.fn)
init_vn_loss = Loss.pos_rec_loss(n_mesh.vn, gt_mesh.vn)
init_fn_loss = Loss.pos_rec_loss(n_mesh.fn, gt_mesh.fn)
print("init_mad: ", init_mad, " init_vn_loss: ", float(init_vn_loss), " init_fn_loss: ", float(init_fn_loss))

""" --- learning loop --- """
for epoch in range(1, FLAGS.iter+1):
    if FLAGS.ntype == "pos":
        posnet.train()
        optimizer_pos.zero_grad()
        pos = posnet(dataset)
        loss_pos1 = Loss.pos_rec_loss(pos, n_mesh.vs)
        loss_pos2 = Loss.mesh_laplacian_loss(pos, n_mesh)
        o1_mesh.vs = pos.to('cpu').detach().numpy().copy()
        fn2 = Models.compute_fn(pos, n_mesh.faces).float()
        
        loss_pos = FLAGS.k1 * loss_pos1 + FLAGS.k2 * loss_pos2
        loss_pos.backward()
        optimizer_pos.step()
        writer.add_scalar("pos1", loss_pos1, epoch)
        writer.add_scalar("pos2", loss_pos2, epoch)
        writer.add_scalar("pos", loss_pos, epoch)
        wandb.log({"pos": loss_pos, "pos1": loss_pos1, "pos2": loss_pos2})
    
    elif FLAGS.ntype == "norm":
        normnet.train()
        norm = normnet(dataset)

        loss_norm1 = Loss.norm_rec_loss(norm, n_mesh.fn, ltype="rmse")
        loss_norm2, new_fn = Loss.fn_bnf_loss(norm, n_mesh)

        loss_norm = FLAGS.k3 * loss_norm1 + FLAGS.k4 * loss_norm2
        optimizer_norm.zero_grad()
        loss_norm.backward()

        nn.utils.clip_grad_norm_(normnet.parameters(), config.grad_crip)
        optimizer_norm.step()
        scheduler_norm.step()

        writer.add_scalar("norm1", loss_norm1, epoch)
        writer.add_scalar("norm2", loss_norm2, epoch)
        writer.add_scalar("norm", loss_norm, epoch)
        wandb.log({"norm": loss_norm, "norm1": loss_norm1, "norm2": loss_norm2})
    
    elif FLAGS.ntype == "hybrid":
        posnet.train()
        normnet.train()
        optimizer_pos.zero_grad()
        optimizer_norm.zero_grad()

        pos = posnet(dataset)
        loss_pos1 = Loss.pos_rec_loss(pos, n_mesh.vs)
        loss_pos2 = Loss.mesh_laplacian_loss(pos, n_mesh)

        norm = normnet(dataset)
        loss_norm1 = Loss.norm_rec_loss(norm, n_mesh.fn)
        """ for full-pipeline """
        loss_norm2, new_fn = Loss.fn_bnf_loss(pos, norm, n_mesh, loop=FLAGS.bnfloop)
        """ for ablation study """
        #loss_norm2, new_fn = Loss.fn_bnf_loss(n_mesh.vs, norm, n_mesh, loop=FLAGS.bnfloop)
        
        if epoch <= 100:
            loss_norm2 = loss_norm2 * 0.0

        fn2 = Models.compute_fn(pos, n_mesh.faces).float()

        loss_pos3 = Loss.pos_norm_loss(pos, norm, n_mesh)
        #loss_pos3 = Loss.norm_rec_loss(norm, fn2)
        
        loss = FLAGS.k1 * loss_pos1 + FLAGS.k2 * loss_pos2 + FLAGS.k3 * loss_norm1 + FLAGS.k4 * loss_norm2 + FLAGS.k5 * loss_pos3
        loss.backward()
        nn.utils.clip_grad_norm_(normnet.parameters(), config.grad_crip)
        optimizer_pos.step()
        optimizer_norm.step()
        scheduler_pos.step()
        scheduler_norm.step()

        writer.add_scalar("pos1", loss_pos1, epoch)
        writer.add_scalar("pos2", loss_pos2, epoch)
        writer.add_scalar("pos3", loss_pos3, epoch)
        writer.add_scalar("norm1", loss_norm1, epoch)
        writer.add_scalar("norm2", loss_norm2, epoch)
        wandb.log({"P_rec": loss_pos1, "P_lap": loss_pos2, "PN": loss_pos3, "N_rec": loss_norm1, "N_bnf": loss_norm2})

    if epoch % 10 == 0:
        if FLAGS.ntype == "pos":
            print('Epoch %d || Loss_P: %.4f' % (epoch, loss_pos.item()))
        elif FLAGS.ntype == "norm":
            print('Epoch %d || Loss_N: %.4f || lr: %.4f' % (epoch, loss_norm.item(), scheduler_norm.get_last_lr()[0]))
        elif FLAGS.ntype == "hybrid":
            print('Epoch %d || Loss_P: %.4f' % (epoch, loss))
        else:
            print("[ERROR]: ntype error")
            exit()
        
    if epoch % 50 == 0:
        if FLAGS.ntype == "pos":
            o1_mesh.vs = pos.to('cpu').detach().numpy().copy()
            Mesh.compute_face_normals(o1_mesh)
            Mesh.compute_vert_normals(o1_mesh)
            mad_value = Loss.mad(o1_mesh.fn, gt_mesh.fn)
            min_mad = min(mad_value, min_mad)
            test_rmse_pos = Loss.pos_rec_loss(pos, gt_mesh.vs)
            min_rmse_pos = min(min_rmse_pos, test_rmse_pos)
            writer.add_scalar("MAD", mad_value, epoch)
            wandb.log({"MAD": mad_value, "RMSE_pos": test_rmse_pos})
            o_path = "datasets/" + mesh_name + "/output/" + str(epoch) + "_pos.obj"
            Mesh.save(o1_mesh, o_path)
            ms = ml.MeshSet()
            ms.load_new_mesh(gt_file)
            ms.load_new_mesh(o_path)
            dfrm = Loss.distance_from_reference_mesh(ms)
            min_dfrm = min(min_dfrm, dfrm)
            print("mad_value: ", mad_value, "min_mad: ", min_mad)
            print("dfrm_mae : ", dfrm, "min_dfr: ", min_dfrm)

        elif FLAGS.ntype == "norm":
            mad_value = Loss.mad(norm, gt_mesh.fn)
            min_mad = min(mad_value, min_mad)
            test_rmse_norm = Loss.pos_rec_loss(norm, gt_mesh.fn)
            min_rmse_norm = min(min_rmse_norm, test_rmse_norm)
            writer.add_scalar("MAD", mad_value, epoch)
            writer.add_scalar("test_norm", test_rmse_norm, epoch)
            wandb.log({"MAD": mad_value, "RMSE_norm": test_rmse_norm})
            Mesh.save_as_ply(gt_mesh, "datasets/" + mesh_name + "/output/" + str(epoch) + "_norm.ply", norm.to("cpu").detach().numpy().copy())
            Mesh.save_as_ply(gt_mesh, "datasets/" + mesh_name + "/output/" + str(epoch) + "_norm_bnf.ply", new_fn.to("cpu").detach().numpy().copy())
            print("mad_value: ", mad_value, "min_mad: ", min_mad)
            print("test_rmse: ", float(test_rmse_norm), "min_rmse: ", float(min_rmse_norm))

        elif FLAGS.ntype == "hybrid":
            new_pos = pos.to('cpu').detach().numpy().copy()
            o1_mesh.vs = new_pos
            Mesh.compute_face_normals(o1_mesh)
            Mesh.compute_vert_normals(o1_mesh)
            """ DMP-Pos """
            mad_value = Loss.mad(o1_mesh.fn, gt_mesh.fn)
            min_mad = min(mad_value, min_mad)
            writer.add_scalar("MAD", mad_value, epoch)
            o_path = "datasets/" + mesh_name + "/output/" + str(epoch) + "_hybrid.obj"
            Mesh.save(o1_mesh, o_path)
            ms = ml.MeshSet()
            ms.load_new_mesh(gt_file)
            ms.load_new_mesh(o_path)
            dfrm = Loss.distance_from_reference_mesh(ms)
            min_dfrm = min(min_dfrm, dfrm)
            print(" Pos_mad: {:.3f} min: {:.3f}".format(mad_value, min_mad))
            #print("dfrm_mae : ", dfrm, "min_dfr: ", min_dfrm)
            """ DMP-Norm """
            test_rmse_norm = Loss.pos_rec_loss(norm, gt_mesh.fn)
            min_rmse_norm = min(min_rmse_norm, test_rmse_norm)
            norm_mad = Loss.mad(norm, gt_mesh.fn)
            print("Norm_mad: {:.3f}".format(norm_mad))
            #print("test_rmse: ", float(test_rmse_norm), "min_rmse: ", float(min_rmse_norm))
            writer.add_scalar("test_norm", test_rmse_norm, epoch)
            wandb.log({"MAD": mad_value, "RMSE_norm": test_rmse_norm, "norm_mad": norm_mad})

        
        else:
            print("[ERROR]: ntype error")
            exit()

    wandb.save(log_dir + "/model.h5")