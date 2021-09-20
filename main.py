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
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--iter', type=int, default=5000)
parser.add_argument('--pos_lambda', type=float, default=1.4)
parser.add_argument('--norm_lambda', type=float, default=0.5)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--ntype', type=str, default='hybrid')
FLAGS = parser.parse_args()

for k, v in vars(FLAGS).items():
    print('{:10s}: {}'.format(k, v))

""" --- create dataset --- """
mesh_dic, dataset = Datamaker.create_dataset(FLAGS.input)
gt_file, n_file, s_file, mesh_name = mesh_dic["gt_file"], mesh_dic["n_file"], mesh_dic["s_file"], mesh_dic["mesh_name"]
gt_mesh, n_mesh, o1_mesh, s_mesh = mesh_dic["gt_mesh"], mesh_dic["n_mesh"], mesh_dic["o1_mesh"], mesh_dic["s_mesh"]
_, bnf_mesh = Loss.bnf(n_mesh.fn, n_mesh, sigma_s=0.7, iter=10)
_, e_str = Models.compute_nvt(bnf_mesh)
mask = e_str[:, 2]
mask_inv = 1.0 - mask

dt_now = datetime.datetime.now()

""" --- hyper parameters --- """
wandb.init(project="dmp-adv", group=mesh_name, job_type=FLAGS.ntype, name=dt_now.isoformat(),
           config={
               "pos_lr": 0.01,
               "norm_lr": 0.001,
               "grad_crip": 0.8,
               "pos_lambda": FLAGS.pos_lambda,
               "norm_lambda": FLAGS.norm_lambda
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
optimizer_norm = torch.optim.Adam(normnet.parameters(), lr=config.norm_lr, amsgrad=True)
scheduler_pos = torch.optim.lr_scheduler.StepLR(optimizer_pos, step_size=500, gamma=0.8)
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

""" --- initial condition --- """
min_mad = 1000
min_rmse_norm = 1000
min_rmse_pos = 1000
init_mad = Loss.mad(n_mesh.fn, gt_mesh.fn)
init_vn_loss = Loss.rmse_loss(n_mesh.vn, gt_mesh.vn)
init_fn_loss = Loss.rmse_loss(n_mesh.fn, gt_mesh.fn)
past_norm2 = 1000 #TODO: Remove this!
print("init_mad: ", init_mad, " init_vn_loss: ", float(init_vn_loss), " init_fn_loss: ", float(init_fn_loss))

""" --- learning loop --- """
for epoch in range(1, FLAGS.iter+1):
    if FLAGS.ntype == "pos":
        posnet.train()
        optimizer_pos.zero_grad()
        pos = posnet(dataset)
        loss_pos1 = Loss.rmse_loss(pos, n_mesh.vs)
        loss_pos2 = config.pos_lambda * Loss.mesh_laplacian_loss(pos, n_mesh)
        o1_mesh.vs = pos.to('cpu').detach().numpy().copy()
        fn2 = Models.compute_fn(pos, n_mesh.faces).float()
        
        loss_pos = loss_pos1 + loss_pos2
        loss_pos.backward()
        optimizer_pos.step()
        writer.add_scalar("pos1", loss_pos1, epoch)
        writer.add_scalar("pos2", loss_pos2, epoch)
        writer.add_scalar("pos", loss_pos, epoch)
        wandb.log({"pos": loss_pos, "pos1": loss_pos1, "pos2": loss_pos2})
    
    elif FLAGS.ntype == "norm":
        normnet.train()
        norm = normnet(dataset)

        loss_norm1 = Loss.norm_cos_loss(norm, n_mesh.fn)
        loss_norm2, new_fn = Loss.fn_bnf_loss(norm, n_mesh)
        loss_norm2 = config.norm_lambda * loss_norm2
        #loss_norm2 = config.norm_lambda * Loss.test_loss(norm, gt_mesh.fn)
        
        #TODO: Remove this!
        """
        if loss_norm2 > 1.1 * past_norm2:
            print("norm2 increased!")
            import pdb;pdb.set_trace()
            loss2 = Loss.fn_bnf_loss(norm, n_mesh)
        past_norm2 = loss_norm2
        """

        loss_norm = loss_norm1 + loss_norm2
        optimizer_norm.zero_grad()
        loss_norm.backward()

        nn.utils.clip_grad_norm_(normnet.parameters(), config.grad_crip)
        optimizer_norm.step()
        scheduler_norm.step()

        writer.add_scalar("norm1", loss_norm1, epoch)
        writer.add_scalar("norm2", loss_norm2, epoch)
        writer.add_scalar("norm", loss_norm, epoch)
        wandb.log({"norm": loss_norm, "norm1": loss_norm1, "norm2": loss_norm2})

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
        loss_pos2 = config.pos_lambda * Loss.mesh_laplacian_loss(pos, n_mesh)

        norm = normnet(dataset)
        loss_norm1 = Loss.norm_cos_loss(norm, n_mesh.fn)
        loss_norm2 = config.norm_lambda * Loss.fn_bnf_loss(norm, n_mesh)

        fn2 = Models.compute_fn(pos, n_mesh.faces).float()
        #loss_pos3 = loss_norm3 = Loss.norm_cos_loss(fn2, norm)
        loss_pos3 = Loss.masked_norm_cos_loss(fn2, norm, mask)
        loss_norm3 = Loss.masked_norm_cos_loss(fn2, norm, mask_inv)
        
        loss_pos = loss_pos1 + loss_pos2 + loss_pos3
        loss_norm = loss_norm1 + loss_norm2 + loss_norm3
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
        writer.add_scalar("norm3", loss_norm3, epoch)
        writer.add_scalar("norm", loss_norm, epoch)
        wandb.log({"pos": loss_pos, "pos1": loss_pos1, "pos2": loss_pos2, "pos3": loss_pos3, "norm": loss_norm, "norm1": loss_norm1, "norm2": loss_norm2, "norm3": loss_norm3})

    if epoch % 10 == 0:
        if FLAGS.ntype == "pos":
            print('Epoch %d || Loss_P: %.4f' % (epoch, loss_pos.item()))
        elif FLAGS.ntype == "norm":
            print('Epoch %d || Loss_N: %.4f || lr: %.4f' % (epoch, loss_norm.item(), scheduler_norm.get_last_lr()[0]))
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
            test_rmse_pos = Loss.rmse_loss(pos, gt_mesh.vs)
            min_rmse_pos = min(min_rmse_pos, test_rmse_pos)
            writer.add_scalar("MAD", mad_value, epoch)
            wandb.log({"MAD": mad_value, "RMSE_pos": test_rmse_pos})
            Mesh.save(o1_mesh, "datasets/" + mesh_name + "/output/" + str(epoch) + "_pos.obj")
            print("mad_value: ", mad_value, "min_mad: ", min_mad)

        elif FLAGS.ntype == "norm":
            mad_value = Loss.mad(norm, gt_mesh.fn)
            min_mad = min(mad_value, min_mad)
            if epoch > 200 and mad_value > 10:
                import pdb;pdb.set_trace()
            test_rmse_norm = Loss.rmse_loss(norm, gt_mesh.fn)
            min_rmse_norm = min(min_rmse_norm, test_rmse_norm)
            writer.add_scalar("MAD", mad_value, epoch)
            writer.add_scalar("test_norm", test_rmse_norm, epoch)
            wandb.log({"MAD": mad_value, "RMSE_norm": test_rmse_norm})
            Mesh.save_as_ply(gt_mesh, "datasets/" + mesh_name + "/output/" + str(epoch) + "_norm.ply", norm.to("cpu").detach().numpy().copy())
            Mesh.save_as_ply(gt_mesh, "datasets/" + mesh_name + "/output/" + str(epoch) + "_norm_bnf.ply", new_fn.to("cpu").detach().numpy().copy())
            print("mad_value: ", mad_value, "min_mad: ", min_mad)
            print("test_rmse: ", float(test_rmse_norm), "min_rmse: ", float(min_rmse_norm))

        elif FLAGS.ntype == "sphere":
            normal = Models.uv2xyz(norm)
            test_rmse = Loss.rmse_loss(normal, gt_mesh.fn)
            min_rmse = min(min_rmse, test_rmse)
            print("test_rmse: ", float(test_rmse), "min_rmse: ", float(min_rmse))
            writer.add_scalar("test_norm", test_rmse, epoch)

        elif FLAGS.ntype == "hybrid":
            o1_mesh.vs = pos.to('cpu').detach().numpy().copy()
            Mesh.compute_face_normals(o1_mesh)
            Mesh.compute_vert_normals(o1_mesh)
            """ DMP-Pos """
            mad_value = Loss.mad(o1_mesh.fn, gt_mesh.fn)
            min_mad = min(mad_value, min_mad)
            writer.add_scalar("MAD", mad_value, epoch)
            Mesh.save(o1_mesh, "datasets/" + mesh_name + "/output/" + str(epoch) + "_hybrid.obj")
            print("mad_value: ", mad_value, "min_mad: ", min_mad)
            
            """ DMP-Norm """
            test_rmse_norm = Loss.rmse_loss(norm, gt_mesh.fn)
            min_rmse_norm = min(min_rmse_norm, test_rmse_norm)
            print("test_rmse: ", float(test_rmse_norm), "min_rmse: ", float(min_rmse_norm))
            writer.add_scalar("test_norm", test_rmse_norm, epoch)

            wandb.log({"MAD": mad_value, "RMSE_norm": test_rmse_norm})

    wandb.save(log_dir + "/model.h5")