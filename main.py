import numpy as np
import torch
import torch.nn as nn
import datetime
import argparse
import os
from tqdm import tqdm

import util.loss as Loss
import util.models as Models
import util.datamaker as Datamaker
from util.mesh import Mesh
from util.networks import PosNet, NormalNet

parser = argparse.ArgumentParser(description="Dual Deep Mesh Prior")
parser.add_argument("-i", "--input", type=str, required=True)
parser.add_argument("--pos_lr", type=float, default=0.01)
parser.add_argument("--norm_lr", type=float, default=0.01)
parser.add_argument("--norm_optim", type=str, default="Adam")
parser.add_argument("--iter", type=int, default=1000)
parser.add_argument("--k1", type=float, default=3.0)
parser.add_argument("--k2", type=float, default=4.0)
parser.add_argument("--k3", type=float, default=4.0)
parser.add_argument("--k4", type=float, default=4.0)
parser.add_argument("--k5", type=float, default=1.0)
parser.add_argument("--grad_crip", type=float, default=0.8)
parser.add_argument("--bnfloop", type=int, default=1)
parser.add_argument("--gpu", type=int, default=0)
FLAGS = parser.parse_args()

for k, v in vars(FLAGS).items():
    print("{:12s}: {}".format(k, v))

""" --- create dataset --- """
mesh_dic, dataset = Datamaker.create_dataset(FLAGS.input)
gt_file, n_file, s_file, mesh_name = mesh_dic["gt_file"], mesh_dic["n_file"], mesh_dic["s_file"], mesh_dic["mesh_name"]
gt_mesh, n_mesh, o1_mesh, s_mesh = mesh_dic["gt_mesh"], mesh_dic["n_mesh"], mesh_dic["o1_mesh"], mesh_dic["s_mesh"]
dt_now = datetime.datetime.now()

""" --- create model instance --- """
device = torch.device("cuda:" + str(FLAGS.gpu) if torch.cuda.is_available() else "cpu")
posnet = PosNet(device).to(device)
normnet = NormalNet(device).to(device)
optimizer_pos = torch.optim.Adam(posnet.parameters(), lr=FLAGS.pos_lr)
optimizer_norm = torch.optim.Adam(normnet.parameters(), lr=FLAGS.norm_lr)

os.makedirs("datasets/" + mesh_name + "/output", exist_ok=True)

""" --- initial condition --- """
init_mad = Loss.mad(n_mesh.fn, gt_mesh.fn)
print("initial_mad: {:.3f}".format(init_mad))

""" --- learning loop --- """
with tqdm(total=FLAGS.iter) as pbar:
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

        fn2 = Models.compute_fn(pos, n_mesh.faces).float()

        loss_pos3 = Loss.pos_norm_loss(pos, norm, n_mesh)
        
        loss = FLAGS.k1 * loss_pos1 + FLAGS.k2 * loss_pos2 + FLAGS.k3 * loss_norm1 + FLAGS.k4 * loss_norm2 + FLAGS.k5 * loss_pos3
        loss.backward()
        nn.utils.clip_grad_norm_(normnet.parameters(), FLAGS.grad_crip)
        optimizer_pos.step()
        optimizer_norm.step()

        pbar.set_description("Epoch {}".format(epoch))
        pbar.set_postfix({"loss": loss.item()})
            
        if epoch % 50 == 0:
            new_pos = pos.to("cpu").detach().numpy().copy()
            o1_mesh.vs = new_pos
            Mesh.compute_face_normals(o1_mesh)
            Mesh.compute_vert_normals(o1_mesh)

            mad_value = Loss.mad(o1_mesh.fn, gt_mesh.fn)
            o_path = "datasets/" + mesh_name + "/output/" + str(epoch) + "_ddmp={:.3f}.obj".format(mad_value)
            Mesh.save(o1_mesh, o_path)

        pbar.update(1)

print("final_mad: {:.3f}".format(mad_value))