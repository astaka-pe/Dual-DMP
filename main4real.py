import torch
import torch.nn as nn
import argparse
import os
from tqdm import tqdm

import util.loss as Loss
import util.datamaker as Datamaker
from util.mesh import Mesh
from util.networks import PosNet, NormalNet

def get_parser():
    parser = argparse.ArgumentParser(description='Dual Deep Mesh Prior')
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('--pos_lr', type=float, default=0.01)
    parser.add_argument('--norm_lr', type=float, default=0.01)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--k1', type=float, default=3)
    parser.add_argument('--k2', type=float, default=0)
    parser.add_argument('--k3', type=float, default=3)
    parser.add_argument('--k4', type=float, default=4)
    parser.add_argument('--k5', type=float, default=2)
    parser.add_argument('--grad_crip', type=float, default=0.8)
    parser.add_argument('--bnfloop', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    for k, v in vars(args).items():
        print('{:12s}: {}'.format(k, v))
    
    return args

def main():
    args = get_parser()

    """ --- create dataset --- """
    mesh_dic, dataset = Datamaker.create_dataset(args.input)
    mesh_name = mesh_dic["mesh_name"]
    n_mesh, o1_mesh = mesh_dic["n_mesh"], mesh_dic["o1_mesh"]

    """ --- create model instance --- """
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    posnet = PosNet(device).to(device)
    normnet = NormalNet(device).to(device)
    optimizer_pos = torch.optim.Adam(posnet.parameters(), lr=args.pos_lr)
    optimizer_norm = torch.optim.Adam(normnet.parameters(), lr=args.norm_lr)

    os.makedirs("datasets/" + mesh_name + "/output", exist_ok=True)

    """ --- learning loop --- """
    with tqdm(total=args.iter) as pbar:
        for epoch in range(1, args.iter+1):
            posnet.train()
            normnet.train()
            optimizer_pos.zero_grad()
            optimizer_norm.zero_grad()

            pos = posnet(dataset)
            loss_pos1 = Loss.pos_rec_loss(pos, n_mesh.vs)
            loss_pos2 = Loss.mesh_laplacian_loss(pos, n_mesh)

            norm = normnet(dataset)
            loss_norm1 = Loss.norm_rec_loss(norm, n_mesh.fn)
            loss_norm2, new_fn = Loss.fn_bnf_loss(pos, norm, n_mesh, loop=args.bnfloop)
            
            if epoch <= 100:
                loss_norm2 = loss_norm2 * 0.0

            loss_pos3 = Loss.pos_norm_loss(pos, norm, n_mesh)
            
            loss = args.k1 * loss_pos1 + args.k2 * loss_pos2 + args.k3 * loss_norm1 + args.k4 * loss_norm2 + args.k5 * loss_pos3
            loss.backward()
            nn.utils.clip_grad_norm_(normnet.parameters(), args.grad_crip)
            optimizer_pos.step()
            optimizer_norm.step()

            pbar.set_description("Epoch {}".format(epoch))
            pbar.set_postfix({"loss": loss.item()})

            if epoch % 10 == 0:
                o1_mesh.vs = pos.to('cpu').detach().numpy().copy()
                o_path = "datasets/" + mesh_name + "/output/" + str(epoch) + "_ddmp.obj"
                Mesh.save(o1_mesh, o_path)
            
            pbar.update(1)

if __name__ == "__main__":
    main()