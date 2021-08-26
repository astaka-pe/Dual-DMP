from util.models import Mesh
import util.loss as Loss
import torch

n_mesh = Mesh("datasets/sharp/sharp_noise.obj")
g_mesh = Mesh("datasets/sharp/sharp_gt.obj")
d_mesh = Mesh("datasets/sharp/sharp_dmp.obj")

n1_mesh = Mesh("datasets/dragon/dragon_noise.obj")
g1_mesh = Mesh("datasets/dragon/dragon_gt.obj")
d1_mesh = Mesh("datasets/dragon/dragon_lap.obj")

n2_mesh = Mesh("datasets/grayloc/grayloc_noise.obj")
g2_mesh = Mesh("datasets/grayloc/grayloc_gt.obj")

loss = Loss.fn_bnf_loss(torch.from_numpy(n_mesh.fn).float(), n_mesh.fc, n_mesh.fa, n_mesh.f2f)
import pdb;pdb.set_trace()
"""
loss1 = Loss.fn_lap_loss_with_fa(torch.tensor(n_mesh.fn), n_mesh.f2f_mat_ext)
loss2 = Loss.fn_lap_loss_with_fa(torch.tensor(g_mesh.fn), n_mesh.f2f_mat_ext)
loss3 = Loss.fn_lap_loss_with_fa(torch.tensor(d_mesh.fn), n_mesh.f2f_mat_ext)
"""