import numpy as np
import torch
from util.models import Mesh

def mae_loss(pred_pos, real_pos, verts_mask=None):
    """mean-absolute error for vertex positions"""
    diff_pos = torch.abs(real_pos - pred_pos)
    diff_pos = torch.sum(diff_pos.squeeze(), dim=1)
    if verts_mask == None:
        mae_pos = torch.sum(diff_pos) / len(diff_pos)
    else:
        mae_pos = torch.sum(diff_pos.T * verts_mask) / (torch.sum(verts_mask) + 1.0e-12)
    return mae_pos

def mse_loss(pred_pos, real_pos, verts_mask=None):
    """mean-square error for vertex positions"""
    diff_pos = torch.abs(real_pos - pred_pos)
    diff_pos = diff_pos ** 2
    diff_pos = torch.sum(diff_pos.squeeze(), dim=1)
    diff_pos = torch.sqrt(diff_pos)
    if verts_mask == None:
        mse_pos = torch.sum(diff_pos) / len(diff_pos)
    else:
        mse_pos = torch.sum(diff_pos.T * verts_mask) / (torch.sum(verts_mask) + 1.0e-12)
    return mse_pos

def mae_loss_edge_lengths(pred_pos, real_pos, edges):
    """mean-absolute error for edge lengths"""
    pred_edge_pos = pred_pos[edges,:].clone().detach()
    real_edge_pos = real_pos[edges,:].clone().detach()

    pred_edge_lens = torch.abs(pred_edge_pos[:,0,:]-pred_edge_pos[:,1,:])
    real_edge_lens = torch.abs(real_edge_pos[:,0,:]-real_edge_pos[:,1,:])

    pred_edge_lens = torch.sum(pred_edge_lens, dim=1)
    real_edge_lens = torch.sum(real_edge_lens, dim=1)
    
    diff_edge_lens = torch.abs(real_edge_lens - pred_edge_lens)
    mae_edge_lens = torch.mean(diff_edge_lens)

    return mae_edge_lens

def var_edge_lengths(pred_pos, edges):
    """variance of edge lengths"""
    pred_edge_pos = pred_pos[edges,:].clone().detach()

    pred_edge_lens = torch.abs(pred_edge_pos[:,0,:]-pred_edge_pos[:,1,:])

    pred_edge_lens = torch.sum(pred_edge_lens, dim=1)
    
    mean_edge_len = torch.mean(pred_edge_lens, dim=0, keepdim=True)
    var_edge_len = torch.pow(pred_edge_lens - mean_edge_len, 2.0)
    var_edge_len = torch.mean(var_edge_len)

    return var_edge_len

def mesh_laplacian_loss(pred_pos, ve, edges):
    """simple laplacian for output meshes"""
    pred_pos = pred_pos.T
    sub_mesh_vv = [edges[v_e, :].reshape(-1) for v_e in ve]
    sub_mesh_vv = [set(vv.tolist()).difference(set([i])) for i, vv in enumerate(sub_mesh_vv)]

    num_verts = pred_pos.size(1)
    mat_rows = [np.array([i] * len(vv), dtype=np.int64) for i, vv in enumerate(sub_mesh_vv)]
    mat_rows = np.concatenate(mat_rows)
    mat_cols = [np.array(list(vv), dtype=np.int64) for vv in sub_mesh_vv]
    mat_cols = np.concatenate(mat_cols)

    mat_rows = torch.from_numpy(mat_rows).long().to(pred_pos.device)
    mat_cols = torch.from_numpy(mat_cols).long().to(pred_pos.device)
    mat_vals = torch.ones_like(mat_rows).float()
    neig_mat = torch.sparse.FloatTensor(torch.stack([mat_rows, mat_cols], dim=0),
                                        mat_vals,
                                        size=torch.Size([num_verts, num_verts]))
    pred_pos = pred_pos.T
    sum_neigs = torch.sparse.mm(neig_mat, pred_pos)
    sum_count = torch.sparse.mm(neig_mat, torch.ones((num_verts, 1)).type_as(pred_pos))
    nnz_mask = (sum_count != 0).squeeze()
    #lap_vals = sum_count[nnz_mask, :] * pred_pos[nnz_mask, :] - sum_neigs[nnz_mask, :]
    if len(torch.where(sum_count[:, 0]==0)[0]) == 0:
        lap_vals = pred_pos[nnz_mask, :] - sum_neigs[nnz_mask, :] / sum_count[nnz_mask, :]
    else:
        print("[ERROR] Isorated vertices exist")
        return False
    lap_vals = torch.sqrt(torch.sum(lap_vals * lap_vals, dim=1) + 1.0e-12)
    #lap_vals = torch.sum(lap_vals * lap_vals, dim=1)
    lap_loss = torch.sum(lap_vals) / torch.sum(nnz_mask)

    return lap_loss

def mad(mesh1, mesh2):
    fn1 = Mesh.compute_face_normals(mesh1)
    fn2 = Mesh.compute_face_normals(mesh2)
    inner = [np.inner(fn1[i], fn2[i]) for i in range(fn1.shape[0])]
    sad = np.rad2deg(np.arccos(np.clip(inner, -1.0, 1.0)))
    mad = np.sum(sad) / len(sad)

    return mad
