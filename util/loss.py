import numpy as np
import torch
import torch.nn.functional as F
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

def norm_cos_loss(pred_norm, real_norm, verts_mask=None):
    lap_cos = 1.0 - torch.sum(torch.mul(pred_norm, real_norm), dim=1)
    lap_loss = torch.sum(lap_cos, dim=0) / len(lap_cos)
    return lap_loss

def norm_laplacian_loss(pred_pos, ve, edges):
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
        appr_norm = sum_neigs[nnz_mask, :] / sum_count[nnz_mask, :]
        appr_norm = F.normalize(appr_norm, p=2.0, dim=1)
        lap_cos = 1.0 - torch.sum(torch.mul(pred_pos, appr_norm), dim=1)
    else:
        print("[ERROR] Isorated vertices exist")
        return False
    lap_loss = torch.sum(lap_cos, dim=0) / len(lap_cos)

    return lap_loss

def fn_lap_loss(fn: torch.Tensor, f2f_mat: torch.sparse.Tensor) -> torch.Tensor:
    dif_fn = torch.sparse.mm(f2f_mat.to(fn.device), fn)
    dif_fn = torch.sqrt(torch.sum(dif_fn ** 2, dim=1))
    fn_lap_loss = torch.sum(dif_fn) / len(dif_fn)
    """
    #f2f = torch.from_numpy(f2f).long()
    n_fn = torch.sum(fn[f2f], dim=1) / 3.0
    n_fn = n_fn / torch.norm(n_fn, dim=1).reshape(-1, 1)
    dif_cos = 1.0 - torch.sum(torch.mul(fn, n_fn), dim=1)
    fn_lap_loss = torch.sum(dif_cos, dim=0) / len(dif_cos)
    """
    return fn_lap_loss

def mad(mesh1, mesh2):
    Mesh.compute_face_normals(mesh1)
    Mesh.compute_face_normals(mesh2)
    inner = [np.inner(mesh1.fn[i], mesh2.fn[i]) for i in range(mesh1.fn.shape[0])]
    sad = np.rad2deg(np.arccos(np.clip(inner, -1.0, 1.0)))
    mad = np.sum(sad) / len(sad)

    return mad

def pos4norm(vs, o_mesh, fn):
    #vs = o_mesh.vs
    vf = o_mesh.vf
    fa = torch.tensor(o_mesh.fa).float()
    faces = torch.tensor(o_mesh.faces).long()
    loss = 0.0

    for i, f in enumerate(vf):
        fa_list = fa[list(f)]
        fn_list = fn[list(f)]
        c = torch.sum(vs[faces[list(f)]], dim=1) / 3.0
        x = torch.reshape(vs[i].repeat(len(f)), (-1, 3))
        dot = torch.sum(fn_list * (c - x), 1)
        dot = torch.reshape(dot.repeat(3), (3, -1)).T
        a = torch.reshape(fa_list.repeat(3), (3, -1)).T
        error = torch.sum(a * fn_list * dot, dim=0) / torch.sum(fa_list)
        loss += torch.norm(error)
    loss /= len(vs)
    """
    for i, f in enumerate(vf):
        fa_list = fa[list(f)]
        fn_list = fn[list(f)]
        c = np.sum(vs[faces[list(f)]], 1) / 3.0
        x = np.tile(vs[i], len(f)).reshape(-1, 3)
        dot = np.sum(fn_list * (c - x), 1)
        dot = np.tile(dot, 3).reshape(3, -1).T
        a = np.tile(fa_list, 3).reshape(3, -1).T
        error = np.sum(a * fn_list * dot, 0) / np.sum(fa_list)
        loss += np.linalg.norm(error)
    loss /= len(vs)
    """

    return loss