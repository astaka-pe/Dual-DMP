import numpy as np
import torch
import torch.nn.functional as F
from util.mesh import Mesh
from typing import Union

def rmse_loss(pred_pos: Union[torch.Tensor, np.ndarray], real_pos: np.ndarray) -> torch.Tensor:
    """ root mean-square error for vertex positions """
    if type(pred_pos) == np.ndarray:
        pred_pos = torch.from_numpy(pred_pos)
    real_pos = torch.from_numpy(real_pos).to(pred_pos.device)
    diff_pos = torch.abs(real_pos - pred_pos)
    diff_pos = diff_pos ** 2
    diff_pos = torch.sum(diff_pos.squeeze(), dim=1)
    mse_pos = torch.sum(diff_pos) / len(diff_pos)
    rmse_pos = torch.sqrt(mse_pos + 1.0e-6)

    return rmse_pos

def norm_cos_loss(pred_norm: Union[torch.Tensor, np.ndarray], real_norm: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """ cosine distance for (vertex, face) normal """
    if type(pred_norm) == np.ndarray:
        pred_norm = torch.from_numpy(pred_norm)
    if type(real_norm) == np.ndarray:
        real_norm = torch.from_numpy(real_norm).to(pred_norm.device)
    lap_cos = 1.0 - torch.sum(torch.mul(pred_norm, real_norm), dim=1)
    lap_loss = torch.sum(lap_cos, dim=0) / len(lap_cos)
    return lap_loss

def fn_bnf_loss(fn: torch.Tensor, mesh: Mesh) -> torch.Tensor:
    """ bilateral loss for face normal """
    fc = torch.from_numpy(mesh.fc).float().to(fn.device)
    fa = torch.from_numpy(mesh.fa).float().to(fn.device)
    f2f = torch.from_numpy(mesh.f2f).long().to(fn.device)
    
    neig_fc = fc[f2f]
    neig_fa = fa[f2f]
    fc0_tile = fc.repeat(1, 3).reshape(-1, 3, 3)
    fc_dist = torch.norm(neig_fc - fc0_tile + 1.0e-6, dim=2)
    sigma_c = torch.sum(fc_dist) / (fc_dist.shape[0] * fc_dist.shape[1])

    new_fn = fn
    for i in range(5):
        neig_fn = new_fn[f2f]
        fn0_tile = new_fn.repeat(1, 3).reshape(-1, 3, 3)
        fn_dist = torch.norm(neig_fn - fn0_tile + 1.0e-6, dim=2)
        sigma_s = 0.3
        wc = torch.exp(-1.0 * (fc_dist ** 2) / (2 * (sigma_c ** 2) + 1.0e-6))
        ws = torch.exp(-1.0 * (fn_dist ** 2) / (2 * (sigma_s ** 2) + 1.0e-6))
        
        W = torch.stack([wc*ws*neig_fa, wc*ws*neig_fa, wc*ws*neig_fa], dim=2)

        new_fn = torch.sum(W * neig_fn, dim=1)
        new_fn = new_fn / (torch.norm(new_fn + 1.0e-6, dim=1, keepdim=True) + 1.0e-6)

    dif_fn = new_fn - fn
    dif_fn = dif_fn ** 2
    loss = torch.sum(dif_fn, dim=1)
    loss = torch.sum(loss, dim=0) / fn.shape[0]
    loss = torch.sqrt(loss + 1.0e-6)

    return loss

def mesh_laplacian_loss(pred_pos: torch.Tensor, mesh: Mesh) -> torch.Tensor:
    """ simple laplacian for output meshes """
    ve = mesh.ve
    edges = mesh.edges
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

def mad(norm1: Union[np.ndarray, torch.Tensor], norm2: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """ mean angular distance for (face, vertex) normals """
    if type(norm1) == torch.Tensor:
        norm1 = norm1.to("cpu").detach().numpy().copy()
    if type(norm2) == torch.Tensor:
        norm2 = norm2.to("cpu").detach().numpy().copy()

    inner = [np.inner(norm1[i], norm2[i]) for i in range(norm1.shape[0])]
    sad = np.rad2deg(np.arccos(np.clip(inner, -1.0, 1.0)))
    mad = np.sum(sad) / len(sad)

    return mad

""" --- We don't use the loss functions below --- """

def mae_loss(pred_pos: Union[torch.Tensor, np.ndarray], real_pos: np.ndarray) -> torch.Tensor:
    """mean-absolute error for vertex positions"""
    if type(pred_pos) == np.ndarray:
        pred_pos = torch.from_numpy(pred_pos)
    diff_pos = torch.abs(real_pos - pred_pos)
    diff_pos = torch.sum(diff_pos.squeeze(), dim=1)
    mae_pos = torch.sum(diff_pos) / len(diff_pos)

    return mae_pos

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

def fn_mean_filter_loss(fn: torch.Tensor, f2f_mat: torch.sparse.Tensor) -> torch.Tensor:
    f2f_mat = f2f_mat.to(fn.device)
    neig_fn = fn
    for i in range(10):
        neig_fn = torch.sparse.mm(f2f_mat, neig_fn)
        neig_fn = neig_fn / torch.norm(neig_fn, dim=1, keepdim=True)
    fn_cos = 1.0 - torch.sum(neig_fn * fn, dim=1)

    fn_mean_filter_loss = torch.sum(fn_cos, dim=0) / len(fn_cos)
    
    return fn_mean_filter_loss

def sphere_lap_loss_with_fa(uv: torch.Tensor, f2f_ext: torch.sparse.Tensor) -> torch.Tensor:
    neig_uv = torch.sparse.mm(f2f_ext.to(uv.device), uv.float())
    dif_uv = uv - neig_uv
    dif_uv = torch.norm(dif_uv, dim=1)
    fn_lap_loss = torch.sum(dif_uv, dim=0) / len(dif_uv)
    
    return fn_lap_loss

def fn_bilap_loss(fn: torch.Tensor, fc: np.ndarray, f2f: np.ndarray) -> torch.Tensor:
    fc = torch.from_numpy(fc).float().to(fn.device)
    f2f = torch.from_numpy(f2f).long().to(fn.device)
    
    neig_fc = fc[f2f]
    fc0_tile = fc.repeat(1, 3).reshape(-1, 3, 3)
    fc_dist = torch.norm(neig_fc - fc0_tile, dim=2)
    
    neig_fn = fn[f2f]
    fn0_tile = fn.repeat(1, 3).reshape(-1, 3, 3)
    fn_dist = 1.0 - torch.sum(neig_fn * fn0_tile, dim=2)
   
    sigma_c, _ = torch.max(fc_dist, dim=1)
    sigma_c = sigma_c.reshape(-1, 1)
    sigma_s = torch.std(fn_dist, dim=1).reshape(-1, 1)

    wc = torch.exp(-1.0 * (fc_dist ** 2) / (2 * (sigma_c ** 2) + 1.0e-12))
    ws = torch.exp(-1.0 * (fn_dist ** 2) / (2 * (sigma_s ** 2) + 1.0e-12))

    loss = torch.sum(wc * ws * fn_dist, dim=1) / (torch.sum(wc * ws, dim=1) + 1.0e-12)
    loss = torch.sum(loss) / len(loss)
    
    return loss

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