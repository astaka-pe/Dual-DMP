import numpy as np
import torch
from functools import reduce
from collections import Counter
import scipy as sp
from sklearn.preprocessing import normalize
from util.mesh import Mesh

def build_div(n_mesh, vn):
    vs = n_mesh.vs
    faces = n_mesh.faces
    fn = n_mesh.fn
    fa = n_mesh.fa
    vf = n_mesh.vf
    grad_b = [[] for _ in range(len(vs))]
    for i, v in enumerate(vf): # vf: triangle indices around vertex i
        for t in v: # for each triangle index t
            f = faces[t] # vertex indices in t
            f_n = fn[t]  # face normal of t
            a = fa[t]    # face area of t
            """sort vertex indices"""
            if f[1] == i:
                f = [f[1], f[2], f[0]]
            elif f[2] == i:
                f = [f[2], f[1], f[0]]
            x_kj = vs[f[2]] - vs[f[1]]
            x_kj = f_n * np.dot(x_kj, f_n) + np.cross(f_n, x_kj)
            x_kj /= 2
            grad_b[i].append(x_kj.tolist())

    div_v = [[] for _ in range(len(vn))]
    for i in range(len(vn)):
        tn = vn[i]
        g = np.array(grad_b[i])
        div_v[i] = np.dot(np.sum(g, 0), tn)
    div = np.array(div_v)
    div = np.tile(div, 3).reshape(3, -1).T
    return div

def jacobi(n_mesh, div, iter=10):
    # preparation
    C = n_mesh.mesh_lap.to_dense()
    #C = n_mesh.lapmat.to_dense()
    B = div
    # boundary condition
    C_add = torch.eye(len(n_mesh.vs))
    C = torch.cat([C, C_add], dim=0)
    B_add = torch.from_numpy(n_mesh.vs)
    B = torch.cat([torch.from_numpy(B), B_add], dim=0)
    B = torch.matmul(C.T.float(), B.float())
    A = torch.matmul(C.T, C)
    # solve Ax=b by jacobi
    x = torch.from_numpy(n_mesh.vs).float()

    for i in range(iter):
        r = B - torch.matmul(A, x)
        alpha = torch.diagonal(torch.matmul(r.T, r)) / (torch.diagonal(torch.matmul(torch.matmul(r.T, A), r)) + 1e-12)
        x += alpha * r
    
    return x

def poisson_mesh_edit(n_mesh, div):
    C = n_mesh.mesh_lap.to_dense()
    B = div
    # boundary condition
    C_add = torch.eye(len(n_mesh.vs))
    C = torch.cat([C, C_add], dim=0)
    B_add = torch.from_numpy(n_mesh.vs)
    B = torch.cat([torch.from_numpy(B), B_add], dim=0)
    A = torch.matmul(C.T, C)
    Ainv = torch.inverse(A)
    CtB = torch.matmul(C.T.float(), B.float())
    new_vs = torch.matmul(Ainv, CtB)

    return new_vs

def cg(n_mesh, div, iter=50, a=100):
    # preparation
    #C = n_mesh.mesh_lap.to_dense()
    #C = n_mesh.lapmat.to_dense()
    C = n_mesh.cot_mat.to_dense()
    B = div
    # boundary condition
    C_add = torch.eye(len(n_mesh.vs)) * a
    C = torch.cat([C, C_add], dim=0)
    Ct = C.T.to_sparse()
    C = C.to_sparse()
    B_add = torch.from_numpy(n_mesh.vs) * a
    B = torch.cat([torch.from_numpy(B), B_add], dim=0)
    B = torch.sparse.mm(Ct.float(), B.float())
    A = torch.sparse.mm(Ct, C.to_dense())
    # solve Ax=b by cg
    """
    x_0 = torch.from_numpy(n_mesh.vs).float()
    r_0 = B - torch.matmul(A, x_0)
    p_0 = r_0
    for i in range(iter):
        y_0 = torch.matmul(A, p_0)
        alpha = torch.diagonal(torch.matmul(r_0.T, r_0)) / (torch.diagonal(torch.matmul(p_0.T, y_0) + 1e-12))
        x_1 = x_0 + alpha * p_0
        r_1 = r_0 - alpha * y_0
        if torch.sum(torch.norm(r_1, dim=0), dim=0) < 1e-4:
            break
        beta = torch.diagonal(torch.matmul(r_1.T, r_1)) / (torch.diagonal(torch.matmul(r_0.T, r_0)) + 1e-12)
        p_1 = r_1 + beta * p_0
        
        x_0 = x_1
        r_0 = r_1
        p_0 = p_1
    return x_1
    """
    A = A.detach().numpy().copy()
    x_0 = torch.from_numpy(n_mesh.vs).float()
    x_1 = []
    for i in range(3):
        x_1.append(sp.sparse.linalg.cg(A, B[:,i], x0=x_0[:,i], maxiter=iter)[0].tolist())
    return np.array(x_1).T
    
def compute_fn(vs: torch.Tensor, faces: np.ndarray) -> torch.Tensor:
    """ compute face normals from mesh with Tensor """
    face_normals = torch.cross(vs[faces[:, 1]] - vs[faces[:, 0]], vs[faces[:, 2]] - vs[faces[:, 0]])
    norm = torch.sqrt(torch.sum(face_normals**2, dim=1))
    face_normals = face_normals / norm.repeat(3, 1).T
    return face_normals

def compute_vn(vs: torch.Tensor, fn: torch.Tensor, faces: np.ndarray) -> torch.Tensor:
    """ compute vertex normals from mesh with Tensor"""
    vert_normals = torch.zeros((3, len(vs)))
    face_normals = fn
    faces = torch.from_numpy(faces).long().to(vs.device)

    nv = len(vs)
    nf = len(faces)
    mat_rows = torch.reshape(faces, (-1,)).to(vs.device)
    mat_cols = torch.tensor([[i] * 3 for i in range(nf)]).reshape(-1).to(vs.device)
    mat_vals = torch.ones(len(mat_rows)).to(vs.device)
    f2v_mat = torch.sparse.FloatTensor(torch.stack([mat_rows, mat_cols], dim=0),
                                        mat_vals,
                                        size=torch.Size([nv, nf]))
    vert_normals = torch.sparse.mm(f2v_mat, face_normals)
    norm = torch.sqrt(torch.sum(vert_normals**2, dim=1))                    
    vert_normals = vert_normals / norm.repeat(3, 1).T
    return vert_normals

def uv2xyz(uv):
    u = 2.0 * np.pi * uv[:, 0] - np.pi
    v = np.pi * uv[:, 1]
    x = torch.sin(v) * torch.cos(u)
    y = torch.sin(v) * torch.sin(u)
    z = torch.cos(v)
    xyz = torch.stack([x, y, z]).T
    return xyz

def compute_nvt(mesh: Mesh) -> np.ndarray:
    f2ring = mesh.f2ring
    fa = mesh.fa
    fn = mesh.fn
    fc = mesh.fc
    
    f_group = np.zeros([len(fn), 3])

    for i, f in enumerate(f2ring):
        ci = fc[i].reshape(1, -1)
        cj = fc[f]
        nj = fn[f]
        """ (a cross b) cross a = (a dot a)b - (b dot a)a """
        a_a = np.sum((cj - ci) ** 2, 1).reshape(-1, 1)
        b_a = np.sum((cj - ci) * nj, 1).reshape(-1, 1)
        wj = a_a * nj - b_a * (cj - ci)
        wj = normalize(wj, norm="l2", axis=1)

        nw = np.sum(nj * wj, 1).reshape(-1, 1)
        nj_prime = 2 * nw * wj - nj
        
        am = np.max(fa[f]) + 1.0e-12
        aj = fa[f]
        cji_norm = np.linalg.norm(cj - ci, axis=1)
        sigma = np.mean(cji_norm) + 1.0e-12
        mu = (aj / am * np.exp(-cji_norm / sigma)).reshape(-1, 1)
        Ti = np.matmul(nj_prime.T, (nj_prime * mu))
        e_vals = np.sort(np.linalg.eig(Ti)[0])[::-1]
        if len(e_vals) != 3:
            print("len(e_vals) < 3 !")
        elif e_vals[1] < 0.01 and e_vals[2] < 0.001:
            f_group[i] = np.array([-1, -1, 1])
        elif e_vals[1] > 0.01 and e_vals[2] < 0.1:
            f_group[i] = np.array([0, 1, -1])
        elif e_vals[2] > 0.1:
            f_group[i] = np.array([-1, 1, -1])
        else:
            f_group[i] = np.array([-1, 0, 1])
        
    return f_group