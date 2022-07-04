import numpy as np
import torch
from util.mesh import Mesh
    
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

def vertex_updating(pos: torch.Tensor, norm: torch.Tensor, mesh: Mesh, loop=10) -> torch.Tensor:
    new_pos = pos.detach().clone()
    norm = norm.detach().clone()
    for iter in range(loop):
        fc = torch.sum(new_pos[mesh.faces], 1) / 3.0
        for i in range(len(new_pos)):
            cis = fc[list(mesh.vf[i])]
            nis = norm[list(mesh.vf[i])]
            cvis = cis - new_pos[i].reshape(1, -1)
            ncvis = torch.sum(nis * cvis, dim=1)
            dvi = torch.sum(ncvis.reshape(-1, 1) * nis, dim=0)
            dvi /= len(mesh.vf[i])
            new_pos[i] += dvi
    return new_pos
