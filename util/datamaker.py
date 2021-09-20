import glob
import numpy as np
import torch
from .mesh import Mesh
from torch_geometric.data import Data
from typing import Tuple

class Dataset:
    def __init__(self, data):
        self.keys = data.keys
        self.num_nodes = data.num_nodes
        self.num_edges = data.num_edges
        self.num_node_features = data.num_node_features
        self.contains_isolated_nodes = data.contains_isolated_nodes()
        self.contains_self_loops = data.contains_self_loops()
        self.z1 = data['z1']
        self.z2 = data['z2']
        self.x_pos = data['x_pos']
        self.x_norm = data['x_norm']
        self.edge_index = data['edge_index']
        self.face_index = data['face_index']

def create_dataset(file_path: str) -> Tuple[dict, Dataset]:
    """ create mesh """
    mesh_dic = {}
    gt_file = glob.glob(file_path + '/*_gt.obj')[0]
    n_file = glob.glob(file_path + '/*_noise.obj')[0]
    s_file = glob.glob(file_path + '/*_smooth.obj')[0]
    mesh_name = gt_file.split('/')[-2]

    gt_mesh = Mesh(gt_file)
    n_mesh = Mesh(n_file)
    o1_mesh = Mesh(n_file)
    #o2_mesh = Mesh(n_file)
    s_mesh = Mesh(s_file)

    """ create graph """
    np.random.seed(314)
    z1 = np.random.normal(size=(n_mesh.vs.shape[0], 16))
    np.random.seed(314)
    z2 = np.random.normal(size=(n_mesh.fn.shape[0], 16))
    z1, z2 = torch.tensor(z1, dtype=torch.float, requires_grad=True), torch.tensor(z2, dtype=torch.float, requires_grad=True)

    x_pos = torch.tensor(s_mesh.vs, dtype=torch.float)
    x_norm = torch.tensor(n_mesh.fn, dtype=torch.float)

    edge_index = torch.tensor(n_mesh.edges.T, dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index[[1,0],:]], dim=1)
    face_index = torch.from_numpy(n_mesh.f_edges)

    """ create dataset """
    data = Data(x=z1, z1=z1, z2=z2, x_pos=x_pos, x_norm=x_norm, edge_index=edge_index, face_index=face_index)
    dataset = Dataset(data)

    mesh_dic["gt_file"] = gt_file
    mesh_dic["n_file"] = n_file
    mesh_dic["s_file"] = s_file
    mesh_dic["mesh_name"] = mesh_name
    mesh_dic["gt_mesh"] = gt_mesh
    mesh_dic["n_mesh"] = n_mesh
    mesh_dic["o1_mesh"] = o1_mesh
    mesh_dic["s_mesh"] = s_mesh

    return mesh_dic, dataset