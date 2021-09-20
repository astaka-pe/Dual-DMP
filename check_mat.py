import pandas as pd
from util.models import Mesh
import util.loss as Loss
import util.models as Models
import os

n_mesh = Mesh("datasets/sharp/sharp_noise.obj")
g_mesh = Mesh("datasets/grayloc/grayloc_gt.obj")

nvt, e_str = Models.compute_nvt(g_mesh)
path = os.path.basename(g_mesh.path).split(".")[0] + "_estr.ply"
Mesh.save_as_ply(g_mesh, path, e_str)