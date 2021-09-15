from util.models import Mesh
import util.loss as Loss
import util.models as Models
import os

n_mesh = Mesh("datasets/sharp/sharp_noise.obj")
g_mesh = Mesh("datasets/fandisk03/fandisk03_noise.obj")

nvt = Models.compute_nvt(g_mesh)
path = os.path.basename(g_mesh.path).split(".")[0] + ".ply"
Mesh.save_as_ply(g_mesh, path, nvt)