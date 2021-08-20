from util.models import Mesh
import util.loss as Loss

n_mesh = Mesh("datasets/dragon/dragon_noise.obj", build_mat=True)
g_mesh = Mesh("datasets/dragon/dragon_gt.obj")

loss = Loss.pos4norm(g_mesh, g_mesh.fn)
print(loss)