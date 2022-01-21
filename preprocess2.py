import pymeshlab as ml
import numpy as np
import shutil
import os
import argparse
from util.mesh import Mesh

def smooth(ms, step):
    ms.apply_filter("laplacian_smooth", stepsmoothnum=step, cotangentweight=False)

def normalize(ms):
    ms.apply_filter("transform_scale_normalize", scalecenter= "barycenter", unitflag=True)
    ms.apply_filter("transform_translate_center_set_origin", traslmethod="Center on Layer BBox")

def edge_based_scaling(mesh):
    edge_vec = mesh.vs[mesh.edges][:, 0, :] - mesh.vs[mesh.edges][:, 1, :]
    ave_len = np.sum(np.linalg.norm(edge_vec, axis=1)) / mesh.edges.shape[0]
    mesh.vs /= ave_len
    return mesh

def gausian_noise(mesh, noise_level):
    np.random.seed(314)
    noise = np.random.normal(loc=0, scale=noise_level, size=(len(mesh.vs), 1))
    mesh.vs += mesh.vn * noise
    return mesh
    

""" create datasets(noisy mesh & smoothed mesh) from a single clean mesh """
parser = argparse.ArgumentParser(description='preprocessing')
parser.add_argument('-i', '--input', type=str, required=True)
parser.add_argument('--noise', type=str, default="gaussian")
parser.add_argument('--level', type=float, default="0.2")
parser.add_argument('--step', type=int, default=30)
FLAGS = parser.parse_args()

for k, v in vars(FLAGS).items():
    print('{:12s}: {}'.format(k, v))

ms = ml.MeshSet()
root_dir = os.path.dirname(FLAGS.input)
mesh_name = root_dir.split("/")[-1]

n_file = os.path.join(root_dir, mesh_name + "_noise.obj")
s_file = os.path.join(root_dir, mesh_name + "_smooth.obj")
g_file = os.path.join(root_dir, mesh_name + "_gt.obj")

ms.load_new_mesh(FLAGS.input)
os.makedirs(os.path.join(root_dir, "original"), exist_ok=True)
shutil.move(FLAGS.input, os.path.join(root_dir, "original", os.path.basename(FLAGS.input)))
normalize(ms)                       # pre-scaling & transformation
ms.save_current_mesh(g_file)        # pre-saving
g_mesh = Mesh(g_file)
g_mesh = edge_based_scaling(g_mesh) # re-scaling
g_mesh.save(g_file)

if FLAGS.noise == "gaussian":
    n_mesh = gausian_noise(g_mesh, FLAGS.level)
    n_mesh.save(n_file)
else:
    n_mesh = gausian_noise(g_mesh, level)
    n_mesh.save(n_file)

ms.load_new_mesh(n_file)
smooth(ms, step=FLAGS.step)         # smoothing
ms.save_current_mesh(s_file)