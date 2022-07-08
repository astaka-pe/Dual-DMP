import pymeshlab as ml
import numpy as np
import glob
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from util.mesh import Mesh

def get_parser():
    parser = argparse.ArgumentParser(description='preprocessing')
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('--step', type=int, default=30)
    args = parser.parse_args()

    for k, v in vars(args).items():
        print('{:12s}: {}'.format(k, v))
    
    return args

def smooth(ms, s_file, step):
    ms.apply_filter("laplacian_smooth", stepsmoothnum=step, cotangentweight=False)
    ms.save_current_mesh(s_file)

def normalize(ms, n_file, s_file, g_file=None):
    ms.apply_filter("transform_scale_normalize", scalecenter = "barycenter", unitflag=True, alllayers=True)
    ms.apply_filter("transform_translate_center_set_origin", traslmethod="Center on Layer BBox", alllayers=True)
    if ms.number_meshes() == 2:
        ms.set_current_mesh(0)
        ms.save_current_mesh(s_file)
        ms.set_current_mesh(1)
        ms.save_current_mesh(n_file)
    elif ms.number_meshes() == 3:
        ms.set_current_mesh(0)
        ms.save_current_mesh(s_file)
        ms.set_current_mesh(1)
        ms.save_current_mesh(g_file)
        ms.set_current_mesh(2)
        ms.save_current_mesh(n_file)

def main():
    args = get_parser()
    
    ms = ml.MeshSet()
    n_file = glob.glob(args.input + "/*_noise.obj")[0]
    mesh_name = n_file.split("/")[-2]
    s_file = args.input + "/" + mesh_name + "_smooth.obj"
    g_file = args.input + "/" + mesh_name + "_gt.obj"
    ms.load_new_mesh(n_file)
    smooth(ms, s_file, step=args.step)

    if len(glob.glob(g_file)) > 0:
        ms.load_new_mesh(g_file) #ms: [smooth, gt]
        ms.load_new_mesh(n_file) #ms: [smooth, gt, noise]
        normalize(ms, n_file, s_file, g_file)

    else:
        ms.load_new_mesh(n_file) #ms: [smooth, noise]
        normalize(ms, n_file, s_file)

    n_mesh = Mesh(n_file)
    s_mesh = Mesh(s_file)
    g_mesh = Mesh(g_file)

    edge_vec = n_mesh.vs[n_mesh.edges][:, 0, :] - n_mesh.vs[n_mesh.edges][:, 1, :]
    ave_len = np.sum(np.linalg.norm(edge_vec, axis=1)) / n_mesh.edges.shape[0]

    n_mesh.vs /= ave_len
    g_mesh.vs /= ave_len
    s_mesh.vs /= ave_len

    n_mesh.save(n_file)
    s_mesh.save(s_file)
    g_mesh.save(g_file)


if __name__ == "__main__":
    main()