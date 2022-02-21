import pymeshlab as ml
import numpy as np
import shutil
import os
import argparse
from util.mesh import Mesh
import util.loss as Loss

def normalize(ms):
    ms.apply_filter("transform_scale_normalize", scalecenter= "barycenter", unitflag=True)
    ms.apply_filter("transform_translate_center_set_origin", traslmethod="Center on Layer BBox")

def edge_based_scaling(mesh):
    edge_vec = mesh.vs[mesh.edges][:, 0, :] - mesh.vs[mesh.edges][:, 1, :]
    ave_len = np.sum(np.linalg.norm(edge_vec, axis=1)) / mesh.edges.shape[0]
    mesh.vs /= ave_len
    return mesh
    
def main():
    """ create datasets(noisy mesh & smoothed mesh) from a single clean mesh """
    parser = argparse.ArgumentParser(description='preprocessing')
    parser.add_argument('-i', '--input', type=str, required=True)
    FLAGS = parser.parse_args()

    for k, v in vars(FLAGS).items():
        print('{:12s}: {}'.format(k, v))
    
    ms = ml.MeshSet()
    ms.load_new_mesh(FLAGS.input)
    normalize(ms)                       # pre-scaling & transformation
    ms.save_current_mesh(FLAGS.input)
    g_mesh = Mesh(FLAGS.input)
    g_mesh = edge_based_scaling(g_mesh) # re-scaling
    g_mesh.compute_face_normals()
    g_mesh.save(FLAGS.input)

if __name__ == "__main__":
    main()