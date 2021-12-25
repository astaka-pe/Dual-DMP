import pymeshlab as ml
import glob
import argparse

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

parser = argparse.ArgumentParser(description='preprocessing')
parser.add_argument('-i', '--input', type=str, required=True)
parser.add_argument('--step', type=int, default=30)
FLAGS = parser.parse_args()

for k, v in vars(FLAGS).items():
    print('{:12s}: {}'.format(k, v))

ms = ml.MeshSet()
n_file = glob.glob(FLAGS.input + "/*_noise.obj")[0]
mesh_name = n_file.split("/")[-2]
s_file = FLAGS.input + "/" + mesh_name + "_smooth.obj"
g_file = FLAGS.input + "/" + mesh_name + "_gt.obj"
ms.load_new_mesh(n_file)
smooth(ms, s_file, step=FLAGS.step)

if len(glob.glob(g_file)) > 0:
    ms.load_new_mesh(g_file) #ms: [smooth, gt]
    ms.load_new_mesh(n_file) #ms: [smooth, gt, noise]
    normalize(ms, n_file, s_file, g_file)

else:
    ms.load_new_mesh(n_file) #ms: [smooth, noise]
    normalize(ms, n_file, s_file)
