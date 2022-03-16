import argparse
import glob
import os
import sys
import numpy as np
from matplotlib import cm

sys.path.append(".")
from util.mesh import Mesh
import util.loss as Loss

def mad2color(mad, max_th=50):
    color = np.zeros([len(mad), 3]).astype(np.float)
    mad = np.clip(mad, 0.0, max_th)
    mad /= max_th
    c = cm.jet(mad)
    color = c[:, :3]
    # blue = 1.0 - green
    # color[:, 1] = green
    # color[:, 2] = blue
    return color

def main():
    """ display MADs  """
    parser = argparse.ArgumentParser(description='preprocessing')
    parser.add_argument('-i', '--input', type=str, required=True)
    FLAGS = parser.parse_args()

    for k, v in vars(FLAGS).items():
        print('{:12s}: {}'.format(k, v))
    
    all_path = glob.glob(os.path.join(FLAGS.input, "*.obj"))
    g_path = glob.glob(os.path.join(FLAGS.input, "*_gt.obj"))
    if len(g_path) == 0:
        print("No ground-truth mesh was detected!")
        sys.exit()

    os.makedirs(FLAGS.input + "/mad", exist_ok=True)
    g_mesh = Mesh(g_path[0])
    mad_list = {}
    for a in all_path:
        if g_path[0] != a:
            a_mesh = Mesh(a)
            mad = Loss.mad(a_mesh.fn, g_mesh.fn)
            sad = Loss.angular_difference(a_mesh.fn, g_mesh.fn)
            mad_list[os.path.basename(a)] = mad
            color = mad2color(sad)
            a_mesh.save_as_ply(FLAGS.input + "/mad/" + os.path.basename(a).split(".")[0] + "={:.3f}.ply".format(mad), color)
    
    for k, v in mad_list.items():
        print("{:20s}: {:.3f}".format(k, v))

if __name__ == "__main__":
    main()