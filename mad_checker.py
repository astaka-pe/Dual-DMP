import argparse
import glob
import os
import sys

from util.mesh import Mesh
import util.loss as Loss

def main():
    """ display MAD's  """
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

    g_mesh = Mesh(g_path[0])
    mad_list = {}
    for a in all_path:
        if g_path[0] != a:
            a_mesh = Mesh(a)
            mad_list[os.path.basename(a)] = Loss.mad(a_mesh.fn, g_mesh.fn)
    
    for k, v in mad_list.items():
        print("{:20s}: {:.3f}".format(k, v))

if __name__ == "__main__":
    main()