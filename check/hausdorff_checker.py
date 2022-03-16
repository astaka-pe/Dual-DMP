import pymeshlab as ml
import numpy as np
import argparse
import glob
import os
import sys

def main():
    """ calculate hausdorff distances """
    parser = argparse.ArgumentParser(description="calculate hausdorff distances")
    parser.add_argument("-i", "--input", type=str, required=True)
    FLAGS = parser.parse_args()

    for k, v in vars(FLAGS).items():
        print("{:12s}: {}".format(k, v))

    all_path = glob.glob(os.path.join(FLAGS.input, "*.obj"))
    g_path = glob.glob(os.path.join(FLAGS.input, "*_gt.obj"))
    if len(g_path) == 0:
        print("No ground-truth mesh was detected!")
        sys.exit()

    os.makedirs(FLAGS.input + "/hd", exist_ok=True)
    ms = ml.MeshSet()
    hd_list = {}
    ms.load_new_mesh(g_path[0])
    face_num = ms.current_mesh().face_number()
    diag = ms.current_mesh().bounding_box().diagonal()
    max_val = diag * 0.002

    with open(FLAGS.input + "/hd/max_val.txt", mode="w") as f:
        f.write("{:.7f}".format(max_val))

    for a in all_path:
        if g_path[0] != a:
            #import pdb;pdb.set_trace()
            ms.load_new_mesh(a)
            res1 = ms.apply_filter("hausdorff_distance", sampledmesh=ms.current_mesh_id(), targetmesh=0, samplenum=face_num*3)
            quality = ms.current_mesh().vertex_quality_array()
            ms.apply_filter("colorize_by_vertex_quality", minval=0, maxval=max_val, zerosym=True)
            out_path = FLAGS.input + "/hd/" + os.path.basename(a).split(".")[0] + "={:.6f}.ply".format(res1["mean"]/res1["diag_mesh_0"])
            ms.save_current_mesh(out_path)
            
            res2 = ms.apply_filter("hausdorff_distance", sampledmesh=0, targetmesh=ms.current_mesh_id(), samplenum=face_num*3)
            hd_list[os.path.basename(a)] = 0.5 * (res1["mean"] / res1["diag_mesh_0"] + res2["mean"] / res2["diag_mesh_0"])
    
    for k, v in hd_list.items():
        print("{:20s}: {:.7f}".format(k, v))
    
if __name__ == "__main__":
    main()