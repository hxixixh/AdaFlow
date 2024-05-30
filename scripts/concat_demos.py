import sys
from os import path
sys.path.append(path.dirname( path.abspath(__file__) ) )
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import argparse
import h5py
import json
import numpy as np
import os

import robosuite
import robosuite.utils.transform_utils as T
import robosuite.macros as macros

import libero

from pathlib import Path
from tqdm import tqdm

import libero.libero.utils.utils as chiliocosm_utils
from libero.libero.envs import *


def copy_group(source_group, target_group):
    for key, item in source_group.items():
        if isinstance(item, h5py.Group):
            new_group = target_group.create_group(key)
            copy_group(item, new_group)
        elif isinstance(item, h5py.Dataset):
            target_group.create_dataset(key, data=item)
    for attr_key, attr_val in source_group.attrs.items():
        target_group.attrs[attr_key] = attr_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--demo-file',
        default="demo.hdf5"
    )
    parser.add_argument(
        '--out_hdf5_path',
        type=str,
        default="exps/0320_vis_demo/orig",
    ) 
    parser.add_argument(
        '--num_demos', 
        type=int, 
        default=20, 
    )
    parser.add_argument(
        "--path_to_libero_lib", 
        type=str,
        default="libero/libero",
        help="The path to the libero lib directory"
    )
    args = parser.parse_args()

    f = h5py.File(args.demo_file, "r")
    
    if "env_name" in f["data"].attrs:
        env_name = f["data"].attrs["env_name"]  # generated demos
    elif "env" in f["data"].attrs:
        env_name = f["data"].attrs["env"]   # original collect demos

    if "env_args" in f["data"].attrs:
        env_args = f["data"].attrs["env_args"]
        env_kwargs = json.loads(env_args)['env_kwargs']
    elif "env_info" in f["data"].attrs:
        env_args = f["data"].attrs["env_info"]
        env_kwargs = json.loads(env_args)

    problem_info = json.loads(f["data"].attrs["problem_info"])
    problem_info["domain_name"]
    problem_name = problem_info["problem_name"]

    # list of all demonstrations episodes
    demos = list(f["data"].keys())
    bddl_file_name = f["data"].attrs["bddl_file_name"]
    if not os.path.exists(bddl_file_name): 
        task_group = args.demo_file.split("/")[-2]
        hdf5_file_name = args.demo_file.split("/")[-1]
        bddl_file_name = os.path.join(args.path_to_libero_lib, "libero/libero/bddl_files", task_group, hdf5_file_name.replace("_demo.hdf5", ".bddl"))

    h5py_f = h5py.File(args.out_hdf5_path, "w")
    grp = h5py_f.create_group("data")

    grp.attrs["env_name"] = env_name
    grp.attrs["problem_info"] = f["data"].attrs["problem_info"]
    grp.attrs["macros_image_convention"] = macros.IMAGE_CONVENTION

    chiliocosm_utils.update_env_kwargs(env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_depths=False, 
        camera_names=["robot0_eye_in_hand",
                    "agentview",
                    ],
        reward_shaping=True,
        control_freq=20,
        camera_heights=128,
        camera_widths=128,
        camera_segmentations=None, 
    )

    grp.attrs["bddl_file_name"] = bddl_file_name
    grp.attrs["bddl_file_content"] = open(bddl_file_name, "r").read()
    print(grp.attrs["bddl_file_content"])

    env_args = {"type": 1,
                "env_name": env_name,
                "problem_name": problem_name,
                "bddl_file": bddl_file_name, 
                "env_kwargs": env_kwargs}
    
    grp.attrs["env_args"] = json.dumps(env_args)
    print(grp.attrs["env_args"])

    total_len = 0
    
    for demo in demos.copy(): # Use copy() to avoid modifying the list while iterating
        demo_num = int(demo.split('_')[1]) # Split the string and convert the second part to an integer
        if demo_num >= args.num_demos:
            demos.remove(demo)

    # append "demo_0" if it's not in the list
    if "demo_0" not in demos:
        demos.append("demo_0")
        
    # check if the demos exist
    for demo in demos: 
        demo_idx = int(demo.split("_")[-1])
        hdf5_demo_path = args.out_hdf5_path.replace(".hdf5", f"_{demo_idx}.hdf5")
        if not os.path.exists(hdf5_demo_path): 
            print(f"demo {demo_idx} does not exist")
            sys.exit()
    
    for demo in demos: 
        demo_idx = int(demo.split("_")[-1])
        hdf5_demo_path = args.out_hdf5_path.replace(".hdf5", f"_{demo_idx}.hdf5")

        f_demo = h5py.File(hdf5_demo_path, "r")
        
        try: 
            demo_grp = f_demo["data"][demo]
        except:
            import pdb; pdb.set_trace()

        ep_data_grp = grp.create_group(f"demo_{demo_idx}")
        copy_group(demo_grp, ep_data_grp)

        total_len += len(ep_data_grp['obs']["agentview_rgb"])

        f_demo.close()
        
        os.remove(hdf5_demo_path)

    f.close()

if __name__ == "__main__":
    main()