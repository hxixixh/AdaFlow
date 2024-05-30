import sys
from os import path
sys.path.append(path.dirname( path.abspath(__file__) ) )
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import argparse
import copy
import glob
import h5py
import json
import numpy as np
import os
import re

import robosuite
import robosuite.utils.transform_utils as T
import robosuite.macros as macros

from pathlib import Path
from PIL import Image
from tqdm import tqdm

import libero.libero.utils.utils as chiliocosm_utils
from libero.libero.envs import *


def extract_rgb(demo_file):

    f = h5py.File(demo_file, "r")

    demos = list(f["data"].keys())

    for demo in tqdm(demos, desc='Processing demos'): 
        obs = f["data"][demo]["obs"]

        for key in obs.keys(): 
            if "rgb" in key or "image" in key: 
                rgb_save_dir = os.path.join(demo_file.replace(".hdf5", ""), demo, key)

                if os.path.exists(rgb_save_dir): 
                    hdf5_num_img = obs[key][()].shape[0]
                    num_files = len(os.listdir(rgb_save_dir))
                    if hdf5_num_img != num_files:
                        pass
                    else: 
                        print(f"Directory {rgb_save_dir} already exists. Skipping...")
                        continue
                os.makedirs(rgb_save_dir, exist_ok=True)
                rgbs = obs[key][()]
                T, H, W, C = rgbs.shape
                for t in tqdm(range(T), desc=f'Processing frames in {demo}'):
                    rgb_save_path = os.path.join(rgb_save_dir, f"{str(t).zfill(5)}.jpg")
                    rgb = rgbs[t]

                    # Save rgb to jpg
                    image = Image.fromarray(rgb)
                    image.save(rgb_save_path)
    f.close()


def copy_group_without_img(source_group, target_group):
    for key, item in source_group.items():
        if isinstance(item, h5py.Group):
            if key == "obs":
                new_group = target_group.create_group(key)
                for obs_key, obs_item in item.items(): 
                    if "rgb" not in obs_key and "depth" not in obs_key:
                        if isinstance(obs_item, h5py.Dataset):
                            new_group.create_dataset(obs_key, data=obs_item)
                for attr_key, attr_val in item.attrs.items():
                    new_group.attrs[attr_key] = attr_val
            else:
                new_group = target_group.create_group(key)
                copy_group_without_img(item, new_group)
        elif isinstance(item, h5py.Dataset):
            target_group.create_dataset(key, data=item)
    for attr_key, attr_val in source_group.attrs.items():
        target_group.attrs[attr_key] = attr_val


def save_lowdim_hdf5(demo_file):
    # Open the input file in read mode
    f_in = h5py.File(demo_file, "r")
    
    # Create the output file  
    out_file = demo_file.replace(".hdf5", "_lowdim.hdf5")
    f_out = h5py.File(out_file, "w")
    
    if "env_name" in f_in["data"].attrs:
        env_name = f_in["data"].attrs["env_name"]  # generated demos
    elif "env" in f_in["data"].attrs:
        env_name = f_in["data"].attrs["env"]   # original collect demos
    if "env_args" in f_in["data"].attrs:
        env_args = f_in["data"].attrs["env_args"]
        env_kwargs = json.loads(env_args)['env_kwargs']
    elif "env_info" in f_in["data"].attrs:
        env_args = f_in["data"].attrs["env_info"]
        env_kwargs = json.loads(env_args)

    problem_info = json.loads(f_in["data"].attrs["problem_info"])
    problem_info["domain_name"]
    problem_name = problem_info["problem_name"]

    # list of all demonstrations episodes
    demos = list(f_in["data"].keys())
    bddl_file_name = f_in["data"].attrs["bddl_file_name"]


    grp = f_out.create_group("data")

    grp.attrs["env_name"] = env_name
    grp.attrs["problem_info"] = f_in["data"].attrs["problem_info"]
    grp.attrs["macros_image_convention"] = macros.IMAGE_CONVENTION

    chiliocosm_utils.update_env_kwargs(env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_depths=False,
        camera_names=[],
        reward_shaping=True,
        control_freq=20,
        camera_heights=128,
        camera_widths=128,
        camera_segmentations=None, 
    )
    grp.attrs["bddl_file_name"] = bddl_file_name
    grp.attrs["bddl_file_content"] = open(bddl_file_name, "r").read()

    env_args = {"type": 1,
                "env_name": env_name,
                "problem_name": problem_name,
                "bddl_file": bddl_file_name, 
                "env_kwargs": env_kwargs}
    
    grp.attrs["env_args"] = json.dumps(env_args)

    for demo in demos: 
        demo_grp = f_in["data"][demo]

        ep_data_grp = grp.create_group(demo)
        copy_group_without_img(demo_grp, ep_data_grp)

    # Close the input and output files
    f_in.close()
    f_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-folder",
        type=str, 
        default="datasets/libero_panda/libero_90"
    )
    parser.add_argument(
        '--save_rgb',
        action="store_true"
    )
    parser.add_argument(
        '--save_lowdim', 
        action="store_true"
    )
    args = parser.parse_args()

    # Grab all hdf5 path from args.dataset_folder
    hdf5_paths = glob.glob(f"{args.dataset_folder}/*.hdf5")
    hdf5_paths = [f for f in hdf5_paths if not re.search(r'\d\.hdf5$', f)]

    # Print the list of HDF5 files
    for hdf5_path in tqdm(hdf5_paths):
        print("processing", hdf5_path)
        if "lowdim" in hdf5_path: 
            continue
        if not "abs" in hdf5_path: 
            continue
        if args.save_rgb: 
            extract_rgb(hdf5_path)
        if args.save_lowdim:
            save_lowdim_hdf5(hdf5_path)
    