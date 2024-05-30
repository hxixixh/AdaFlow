if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import click
import copy
import h5py
import numpy as np
import os
import pathlib
import shutil
from tqdm import tqdm

from scipy.ndimage import zoom

import robosuite.utils.transform_utils as T


def resize_img_array(img_arr): 
    n_imgs = img_arr.shape[0]

    # Calculate the scaling factors for the image dimensions
    scale_factor_y = 84 / 128
    scale_factor_x = 84 / 128
    
    # Resize the images
    resized_images = zoom(img_arr, (1, scale_factor_y, scale_factor_x, 1))

    return resized_images


@click.command()
@click.option('-i', '--input_file', required=True, help='input hdf5 path')
@click.option('-o', '--output_file', required=True, help='output hdf5 path. Parent directory must exist')
def main(input_file, output_file):
    # process inputs
    input_file = pathlib.Path(input_file).expanduser()
    assert input_file.is_file()
    output_file = pathlib.Path(output_file).expanduser()
    os.makedirs(output_file.parent, exist_ok=True)
    assert output_file.parent.is_dir()
    assert not output_file.is_dir()

    # save output
    print('Copying hdf5')
    shutil.copy(str(input_file), str(output_file))

    # modify observation keys
    with h5py.File(output_file, "r+") as out_file: 
        for demo in tqdm(out_file["data"].keys()): 
            for obs_key in out_file["data"][demo]["obs"].keys(): 
                if obs_key in ["ee_quat", "ee_ori"]: 
                    ee_ori = out_file["data"][demo]["obs"][obs_key][()]
                    ee_quat = []
                    for i in range(ee_ori.shape[0]): 
                        ee_ori_i = ee_ori[i]
                        ee_quat_i = T.axisangle2quat(ee_ori_i)
                        ee_quat.append(ee_quat_i)
                    ee_quat = np.stack(ee_quat)
                    out_file["data"][demo]["obs"]["robot0_eef_quat"] = ee_quat
                if obs_key == "ee_pos": 
                    out_file.copy(out_file[f"data/{demo}/obs/ee_pos"], f"data/{demo}/obs/robot0_eef_pos")
                if obs_key == "gripper_states": 
                    out_file.copy(out_file[f"data/{demo}/obs/gripper_states"], f"data/{demo}/obs/robot0_gripper_qpos")
                if obs_key == "agentview_rgb": 
                    rgb_arr = out_file[f"data/{demo}/obs/agentview_rgb"][()]
                    resized_rgb_arr = resize_img_array(rgb_arr)
                    out_file[f"data/{demo}/obs/agentview_image"] = resized_rgb_arr
                if obs_key == "eye_in_hand_rgb": 
                    rgb_arr = out_file[f"data/{demo}/obs/eye_in_hand_rgb"][()]
                    resized_rgb_arr = resize_img_array(rgb_arr)
                    out_file[f"data/{demo}/obs/robot0_eye_in_hand_image"] = resized_rgb_arr
                del out_file["data"][demo]["obs"][obs_key]
                


if __name__ == "__main__":
    main()