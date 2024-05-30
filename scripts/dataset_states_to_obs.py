"""
Reference: https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/scripts/playback_demonstrations_from_hdf5.py

A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.
"""
import sys
from os import path
sys.path.append(path.dirname( path.abspath(__file__) ) )
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import argparse
import h5py
import json
import numpy as np
import os
import pickle

import robosuite.utils.camera_utils as CU
import robosuite.utils.transform_utils as T
import robosuite.macros as macros

from pathlib import Path
from tqdm import tqdm

import libero.libero.utils.utils as chiliocosm_utils
from libero.libero.envs import *


def save_hdf5_ep_grp(grp, args, demo_obs, init_idx, demo_idx, total_len): 
    ep_data_grp = grp.create_group(f"demo_{demo_idx}")

    obs_grp = ep_data_grp.create_group("obs")
    if not args.no_proprio:
        obs_grp.create_dataset("gripper_states", data=demo_obs["gripper_states"])
        obs_grp.create_dataset("joint_states", data=demo_obs["joint_states"])
        obs_grp.create_dataset("ee_states", data=demo_obs["ee_states"])
        obs_grp.create_dataset("ee_pos", data=demo_obs["ee_pos"])
        obs_grp.create_dataset("ee_quat", data=demo_obs["ee_quat"])
    
    obs_grp.create_dataset("agentview_rgb", data=demo_obs["agentview_rgb"])
    obs_grp.create_dataset("eye_in_hand_rgb", data=demo_obs["eye_in_hand_rgb"])

    # object states
    if len(demo_obs["obj_states"]) > 0:
        for key in demo_obs["obj_states"]: 
            obs_grp.create_dataset(key, data=demo_obs["obj_states"][key])

    ep_data_grp.create_dataset("actions", data=demo_obs["actions"])
    ep_data_grp.create_dataset("states", data=demo_obs["states"])
    ep_data_grp.create_dataset("robot_states", data=demo_obs["robot_states"])
    ep_data_grp.create_dataset("rewards", data=demo_obs["rewards"])
    ep_data_grp.create_dataset("dones", data=demo_obs["dones"])

   
    ep_data_grp.attrs["num_samples"] = len(demo_obs["agentview_rgb"])
    ep_data_grp.attrs["model_file"] = demo_obs["model_xml"]
    ep_data_grp.attrs["init_state"] = demo_obs["states"][init_idx]
    total_len += len(demo_obs["agentview_rgb"])

    return grp, total_len


def rollout_demo(problem_name, env_kwargs, demos, demo_index, f, args, init_idx, cap_idx): 
    env = TASK_MAPPING[problem_name](
        **env_kwargs,
    )
    ep = demos[demo_index]
    print(f"Playing back {ep}... (press ESC to quit)")

    # save camera intrinsics and extrinsics
    camera_proj_transform_mats = dict()
    for camera_name in env_kwargs["camera_names"]: 
        if camera_name == "robot0_eye_in_hand": 
            continue
        camera_proj_transform_mats[camera_name] = dict()

        world_to_camera = CU.get_camera_transform_matrix(
            sim=env.sim,
            camera_name=camera_name,
            camera_height=env_kwargs["camera_heights"],
            camera_width=env_kwargs["camera_widths"],
        )

        camera_proj_transform_mats[camera_name] = world_to_camera

    cam_mats_save_path = os.path.join(Path(args.out_hdf5_path).parent, "camera_proj_transform_mats.pkl")
    if not os.path.exists(cam_mats_save_path):
        with open(cam_mats_save_path, "wb") as file:
            pickle.dump(camera_proj_transform_mats, file)

    # select an episode randomly
    # read the model xml, using the metadata stored in the attribute for this episode
    states = f["data/{}/states".format(ep)][()]
    initial_state = dict(states=states[0])
    initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

    # extract obs, rewards, dones
    actions = f["data/{}/actions".format(ep)][()]

    model_xml = f["data/{}".format(ep)].attrs["model_file"]
    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except:
            continue
    
    model_xml = chiliocosm_utils.postprocess_model_xml(model_xml, {})
    
    model_xml = model_xml.replace("/home/yifengz/workspace/libero-dev/chiliocosm", f"{args.path_to_libero_lib}/libero/libero")
    
    if not args.use_camera_obs:
        env.viewer.set_camera(0)

    # load the flattened mujoco states
    states = f["data/{}/states".format(ep)][()]
    actions = np.array(f["data/{}/actions".format(ep)][()])

    num_actions = actions.shape[0]
    
    env.sim.reset()        
    env.sim.set_state_from_flattened(states[init_idx])
    env.sim.forward()

    model_xml = env.sim.model.get_xml()

    ee_states = []
    gripper_states = []
    joint_states = []
    robot_states = []

    agentview_images = []
    eye_in_hand_images = []
    
    rewards = []
    dones = []

    valid_index = []

    obj_states = {}

    for j, action in tqdm(enumerate(actions), total=len(actions)):
        # # For debugging
        if args.debug and j > 10: 
            print(f"debugging at {j}-th step")
            break

        obs, reward, done, info = env.step(action)

        if j < cap_idx: 
            continue

        # reset state rather than rollout actions
        if j == len(actions) - 1: 
            obs, reward, done, info = env.step(action)
        else: 
            env.reset_from_xml_string(model_xml)
            env.sim.reset()        
            env.sim.set_state_from_flattened(states[j+1])
            env.sim.forward()

            env._update_observables(force=True)
            obs = env._get_observations()

        if j < num_actions - 1:
            # ensure that the actions deterministically lead to the same recorded states
            state_playback = env.sim.get_state().flatten()
            err = np.linalg.norm(states[j + 1] - state_playback)

            if err > 0.01:
                print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")

        valid_index.append(j)

        if not args.no_proprio:
            if "robot0_gripper_qpos" in obs:
                gripper_states.append(obs["robot0_gripper_qpos"])

            joint_states.append(obs["robot0_joint_pos"])

            ee_states.append(np.hstack((obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"]))))

        robot_states.append(env.get_robot_state_vector(obs))

        if args.use_camera_obs:
            agentview_images.append(obs["agentview_image"])
            eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])
        else:
            env.render()

    # end of one trajectory
    states = states[valid_index]
    actions = actions[valid_index]
    dones = np.zeros(len(actions)).astype(np.uint8)
    dones[-1] = 1
    rewards = np.zeros(len(actions)).astype(np.uint8)
    rewards[-1] = 1 
    print(len(actions) ,len(agentview_images))
    assert(len(actions) == len(agentview_images))

    env.close()

    # concatenate obj states: 
    if len(obj_states.keys()) > 0:
        for key in obj_states.keys(): 
            obj_states[key] = np.stack(obj_states[key], axis=0)

    return {
        "gripper_states": np.stack(gripper_states, axis=0),
        "joint_states": np.stack(joint_states, axis=0),
        "ee_states": np.stack(ee_states, axis=0),
        "ee_pos": np.stack(ee_states, axis=0)[:, :3], 
        "ee_quat": np.stack(ee_states, axis=0)[:, 3:],
        "agentview_rgb": np.stack(agentview_images, axis=0), 
        "eye_in_hand_rgb": np.stack(eye_in_hand_images, axis=0), 
        "actions": actions, 
        "states": states,
        "robot_states": np.stack(robot_states, axis=0),
        "obj_states": obj_states, 
        "rewards": rewards,
        "dones": dones,
        "model_xml": model_xml,
    }
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--demo-file',
        default="demo.hdf5"
    )
    parser.add_argument(
        "--use-actions", 
        action='store_true',
    )
    parser.add_argument(
        "--use-camera-obs", 
        action="store_true"
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default="datasets/",
    )
    parser.add_argument(
        '--out_hdf5_path',
        type=str,
        default=None,
    ) 
    parser.add_argument(
        '--no-proprio',
        action='store_true'
    )

    parser.add_argument(
        '--use-depth',
        action='store_true',
    )
    parser.add_argument(
        "--debug", 
        action='store_true',
        help="If set, will only process the first 4 steps of each demos"
    )
    parser.add_argument(
        "--demo_index", 
        type=int, 
        default=-1,
        help="The index of the demo to process"
    )
    parser.add_argument(
        "--path_to_libero_lib", 
        type=str,
        default="libero/libero",
        help="The path to the libero lib directory"
    )

    args = parser.parse_args()

    assert args.demo_index >= 0, "Please specify the demo index to process"

    print("=====================================")
    print("Processing one demo only")
    print("=====================================")

    
    hdf5_path = args.demo_file
    hdf5_out_path = args.out_hdf5_path.replace(".hdf5", f"_{args.demo_index}.hdf5")

    if os.path.exists(hdf5_out_path):
        print(f"hdf5 file exists: {hdf5_out_path}")
        sys.exit()

    f = h5py.File(hdf5_path, "r")
    if "env_name" in f["data"].attrs:
        env_name = f["data"].attrs["env_name"]  # generated demos
        cap_index = 0
    elif "env" in f["data"].attrs:
        env_name = f["data"].attrs["env"]   # original collect demos
        cap_index = 5

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
        task_group = hdf5_path.split("/")[-2]
        hdf5_file_name = hdf5_path.split("/")[-1]
        bddl_file_name = os.path.join(args.path_to_libero_lib, "libero/libero/bddl_files", task_group, hdf5_file_name.replace("_demo.hdf5", ".bddl"))

    output_parent_dir = Path(hdf5_out_path).parent
    output_parent_dir.mkdir(parents=True, exist_ok=True)

    h5py_f = h5py.File(hdf5_out_path, "w")

    grp = h5py_f.create_group("data")

    grp.attrs["env_name"] = env_name
    grp.attrs["problem_info"] = f["data"].attrs["problem_info"]
    grp.attrs["macros_image_convention"] = macros.IMAGE_CONVENTION

    chiliocosm_utils.update_env_kwargs(env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=not args.use_camera_obs,
        has_offscreen_renderer=args.use_camera_obs,
        ignore_done=True,
        use_camera_obs=args.use_camera_obs,
        camera_depths=args.use_depth,
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

    problem_name = problem_name 
    
    env_args = {"type": 1,
                "env_name": env_name,
                "problem_name": problem_name,
                "bddl_file": bddl_file_name, 
                "env_kwargs": env_kwargs}

    grp.attrs["env_args"] = json.dumps(env_args)
    print(grp.attrs["env_args"])
    total_len = 0

    init_idx = 0
    
    demo_obs = rollout_demo(problem_name=problem_name,
                                env_kwargs=env_kwargs,
                                demos=demos,
                                demo_index=args.demo_index,
                                f=f,
                                args=args, 
                                init_idx=init_idx,
                                cap_idx=cap_index,)
    
    grp, total_len = save_hdf5_ep_grp(grp, args, demo_obs, init_idx, args.demo_index, total_len)

    grp.attrs["num_demos"] = 1
    grp.attrs["total"] = total_len

    h5py_f.close()        
    f.close()

    print("The created dataset is saved in the following path: ")
    print(hdf5_out_path)
    
    
if __name__ == "__main__":
    main()



