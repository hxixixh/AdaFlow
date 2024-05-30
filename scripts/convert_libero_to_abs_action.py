if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import click
import collections
import copy
import h5py
import json
import numpy as np
import os
import pathlib
import pickle
import shutil
import subprocess
import tempfile

from scipy.spatial.transform import Rotation
from tqdm import tqdm

import libero.libero.utils.utils as chiliocosm_utils
from libero.libero.envs import TASK_MAPPING


class LiberoAbsoluteActionConverter:
    def __init__(self, dataset_path, path_to_libero_lib, algo_name='bc'):
        f = h5py.File(dataset_path, "r")

        env_name = f["data"].attrs["env_name"]
        env_args = f["data"].attrs["env_args"]
        env_kwargs = json.loads(env_args)['env_kwargs']

        problem_info = json.loads(f["data"].attrs["problem_info"])
        problem_name = problem_info["problem_name"]

        bddl_file_name = f["data"].attrs["bddl_file_name"]
        if not os.path.exists(bddl_file_name): 
            task_group = str(dataset_path).split("/")[-2]
            hdf5_file_name = str(dataset_path).split("/")[-1]
            bddl_file_name = os.path.join(path_to_libero_lib, "libero/libero/bddl_files", task_group, hdf5_file_name.replace("_demo.hdf5", ".bddl"))
        chiliocosm_utils.update_env_kwargs(
            env_kwargs,
            bddl_file_name=bddl_file_name,
            scene_xml=None,
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
            camera_heights=84,
            camera_widths=84,
            camera_segmentations=None, 
        )

        env_args = {"type": 1,
                "env_name": env_name,
                "problem_name": problem_name,
                "bddl_file": bddl_file_name, 
                "env_kwargs": env_kwargs}
        
        env = TASK_MAPPING[problem_name](
            **env_kwargs,
        )

        abs_env_kwargs = copy.deepcopy(env_kwargs)
        abs_env_kwargs["controller_configs"]["control_delta"] = False
        abs_env = TASK_MAPPING[problem_name](
            **abs_env_kwargs,
        )

        self.env = env
        self.abs_env = abs_env
        self.file = f
        
    def __len__(self):
        return len(self.file['data'])

    def convert_actions(self, 
            states: np.ndarray, 
            actions: np.ndarray) -> np.ndarray:
        """
        Given state and delta action sequence
        generate equivalent goal position and orientation for each step
        keep the original gripper action intact.
        """
        # in case of multi robot
        # reshape (N,14) to (N,2,7)
        # or (N,7) to (N,1,7)
        stacked_actions = actions.reshape(*actions.shape[:-1],-1,7)

        env = self.env
        # generate abs actions
        action_goal_pos = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        action_goal_ori = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        action_gripper = stacked_actions[...,[-1]]

        env.sim.reset()        
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()
        model_xml = env.sim.model.get_xml()

        for j, action in tqdm(enumerate(actions), total=len(actions)):
            obs, reward, done, info = env.step(action)

            # taken from robot_env.py L#454
            for idx, robot in enumerate(env.robots):
                # run controller goal generator
                robot.control(stacked_actions[j,idx], policy_step=True)
            
                # read pos and ori from robots
                controller = robot.controller
                action_goal_pos[j,idx] = controller.goal_pos
                action_goal_ori[j,idx] = Rotation.from_matrix(
                    controller.goal_ori).as_rotvec()

            if j == len(actions) - 1: 
                obs, reward, done, info = env.step(action)
            else: 
                env.reset_from_xml_string(model_xml)
                env.sim.reset()        
                env.sim.set_state_from_flattened(states[j+1])
                env.sim.forward()

                env._update_observables(force=True)
                obs = env._get_observations()

        stacked_abs_actions = np.concatenate([
            action_goal_pos,
            action_goal_ori,
            action_gripper
        ], axis=-1)
        abs_actions = stacked_abs_actions.reshape(actions.shape)
        return abs_actions

    def convert_idx(self, idx):
        file = self.file
        demo = file[f'data/demo_{idx}']
        # input
        states = demo['states'][:]
        actions = demo['actions'][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)
        self.env.close()
        self.abs_env.close()
        return abs_actions

    def convert_and_eval_idx(self, idx):
        env = self.env
        abs_env = self.abs_env
        file = self.file
        # first step have high error for some reason, not representative
        eval_skip_steps = 1

        demo = file[f'data/demo_{idx}']
        # input
        states = demo['states'][:]
        actions = demo['actions'][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)
        
        # verify
        robot0_eef_pos = demo['obs']['robot0_eef_pos'][:]
        robot0_eef_quat = demo['obs']['robot0_eef_quat'][:]

        delta_error_info = self.evaluate_rollout_error(
            env, states, actions, robot0_eef_pos, robot0_eef_quat, 
            metric_skip_steps=eval_skip_steps)
        abs_error_info = self.evaluate_rollout_error(
            abs_env, states, abs_actions, robot0_eef_pos, robot0_eef_quat,
            metric_skip_steps=eval_skip_steps)
        
        self.env.close()
        self.abs_env.close()

        info = {
            'delta_max_error': delta_error_info,
            'abs_max_error': abs_error_info
        }
        return abs_actions, info

    @staticmethod
    def evaluate_rollout_error(env, 
            states, actions, 
            robot0_eef_pos, 
            robot0_eef_quat, 
            metric_skip_steps=1):
        # first step have high error for some reason, not representative

        # evaluate abs actions
        rollout_next_states = list()
        rollout_next_eef_pos = list()
        rollout_next_eef_quat = list()

        env.sim.reset()        
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()
        model_xml = env.sim.model.get_xml()

        for i in tqdm(range(len(states))):
            env.reset_from_xml_string(model_xml)
            env.sim.reset()        
            env.sim.set_state_from_flattened(states[i])
            env.sim.forward()
            obs, reward, done, info = env.step(actions[i])
            
            rollout_next_states.append(env.sim.get_state().flatten())
            rollout_next_eef_pos.append(obs['robot0_eef_pos'])
            rollout_next_eef_quat.append(obs['robot0_eef_quat'])
            
        rollout_next_states = np.array(rollout_next_states)
        rollout_next_eef_pos = np.array(rollout_next_eef_pos)
        rollout_next_eef_quat = np.array(rollout_next_eef_quat)

        next_state_diff = states[1:] - rollout_next_states[:-1]
        max_next_state_diff = np.max(np.abs(next_state_diff[metric_skip_steps:]))

        next_eef_pos_diff = robot0_eef_pos[1:] - rollout_next_eef_pos[:-1]
        next_eef_pos_dist = np.linalg.norm(next_eef_pos_diff, axis=-1)
        max_next_eef_pos_dist = next_eef_pos_dist[metric_skip_steps:].max()

        next_eef_rot_diff = Rotation.from_quat(robot0_eef_quat[1:]) \
            * Rotation.from_quat(rollout_next_eef_quat[:-1]).inv()
        next_eef_rot_dist = next_eef_rot_diff.magnitude()
        max_next_eef_rot_dist = next_eef_rot_dist[metric_skip_steps:].max()
        
        info = {
            'state': max_next_state_diff,
            'pos': max_next_eef_pos_dist,
            'rot': max_next_eef_rot_dist
        }
        return info


@click.command()
@click.option('-i', '--input', required=True, help='input hdf5 path')
@click.option('-l', '--path_to_libero_lib', required=True, help='path to libero lib')
@click.option('-o', '--output', required=True, help='output hdf5 path. Parent directory must exist')
@click.option('-e', '--eval_dir', default=None, help='directory to output evaluation metrics')
def main(input, path_to_libero_lib, output, eval_dir):
    # process inputs
    input = pathlib.Path(input).expanduser()
    assert input.is_file()
    output = pathlib.Path(output).expanduser()
    os.makedirs(output.parent, exist_ok=True)
    assert output.parent.is_dir()
    assert not output.is_dir()
    
    do_eval = False
    if eval_dir is not None:
        eval_dir = pathlib.Path(eval_dir).expanduser()
        os.makedirs(eval_dir.parent, exist_ok=True)
        assert eval_dir.parent.exists()
        do_eval = True

    converter = LiberoAbsoluteActionConverter(input, path_to_libero_lib)
    converter.env.close()
    converter.abs_env.close()

    # List to hold the subprocesses
    processes = []
    temp_files = []

    for i in tqdm(range(len(converter))): 
        # Create a temporary file
        camera_utils
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_files.append(tmp_file.name)

        p = subprocess.Popen(['python', "scripts/convert_libero_to_abs_action_called.py",
                              '--input', str(input),
                              '--path_to_libero_lib', str(path_to_libero_lib), 
                              '--output', str(tmp_file.name),
                              '--demo_index', str(i),
                              '--gpu_id', str(i % 8)])  # This assigns GPU in a round-robin fashion
        processes.append(p)
        
    # Wait until all processes finish
    for p in processes:
        p.wait()
        
    # Read and concat tmp abs actions
    results = []
    for file_name in temp_files:
        with open(file_name, 'rb') as f:
            data = np.load(f)
            abs_actions = data['abs_actions']
            results.append(abs_actions)

    # save output
    print('Copying hdf5')
    shutil.copy(str(input), str(output))

    # modify action
    with h5py.File(output, 'r+') as out_file:
        for i in tqdm(range(len(converter)), desc="Writing to output"):
            abs_actions= results[i]
            demo = out_file[f'data/demo_{i}']
            demo['actions'][:] = abs_actions

    # save eval
    if do_eval:
        eval_dir.mkdir(parents=False, exist_ok=True)

        print("Writing error_stats.pkl")
        infos = [info for _, info in results]
        pickle.dump(infos, eval_dir.joinpath('error_stats.pkl').open('wb'))

        print("Generating visualization")
        metrics = ['pos', 'rot']
        metrics_dicts = dict()
        for m in metrics:
            metrics_dicts[m] = collections.defaultdict(list)

        for i in range(len(infos)):
            info = infos[i]
            for k, v in info.items():
                for m in metrics:
                    metrics_dicts[m][k].append(v[m])

        from matplotlib import pyplot as plt
        plt.switch_backend('PDF')

        fig, ax = plt.subplots(1, len(metrics))
        for i in range(len(metrics)):
            axis = ax[i]
            data = metrics_dicts[metrics[i]]
            for key, value in data.items():
                axis.plot(value, label=key)
            axis.legend()
            axis.set_title(metrics[i])
        fig.set_size_inches(10,4)
        fig.savefig(str(eval_dir.joinpath('error_stats.pdf')))
        fig.savefig(str(eval_dir.joinpath('error_stats.png')))



if __name__ == "__main__":
    main()