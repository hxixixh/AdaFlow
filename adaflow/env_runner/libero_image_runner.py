import collections
import h5py
import json
import math
import numpy as np
import os
import pathlib
import torch
import tqdm
import wandb
import wandb.sdk.data_types.video as wv

import robomimic.utils.obs_utils as ObsUtils

from adaflow.common.pytorch_util import dict_apply
from adaflow.env.libero.libero_image_wrapper import LIBEROImageWrapper
from adaflow.env_runner.base_image_runner import BaseImageRunner
from adaflow.gym_util.multistep_wrapper import LIBEROMultiStepWrapper
from adaflow.gym_util.video_recording_wrapper import VideoRecorder
from adaflow.model.common.rotation_transformer import RotationTransformer
from adaflow.policy.base_image_policy import BaseImagePolicy

from libero.libero.envs import DummyVectorEnv 
from adaflow.libero_extension.envs import OffScreenRenderEnv


class LIBEROImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            dataset_type,
            shape_meta:dict,
            task_emb=None, 
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None, 
            n_episodes=None, 
            init_states_folder=None, 
            seed=None, 
            camera_heights=84, 
            camera_widths=84,
            camera_view="agentview", 
        ):
        super().__init__(output_dir)

        dataset_path = os.path.expanduser(dataset_path)

        # initialize obs normalizer
        modality_mapping = collections.defaultdict(list)
        for key, attr in shape_meta['obs'].items():
            modality_mapping[attr.get('type', 'low_dim')].append(key)
        ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
        
        # get fixed init states to control the experiment randomness
        init_states_path = os.path.join(init_states_folder, "{}.pruned_init".format(dataset_type.replace("_demo", "")))
        init_states = torch.load(init_states_path)

        if n_episodes is not None:
            init_states = init_states[:n_episodes]

        f = h5py.File(dataset_path, "r")

        problem_info = json.loads(f["data"].attrs["problem_info"])
        problem_name = problem_info["problem_name"]

        bddl_file_name = f["data"].attrs["bddl_file_name"]
        if not os.path.exists(bddl_file_name): 
            task_group = str(dataset_path).split("/")[-2]
            hdf5_file_name = str(dataset_path).split("/")[-1]
            bddl_file_name = os.path.join("libero/libero/bddl_files", task_group, hdf5_file_name.replace("_abs", "").replace("_demo.hdf5", ".bddl"))
        
        env_args = {
            "bddl_file_name": bddl_file_name,
            "camera_heights": camera_heights,
            "camera_widths": camera_widths,
            "camera_names": [camera_view, "robot0_eye_in_hand"],
            "control_delta": not abs_action,  
        }

        self.video_recorders = []
        self.video_file_names = []
        for i in range(n_envs): 
            filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
            filename.parent.mkdir(parents=False, exist_ok=True)
            filename = str(filename)
            self.video_file_names.append(filename)

            self.video_recorders.append(
                VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    )
            )
            
        def env_fn(): 
            libero_env = OffScreenRenderEnv(**env_args)
            return LIBEROMultiStepWrapper(
                LIBEROImageWrapper(
                        env=libero_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ), 
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps
            )
        
        env = DummyVectorEnv([env_fn for _ in range(n_envs)])

        rotation_transformer = None
        if abs_action:
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        if task_emb is not None: 
            task_emb = task_emb.unsqueeze(0).repeat(n_envs, 1)

        self.dataset_path = dataset_path
        self.camera_heights = camera_heights
        self.camera_widths = camera_widths
        self.output_dir = output_dir
        self.env = env
        self.task_emb = task_emb
        self.init_states = init_states
        self.n_envs = n_envs
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.seed = seed
        self.problem_name = problem_name
        self.env_args = env_args

    def raw_obs_to_array_obs(self, obs): 
        data = {
            "agentview_image": [], 
            "robot0_eye_in_hand_image": [], 
            "robot0_eef_pos": [], 
            "robot0_eef_quat": [], 
            "robot0_gripper_qpos": [], 
        }
        for i in range(self.n_envs): 
            data["agentview_image"].append(ObsUtils.process_obs(obs[i]["agentview_image"], obs_key="agentview_image"))
            data["robot0_eye_in_hand_image"].append(ObsUtils.process_obs(obs[i]["robot0_eye_in_hand_image"], obs_key="robot0_eye_in_hand_image"))
            data["robot0_eef_pos"].append(ObsUtils.process_obs(obs[i]["robot0_eef_pos"], obs_key="robot0_eef_pos"))
            data["robot0_eef_quat"].append(ObsUtils.process_obs(obs[i]["robot0_eef_quat"], obs_key="robot0_eef_quat")) 
            data["robot0_gripper_qpos"].append(ObsUtils.process_obs(obs[i]["robot0_gripper_qpos"], obs_key="robot0_gripper_qpos"))

        for key in data: 
            data[key] = np.stack(data[key], axis=0)

        return data

    def run(self, policy: BaseImagePolicy, fix_start=False):
        device = policy.device

        if self.task_emb is not None:
            self.task_emb = self.task_emb.to(device)

        env = self.env
        
        # plan for rollout
        n_inits = self.init_states.shape[0]
        n_envs = self.n_envs
        n_chunks = math.ceil(n_inits / n_envs)

        num_success = 0

        step_list = []
        
        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)

            if fix_start: 
                this_global_slice = [0] * n_envs
            else: 
                this_global_slice = slice(start, end)
            
            env.seed(self.seed)
            env.reset()
            
            this_init_states = self.init_states[this_global_slice]

            dones = [False] * n_envs
            dones_step = [self.max_steps] * n_envs
  
            obs = env.set_init_state(this_init_states)

            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {self.problem_name} Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)

            for i, video_recorder in enumerate(self.video_recorders):
                video_recorder.start(self.video_file_names[i])
            
            step_list_chunk_i = []

            for steps in range(self.max_steps // self.n_action_steps):
                obs = self.raw_obs_to_array_obs(obs)
                
                np_obs_dict = dict(obs)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                
                # run policy
                with torch.no_grad():
                    if self.task_emb is not None: 
                        action_dict = policy.predict_action(obs_dict, self.task_emb)
                    else: 
                        policy.sampling_method = "adaptive"
                        action_dict = policy.predict_action(obs_dict)
                
                if "nfe" in action_dict.keys(): 
                    nfe = action_dict["nfe"]
                    step_list_chunk_i.append(nfe)
                    
                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)
                
                obs, reward, done, info = env.step(env_action)

                for i, video_recorder in enumerate(self.video_recorders): 
                    video_recorder.write_frame(obs[i]["agentview_image"][-1][::-1])

                # check whether succeed
                for k in range(self.n_envs):
                    dones[k] = dones[k] or done[k]
                    if done[k] and dones_step[k] == self.max_steps: 
                        dones_step[k] = steps * self.n_action_steps

                if all(dones): break

                # update pbar
                pbar.update(env_action.shape[1])

            for k in range(self.n_envs):
                if chunk_idx * self.n_envs + k < n_inits:
                    num_success += int(dones[k])

            pbar.close()

            # collect data for this round
            for video_recorder in self.video_recorders: 
                video_recorder.stop()

            if len(step_list_chunk_i) > 0: 
                step_list_chunk_i = torch.stack(step_list_chunk_i, dim=-1).repeat_interleave(self.n_action_steps, dim=-1)
                step_list_chunk_i = step_list_chunk_i.tolist()
                
                step_list_chunk_i = [np.mean(x[:dones_step[i]]) for i, x in enumerate(step_list_chunk_i)]
                step_list.extend(step_list_chunk_i)

        # clear out video buffer
        _ = env.reset()
        
        # log
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.seed
            prefix = "test/"

        # visualize sim
        for i in range(n_envs): 
            video_path = self.video_file_names[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        
        log_data["test/mean_score"] = num_success / n_inits
            
        if len(step_list) > 0:
            log_data["test/avg_step"] = np.mean(step_list)
        
        return log_data
    
    def close(self): 
        self.env.close()

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction

    