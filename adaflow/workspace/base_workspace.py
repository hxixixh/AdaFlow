import copy
import dill
import gc
import hydra
import numpy as np
import os
import pathlib
import threading
import torch

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from typing import Optional

from adaflow.dataset.base_dataset import BaseImageDataset
from adaflow.env_runner.base_image_runner import BaseImageRunner


class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str]=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    
    def run(self):
        """
        Create any resource shouldn't be serialized as local variables
        """
        pass

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=True):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')

    def get_best_checkpoint_path(self): 
        ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        ckpts = os.listdir(ckpt_dir)
        ckpts = [ckpt for ckpt in ckpts if ckpt.endswith(".ckpt") and "latest" not in ckpt]
        best_score = 0.0
        for ckpt in ckpts:
            score = float(ckpt.split("=")[-1][:-5])
            if score > best_score:
                best_score = score
                best_ckpt = ckpt
        return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                try: 
                    if list(value.keys())[0].startswith('module.'):
                        value = {k.replace('module.', ''): v for k, v in value.items()}
                    self.__dict__[key].load_state_dict(value, **kwargs)
                except ValueError as e:
                    print(f"Failed to load {key} due to {e}")
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance
    
    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)

    def eval_only(self, output_dir=None): 
        cfg = copy.deepcopy(self.cfg)

        # configure env
        eval_result_output_dir = self.get_evaluate_only_dir()
        os.makedirs(eval_result_output_dir, exist_ok=True)

        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=eval_result_output_dir)
        assert isinstance(env_runner, BaseImageRunner)
        
        ckpt_paths = os.listdir(os.path.join(output_dir, "checkpoints"))
        
        for ckpt_path in ckpt_paths: 
            if ckpt_path.endswith(".ckpt") and "latest" not in ckpt_path: 
                pass
            else: 
                continue
            
            print("Evaluating checkpoint: ", ckpt_path)
            self.load_checkpoint(path=os.path.join(self.output_dir, "checkpoints", ckpt_path))
            self._output_dir = output_dir
            
            # device transfer
            device = torch.device(cfg.training.device)
            self.model.to(device)
            if self.ema_model is not None:
                self.ema_model.to(device)

            policy = self.model
            if cfg.training.use_ema: 
                policy = self.ema_model
            policy.eval()


            if cfg.evaluate_mode == "rand_start": 
                runner_log = env_runner.run(policy, fix_start=False)
            elif cfg.evaluate_mode == "fix_start": 
                runner_log = env_runner.run(policy, fix_start=True)
            else: 
                raise ValueError("Invalid evaluate_mode")

            print("-------------------")
            print(runner_log["test/mean_score"])
            if "test/avg_step" in runner_log.keys():
                print(runner_log["test/avg_step"])
            print("-------------------")
            
            # write to txt file
            with open(os.path.join(eval_result_output_dir, "success_rate_eval.txt"), "a") as f: 
                f.write("{}: {}\n".format(ckpt_path, runner_log["test/mean_score"]))

            if "test/avg_step" in runner_log.keys(): 
                with open(os.path.join(eval_result_output_dir, "avg_step_eval.txt"), "a") as f: 
                    f.write("{}: {}\n".format(ckpt_path, runner_log["test/avg_step"]))
        
        env_runner.close()
        gc.collect()
        
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
        

def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)
