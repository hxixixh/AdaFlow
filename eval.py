import argparse
import hydra
import os 

from omegaconf import OmegaConf
from pathlib import Path

from adaflow.workspace.base_workspace import BaseWorkspace


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_inference_steps", 
        default=20,
        type=int,
    )
    parser.add_argument(
        "--eval_exp_dir", 
        default="",
        type=str,
    )
    parser.add_argument(
        "--sampling_method", 
        default=None, 
        type=str,
    )
    parser.add_argument(
        "--evaluate_mode", 
        default="rand_start",   # rand_start or fix_start
        type=str,
    )
    parser.add_argument(
        "--eta", 
        default=None, 
        type=float, 
    )

    args = parser.parse_args()
    
    if os.path.exists(args.eval_exp_dir): 
        experiment_dir = Path(args.eval_exp_dir) 
    else: 
        raise ValueError("Invalid eval_exp_dir")
    
    cfg = OmegaConf.load(os.path.join(experiment_dir, ".hydra", "config.yaml"))
    
    if "evaluate_mode" not in cfg.keys(): 
        cfg.evaluate_mode = args.evaluate_mode
    
    if cfg.name in ["train_adaflow_unet_image"]: 
        cfg.policy.sampling_method = args.sampling_method
        cfg.policy.eta = args.eta
        cfg.policy.num_inference_steps = args.num_inference_steps
        cfg.evaluate_mode = args.evaluate_mode
    
    if cfg.name in ["train_diffusion_unet_image"]: 
        cfg.policy.num_inference_steps = args.num_inference_steps

    cls = hydra.utils.get_class(cfg._target_)
    
    workspace: BaseWorkspace = cls(cfg, output_dir=experiment_dir)
        
    workspace.eval_only(output_dir=experiment_dir)

if __name__ == "__main__": 
    main()  