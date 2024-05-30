import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
import pathlib

from omegaconf import OmegaConf

from adaflow.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'adaflow','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    
    cls = hydra.utils.get_class(cfg._target_)

    if cfg.training.resume_dir is not None: 
        workspace: BaseWorkspace = cls(cfg, output_dir=cfg.training.resume_dir)
    else: 
        workspace: BaseWorkspace = cls(cfg)
    
    workspace.run()

if __name__ == "__main__":
    main()