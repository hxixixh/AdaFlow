from typing import Dict
import torch

from adaflow.model.common.normalizer import LinearNormalizer
from adaflow.policy.base_image_policy import BaseImagePolicy
from adaflow.model.diffusion.conditional_unet1d import ConditionalUnet1DwithVarianceEstimation
from adaflow.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from adaflow.common.pytorch_util import dict_apply

class AdaflowUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            sampling_method="euler", 
            eta=0.1, 
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1DwithVarianceEstimation(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        self.num_inference_steps = num_inference_steps

        self.freeze_rf = False
        self.sampling_method = sampling_method
        self.eta = eta
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            local_cond=None, global_cond=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):

        noise = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=None)

        z = noise.detach().clone()
        bsz = noise.shape[0]

        sqrt_var_traj = []
        for i in range(self.num_inference_steps):
            t = torch.ones((bsz)).to(noise.device) * i / self.num_inference_steps
            v_pred, log_sqrt_var_pred = self.model(z, t*self.kwargs["pos_emb_scale"], local_cond=local_cond, global_cond=global_cond)
            z = z.detach().clone() + v_pred * (1./self.num_inference_steps)
            sqrt_var_traj.append(log_sqrt_var_pred.exp() ** 2)

        sqrt_var_traj = torch.stack(sqrt_var_traj, dim=-1)
        return z, sqrt_var_traj

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.sampling_method == "euler": 
            return self.predict_action_euler(obs_dict)
        elif self.sampling_method == "adaptive": 
            return self.predict_action_adaptive(obs_dict)

    def predict_action_euler(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample, sqrt_var_traj = self.conditional_sample(
            cond_data, 
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred, 
            "sqrt_var_traj": sqrt_var_traj
        }
        return result
    
    def predict_action_adaptive(self, obs_dict: Dict[str, torch.Tensor], var_comp_method="mean") -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            raise NotImplementedError("not implemented yet")
        
        gen_traj = []
        
        noise = torch.randn(
            size=cond_data.shape, 
            dtype=cond_data.dtype,
            device=cond_data.device,
            generator=None)
        z = noise.detach().clone()

        gen_traj.append(z)
        
        valid_action = torch.zeros_like(noise)
        valid_action_found = torch.zeros(B).bool()
        num_steps_taken = torch.zeros(B)

        current_t = torch.zeros(B).to(noise.device)

        step_traj = [current_t]
        var_traj = []

        for i in range(self.num_inference_steps): 
            t = current_t

            v_pred, log_sqrt_var_pred = self.model(z, t*self.kwargs["pos_emb_scale"], local_cond=local_cond, global_cond=global_cond)
            
            var_pred = log_sqrt_var_pred.exp() ** 2

            step_size = torch.max(self.eta / var_pred.sqrt(), torch.tensor([1 / self.num_inference_steps]).to(self.device))
            step_size = torch.min(step_size, 1 - current_t)
            z = z.detach().clone() + v_pred * step_size[:, None, None]

            gen_traj.append(z)
            
            current_t = current_t + step_size
            step_traj.append(current_t)
            var_traj.append(var_pred)

            if current_t.max() >= 1: 
                mask = (current_t >= 1).cpu() & (~valid_action_found)
                valid_action[mask] = z[mask].detach().clone()
                valid_action_found[mask] = True
                num_steps_taken[mask] = i+1
            
            if valid_action_found.all(): 
                break

        var_traj = torch.stack(var_traj, dim=-1)
        step_traj = torch.stack(step_traj, dim=-1)
        
        # unnormalize prediction
        naction_pred = z[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        gen_traj = torch.stack(gen_traj, dim=1)
        gen_traj = self.normalizer['action'].unnormalize(gen_traj)
        
        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        var_traj = var_traj[:,start:end]
        step_traj = step_traj[:,start:end]

        assert valid_action_found.sum() == B

        if var_comp_method == "max": 
            var = var_traj.max(dim=-1)[0]
        elif var_comp_method == "mean":
            var = var_traj.mean(dim=-1)
        
        result = {
            'action': action,
            'action_pred': action_pred, 
            'nfe': num_steps_taken, 
            "variance": var, 
            "step_traj": step_traj,
            'gen_traj': gen_traj
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_train_tuple(self, z0, z1): 
        t = torch.rand(z1.shape[0], 1, 1).to(z1.device)
        z_t =  t * z1 + (1.-t) * z0
        target = z1 - z0 
        return z_t, t, target

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        
        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        z_t, t, target = self.get_train_tuple(z0=noise, z1=nactions)
        
        # compute loss mask
        velocity_pred, log_sqrt_var_pred = self.model(z_t, t.squeeze()*self.kwargs["pos_emb_scale"], local_cond=local_cond, global_cond=global_cond, freeze_rf=self.freeze_rf )
        
        error = (target - velocity_pred).pow(2).sum(dim=(-1, -2))
        if self.freeze_rf: 
            error = error.detach()
        loss = 1 / (2. * torch.exp(2 * log_sqrt_var_pred)) * error + log_sqrt_var_pred

        loss = loss.mean()
        return loss