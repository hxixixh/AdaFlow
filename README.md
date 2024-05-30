# AdaFlow: Imitation Learning with Variance-Adaptive Flow-Based Policies<br><sub>Official PyTorch Implementation</sub>

This repository provides the official PyTorch implementation for the paper [AdaFlow: Imitation Learning with Variance-Adaptive Flow-Based Policies](https://arxiv.org/abs/2402.04292). 

## Overview 

AdaFlow is a generative-model-based policy model for robotics learning, combining the expressiveness of Diffusion Policies with the inference speed of Behavior Cloning (BC). It offers fast action generation in nearly 1 step, facilitating real-time application in robotics.


AdaFlow is based on Rectified Flow, a recent advancement in generative modeling, that outperforms traditional diffusion models in terms of speed and generation quality. The effectiveness of Rectified Flow has been demonstrated in large-scale image generation tasks as in [Stable diffusion 3](https://arxiv.org/abs/2403.03206). For a deeper understanding of Rectified Flow, refer to this [repo](https://github.com/gnobitab/RectifiedFlow) and [an introductory tutorial](https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html). 


## Reproducing Paper Results

### 1. Toy 1D example

For a practical demonstration of AdaFlow, please explore the toy_1D example provided in the corresponding [Jupyter notebook](https://github.com/hxixixh/AdaFlow/tree/main/toy_1d/1d.ipynb) within the toy_1d folder. 

This example illustrates AdaFlow's ability to dynamically adjust generation steps based on the state-specific uncertainty. We also compare it with the Diffusion Policy (DDIM) and 1-Rectified Flow. 


### 2. Training on LIBERO

This section details the steps to train AdaFlow using the LIBERO and RoboMimic datasets, demonstrating the model's capabilities in a robotics context.


**1.** Environment Setup

Set up your Python environment by running the following commands:
```bash
conda create -n adaflow python=3.8.13
conda activate adaflow
pip install -r requirements.txt
```


**2.** Data Preparation

**2.1.** Install LIBERO: Follow the installation instructions in the [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO) and download LIBERO-100 Data. 

**2.2.** Link libero_90 data to datasets. 
```bash
cd adaflow
ln -s $PATH_TO_LIBERO/libero/datasets/libero_90 datasets
ln -s $PATH_TO_LIBERO/libero/libero/init_files datasets
```

**2.3** Convert the data and extract RGB images using the provided scripts. For a single GPU, adjust the script to process sequentially, or utilize a multi-GPU setup for parallel processing.

```bash
usage: 
# usage: 
./scripts/convert_libero_data.sh $path_to_libero_data $output_file_name $PATH_TO_LIBERO

# example
./scripts/convert_libero_data.sh datasets/libero/libero_90/KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet_demo.hdf5 KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet_demo.hdf5 /u/xixi/WorkSpace/LIBERO
```


**3.** Training Process

Train AdaFlow and a diffusion policy with the following commands:

```bash
# Examples:
# AdaFlow training
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
python train.py --config-name=train_adaflow_unet_image_workspace task=libero_image_abs task.dataset_type=KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet_demo

# Diffusion policy training
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
python train.py --config-name=train_diffusion_unet_ddim_image_workspace task=libero_image_abs task.dataset_type=KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet_demo
```

**4.** Evaluate Model Performance

Assess the success rate of each trained model using the evaluation scripts provided:

```bash
# Examples: 
# AdaFlow evaluation
eval_exp_dir="exps/outputs/2024.05.29/02.36.40_train_adaflow_unet_image_libero_image_KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet_demo"
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
python eval.py --eval_exp_dir=$eval_exp_dir --sampling_method="adaptive" --evaluate_mode="rand_start" --num_inference_steps=5 --eta=0.5

# Diffusion model evaluation
eval_exp_dir="exps/outputs/2024.05.29/02.39.34_train_diffusion_unet_image_libero_image_KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet_demo"
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
python eval.py --eval_exp_dir=$eval_exp_dir --evaluate_mode="rand_start" --num_inference_steps=5
```

## Citation

If you find this implementation helpful, please consider citing it as follows:
```
@article{hu2024adaflow,
  title={AdaFlow: Imitation Learning with Variance-Adaptive Flow-Based Policies},
  author={Hu, Xixi and Liu, Bo and Liu, Xingchao and Liu, Qiang},
  journal={arXiv preprint arXiv:2402.04292},
  year={2024}
}

@article{liu2022flow,
  title={Flow straight and fast: Learning to generate and transfer data with rectified flow},
  author={Liu, Xingchao and Gong, Chengyue and Liu, Qiang},
  journal={arXiv preprint arXiv:2209.03003},
  year={2022}
}
```

## Thanks
A Large portion of this codebase is built upon [Diffusion Policy](https://github.com/real-stanford/diffusion_policy).






