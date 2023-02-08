# 6D Object Pose Estimation using a Particle Filter with Better Initialization

GiJae Lee, Jun-Sik Kim, Seungryong Kim, KangGeon Kim

This repository contains source codes for the paper "6D Object Pose Estimation using a Particle Filter with Better Initialization." (IEEE Access 2023)

## Requirements
 * __CMake >= 3.17__
 * __Python >= 3.6__
 * __CUDA >= 10.2__     (We tested on CUDA 10.2)
 * __Pytorch >= 1.5.0__ (We tested on Pytorch 1.5 and 1.7.1 and 1.9)

We tested this code on Ubuntu 18.04.

## Installation

Change lib/compile.sh line 5 to the glm library include path. This library can be downloaded from this [link](https://github.com/g-truc/glm).

    $ cd lib
    $ sh compile.sh
    $ cd ..
    $ conda env create -f requirements.yaml
    $ conda activate pf_with_cpn

Install python packages:

    $ pip install opencv-python transforms3d open3d scipy

## Preparing the datasets and toolbox
### YCB Video dataset
Download the YCB Video dataset by following the comments [here](https://github.com/yuxng/PoseCNN/issues/81) to your local datasets folder.
### Occluded LINEMOD dataset
Download the Occluded LINEMOD dataset from [BOP: Benchmark for 6D Object Pose Estimation](https://bop.felk.cvut.cz/datasets/) or you can directly download [here](https://ptak.felk.cvut.cz/6DB/public/bop_datasets/lmo_test_all.zip) to your local datasets folder.
### YCB Video toolbox
Download the YCB Video toolbox from [here](https://github.com/yuxng/YCB_Video_toolbox) to `<local path to 6D_pose_estimation_particle_filter directory>/CenterFindNet/YCB_Video_toolbox` directory. And unzip `results_PoseCNN_RSS2018.zip`.

    $ cd <local path to 6D_pose_estimation_particle_filter directory>/CenterFindNet/YCB_Video_toolbox
    $ unzip unzip results_PoseCNN_RSS2018.zip
    
## Demo for 6D pose estimation
You can evaluate the results after saving the estimated pose first.
### Runing the demo (for saving the estimated pose results)
Run `$ ./save_ycb_estimation.sh` for estimating on the YCB Video dataset.

There are 8 options that you can fill out (You must fill out `--dataset_root_dir` as your datasets local directory.) :
 * dataset (str) : `ycb` for YCB Video
 * dataset_root_dir (str) : `example: ~~/YCB_Video_Dataset`
 * save_path (str) : The directory to save the estimated pose. ex) `results/ycb/`
 * visualization (bool) : If you don't want to watch how the prediction going on, set this `False`. Default is True.
 * gaussian_std (str) : This is Gaussian standard diviation of Eq.(5) in the paper. Default is 0.1. You have to set this value for each class. `example: 0.1 0.2 0.3 0.15 (The number of classes is 4.)`
 * tau (str) : This is the start value of misalignment tolerance &tau;<sub>0</sub>. It is decreased from &tau;<sub>0</sub> to 0.1*&tau;<sub>0</sub>. Default is 0.1. You have to set this value for each class. `example: 0.1 0.2 0.3 0.15 (The number of classes is 4.)`
 * max_iteration (int) : This value is the maximum number of iterations for an object. Default is 20.
 * num_particles (int) : This is the number of particles. Default is 100.

There is an additional option related to the ablation study.
If you set `w_o_CPN` as `True`, you can get the result generated without CPN.
If you set `w_o_Scene_occlusion` as `True`, you can get the result generated without considering scene occlusion.
 * w_o_CPN (bool) : Default is False.
 * w_o_Scene_occlusion (bool) : Default is False.


### Evaluating on the saved results
Run `$ ./eval_ycb.sh` for estimating on the YCB Video dataset. This step has to be run after the saving pose results. Or you can run with `--save_path results/ycb/` in `eval_ycb.sh` line 5 for checking the performance of the our result.

## Demo for Centroid Prediction Network
### Training Centroid Prediction Network(CPN)
Run `$ ./train_CPN.sh` after filling out your local path of the YCB Video dataset in the argument of --dataset_root_dir of the sh file.
Weight file would be saved in: `<local path to 6D_pose_estimation_particle_filter directory>/CenterFindNet/trained_model/`

### Evaluating Centroid Prediction Network(CPN)
If you want to evaluate with own your trained model, add the line that argument of --model_path in the `eval_CPN_<dataset>.sh`.
Or you can run directly `$ ./eval_CPN_ycb.sh` for the YCB Video dataset, `$ ./eval_CPN_lmo.sh` for the Occluded LINEMOD.
If you run it with "--visualization True", the result of projection into the image would be shown on the window.
