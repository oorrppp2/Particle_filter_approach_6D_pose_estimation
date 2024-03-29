# 6D Object Pose Estimation using a Particle Filter with Better Initialization

## Requirements
 * __CMake >= 3.17__
 * __Python >= 3.6__
 * __CUDA >= 10.2__     (We tested on CUDA 10.2)
 * __Pytorch >= 1.5.0__ (We tested on Pytorch 1.5 and 1.7.1 and 1.9)

It currently works on Ubuntu 18.04. You can try this code using [docker](https://www.docker.com/) if your OS is not Ubuntu 18.04.

## Installation

Change compile.sh line 5 to the glm library include path. This library can be downloaded from this [link](https://github.com/g-truc/glm).

    $ cd lib
    $ sh compile.sh
    $ cd ..
    $ conda env create -f requirements.yaml
    $ conda activate pf_with_cpn

Or you can just run below in your own anaconda environment.

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
    
### PVNet segmentation results
We generated the segmentation results by using the segmentation network, and pre-trained model from [PVNet](https://github.com/zju3dv/pvnet) in off-line.
To evaluate the Occluded LINEMOD with PVNet segmentation mask, those segmentation results are required.
It can downloaded from [here](https://drive.google.com/file/d/1u5Mtd8vVIa0f6Fo6EbVglVWhJeo8onPw/view?usp=sharing) to downloaded Occluded LINEMOD dataset directory. `<local path to Occluded LINEMOD dataset>/test/000002/labels_pvnet`
### Mask R-CNN segmentation results
We generated the segmentation results by using the [Mask R-CNN](https://github.com/matterport/Mask_RCNN) network, and pre-trained model from [Pix2Pose](https://github.com/kirumang/Pix2Pose) in off-line.
To evaluate the Occluded LINEMOD with Mask R-CNN segmentation mask, those segmentation results are required.
It can downloaded from [here](https://drive.google.com/file/d/1eNNI85d3VU2GKBQfg4aqv0bbQGvkIXJU/view?usp=sharing) to downloaded Occluded LINEMOD dataset directory. `<local path to Occluded LINEMOD dataset>/test/000002/labels_mask_rcnn`
##### YCB Video and LIENMOD objects models can be cound in `<local path to 6D_pose_estimation_particle_filter repo>/models`

## Demo for 6D pose estimation
You can evaluate the results after saving the estimated pose first.
### Runing the demo (for saving the estimated pose results)
Run `$ ./save_lmo_estimation.sh` for estimating on the Occluded LINEMOD, `$ ./save_ycb_estimation.sh` for estimating on the YCB Video dataset.

There are 8 options that you can fill out (You must fill out `--dataset_root_dir` as your datasets local directory.) :
 * dataset(str) : `lmo` for Occluded LINEMOD, `ycb` for YCB Video
 * dataset_root_dir(str) : <your local path to 6D_pose_estimation_particle_filter directory>/test/000002
 * save_path(str) : The directory to save the estimated pose. ex) `results/lmo/`
 * visualization(bool) : If you don't want to watch how the prediction going on, set this `False`. Default is True.
 * gaussian_std(float) : This is Gaussian standard diviation of Eq.(5) in the paper. Default is 0.1.
 * max_iteration(int) : This value is the maximum number of iterations for an object. Default is 20.
 * tau(float) : This is the start value of misalignment tolerance &tau;<sub>0</sub>. It is decreased from &tau;<sub>0</sub> to 0.1*&tau;<sub>0</sub>. Default is 0.1.
 * num_particles(int) : This is the number of particles. Default is 180.

There is an additional option of choosing the input mask type for the demo of Occluded LINEMOD. We evaluated the 6D pose results on each input segmentation mask of Mask R-CNN and PVNet.
 * input_mask : Choose between `pvnet` and `mask_rcnn` as input mask. Default is `pvnet`.

### Evaluating on the saved results
Run `$ ./eval_lmo.sh` for estimating on the Occluded LINEMOD, `$ ./eval_ycb.sh` for estimating on the YCB Video dataset. This step has to be run after the saving pose results. Or you can run with `--save_path results/ycb_multi_init_trans/` in `eval_ycb.sh` line 5 for checking the performance of the our result.

We contain the results of experiments recorded in our paper. Replace the argument of --dataset, --save_path in the file to below.

    (Ours - Multi initial translation) :    --dataset ycb --save_path results/ycb_multi_init_trans/
    (Ours - 180 particles) :                --dataset ycb --save_path results/ycb_180_particles/
    (Ours with Mask R-CNN input) :          --dataset lmo --save_path results/lmo_mask_RCNN_input/
    (Ours with PVNet mask input) :          --dataset lmo --save_path results/lmo_PVNet_input/ in our paper.

## Demo for Centroid Prediction Network
### Training Centroid Prediction Network(CPN)
Run `$ ./train_CPN.sh` after filling out your local path of the YCB Video dataset in the argument of --dataset_root_dir of the sh file.
Weight file would be saved in: `<local path to 6D_pose_estimation_particle_filter directory>/CenterFindNet/trained_model/`

### Evaluating Centroid Prediction Network(CPN)
If you want to evaluate with own your trained model, add the line that argument of --model_path in the `eval_CPN_<dataset>.sh`.
Or you can run directly `$ ./eval_CPN_ycb.sh` for the YCB Video dataset, `$ ./eval_CPN_lmo.sh` for the Occluded LINEMOD.
If you run it with "--visualization True", the result of projection into the image would be shown on the window.


## How to make predefined grasp poses
### Convert obj file to pcd file

    $ cd lib/obj2pcd && mkdir build && cd build && cmake .. && make
    $ ./obj2pcd <obj file path>

### Predefine grasp pose
Install [graspnetAPI](https://github.com/graspnet/graspnetAPI)
Write your code (grasp pose x,y,z,roll,pitch,yaw ...) in function `vis_predefined_grasps`

    $ python grasp_pose_predefined.py <pcd file path>