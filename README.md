# Particle_filter_approach_6D_pose_estimation

## Requirements
 * __CMake >= 3.17__
 * __Python >= 3.6__
 * __CUDA >= 10.0__     (We tested on CUDA 10.0 and 10.2)
 * __Pytorch >= 1.7.1__ (We tested on Pytorch 1.7.1 and 1.9)

We tested on Ubuntu 18.04.

## Installation

Change compile.sh line 5 to the glm library include path. This library can be downloaded from this [link](https://github.com/g-truc/glm).

    $ cd lib
    $ sh compile.sh
    $ cd ..
    $ conda env create -f requirements.yaml
    $ conda activate pf_with_cpn

## Preparing the datasets and toolbox
### YCB Video dataset
Download the YCB Video dataset by following the comments [here](https://github.com/yuxng/PoseCNN/issues/81) to your local datasets folder.
### Occluded LINEMOD dataset
Download the Occluded LINEMOD dataset from [BOP: Benchmark for 6D Object Pose Estimation](https://bop.felk.cvut.cz/datasets/) or you can directly download [here](https://ptak.felk.cvut.cz/6DB/public/bop_datasets/lmo_test_all.zip) to your local datasets folder.
### YCB Video toolbox
Download the YCB Video toolbox from [here](https://github.com/yuxng/YCB_Video_toolbox) to `<local path to 6D_pose_estimation_particle_filter directory>/CenterFindNet/YCB_Video_toolbox` directory.
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
### Runing the demo (for saving the estimated pose results)
Run `$ ./save_lmo_estimation.sh` for estimating on the Occluded LINEMOD, `$ ./save_ycb_estimation.sh` for estimating on the YCB Video dataset.

There are 4 options that you can fill out (You must fill out `--dataset_root_dir` as your datasets local directory.) :
 * dataset : `lmo` for Occluded LINEMOD, `ycb` for YCB Video
 * dataset_root_dir : <your local path to 6D_pose_estimation_particle_filter directory>/test/000002
 * save_path : The directory to save the estimated pose. ex) `results/lmo/`
 * visualization : If you don't want to watch how the prediction going on, set this `False`. Default is True.

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
