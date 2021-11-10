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

## Preparing the datasets and toolbox
### YCB Video dataset
Download the YCB Video dataset by following the comments [here](https://github.com/yuxng/PoseCNN/issues/81) to your local datasets folder.
### Occluded LINEMOD dataset
Download the Occluded LINEMOD dataset from [BOP: Benchmark for 6D Object Pose Estimation](https://bop.felk.cvut.cz/datasets/) or you can directly download [here](https://ptak.felk.cvut.cz/6DB/public/bop_datasets/lmo_test_all.zip) to your local datasets folder.
### YCB Video toolbox
Download the YCB Video toolbox from [here](https://github.com/yuxng/YCB_Video_toolbox) to `<local path to 6D_pose_estimation_particle_filter repo>/CenterFindNet` directory.
##### YCB Video and LIENMOD objects models can be cound in `<local path to 6D_pose_estimation_particle_filter repo>/models`

  $ conda activate pf_with_cpn
