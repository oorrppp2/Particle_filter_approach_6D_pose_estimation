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
  
code end

  $ conda activate pf_with_cpn
