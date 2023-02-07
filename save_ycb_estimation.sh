#!/bin/bash
python demo.py --dataset ycb\
  --dataset_root_dir /media/user/433c5472-5bea-42d9-86c4-e0794e47477f/YCB_Video_Dataset\
  --max_iteration 30 \
  --visualization True\
  --w_o_Scene_occlusion False \
  --w_o_CPN False \
  --save_path results/ycb/ \
  --num_particles 300

