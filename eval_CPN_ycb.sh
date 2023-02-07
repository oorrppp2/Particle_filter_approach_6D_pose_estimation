#!/bin/bash
python CenterFindNet/eval_CPN.py --dataset ycb\
  --dataset_root_dir /media/user/433c5472-5bea-42d9-86c4-e0794e47477f/YCB_Video_Dataset\
  --model_path /trained_model/CPN_model_91_0.00023821471899932882.pth\
  --visualization True
