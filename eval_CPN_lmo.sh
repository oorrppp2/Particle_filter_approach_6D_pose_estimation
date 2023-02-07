#!/bin/bash
python CenterFindNet/eval_CPN.py --dataset lmo\
  --dataset_root_dir /datasets/lmo_test_all/test/000002\
  --model_path /trained_model/CPN_model_91_0.00023821471899932882.pth\
  --visualization True
