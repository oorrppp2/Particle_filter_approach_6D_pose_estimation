"""
  Calulate accuracy
"""

import os
import sys
libpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(libpath)

from lib.utils import quaternion_matrix, add, adi, compute_pose_metrics, calc_pts_diameter, draw_object, str2bool, load_scene_gt
import numpy as np
import scipy.io as scio
import cv2
from transforms3d.quaternions import axangle2quat, mat2quat, qmult, qinverse
from transforms3d.euler import quat2euler, mat2euler, euler2quat
import open3d as o3d
import random
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='', help='dataset')
parser.add_argument('--dataset_root_dir', type=str, default='', help='dataset root dir')
parser.add_argument('--save_path', type=str, default='',  help='save results path')
parser.add_argument('--visualization', type=str2bool, default=True,  help='visualization')
opt = parser.parse_args()

if opt.dataset == 'ycb':
    cam_cx = 312.9869
    cam_cy = 241.3109
    cam_fx = 1066.778
    cam_fy = 1067.487
    cam_scale = 0.0001

    ycb_toolbox_dir = libpath + '/CenterFindNet/YCB_Video_toolbox'
    dataset_config_dir = libpath + '/CenterFindNet/datasets/dataset_config'
    cad_model_root_dir = libpath + '/models/ycb/'

    models = [
        "002_master_chef_can",      # 1
        "003_cracker_box",          # 2
        "004_sugar_box",            # 3
        "005_tomato_soup_can",      # 4
        "006_mustard_bottle",       # 5
        "007_tuna_fish_can",        # 6
        "008_pudding_box",          # 7
        "009_gelatin_box",          # 8
        "010_potted_meat_can",      # 9
        "011_banana",               # 10
        "019_pitcher_base",         # 11
        "021_bleach_cleanser",      # 12
        "024_bowl",                 # 13
        "025_mug",                  # 14
        "035_power_drill",          # 15
        "036_wood_block",           # 16
        "037_scissors",             # 17
        "040_large_marker",         # 18
        "051_large_clamp",          # 19
        "052_extra_large_clamp",    # 20
        "061_foam_brick"            # 21
    ]

    testlist = []
    # input_file = open('{0}/small_test_data_list.txt'.format(dataset_config_dir))
    input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        if input_line[-1:] == '\n':
            input_line = input_line[:-1]
        testlist.append(input_line)
    input_file.close()


elif opt.dataset == "lmo":
    cam_cx = 325.2611
    cam_cy = 242.04899
    cam_fx = 572.4114
    cam_fy = 573.57043
    cam_scale = 0.001
    cad_model_root_dir = libpath + '/models/lmo/'
    models = ['Ape', 'Can', 'Cat', 'Driller', 'Duck', 'Eggbox', 'Glue', 'Holepuncher']
    models_id = [1, 5, 6, 8, 9, 10, 11, 12]


num_visible_points = 2000
dataset_root_dir = opt.dataset_root_dir

K = [[cam_fx, 0, cam_cx],
     [0, cam_fy, cam_cy],
     [0, 0, 1]]
K = np.array(K)

point_clouds = {}
adi_results = {}
add_01d = {}
diameters = {}
corners = {}
sampled_clouds = {}

for model in models:
    print("*** " + model + " adding... ***")
    pcd_path = cad_model_root_dir + '{}/textured_simple.pcd'.format(model)
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd = np.asarray(pcd.points)
    point_clouds[model] = pcd

    dellist = [j for j in range(0, len(pcd))]
    dellist = random.sample(dellist, len(pcd) - num_visible_points)
    model_point = np.delete(pcd, dellist, axis=0)

    sampled_clouds[model] = model_point
    adi_results[model] = []
    add_01d[model] = []
    diameters[model] = calc_pts_diameter(pcd)

    x_max = np.max(pcd[:,0])
    x_min = np.min(pcd[:,0])
    y_max = np.max(pcd[:,1])
    y_min = np.min(pcd[:,1])
    z_max = np.max(pcd[:,2])
    z_min = np.min(pcd[:,2])

    corner = np.array([[x_min, y_min, z_min],
                        [x_max, y_min, z_min],
                        [x_max, y_max, z_min],
                        [x_min, y_max, z_min],

                        [x_min, y_min, z_max],
                        [x_max, y_min, z_max],
                        [x_max, y_max, z_max],
                        [x_min, y_max, z_max]])
    corners[model] = corner

if opt.dataset == "ycb":
    for now in tqdm(range(len(testlist))):
    # for now in tqdm(range(0, 1857)):
        if opt.visualization == True:
            draw_box = cv2.imread('{0}/{1}-color.png'.format(dataset_root_dir, testlist[now]))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(dataset_root_dir, testlist[now]))
        indices = meta['cls_indexes']
        for itemid in indices:
            idx = np.where(indices == itemid)
            model = models[itemid[0]-1]
            # print("*** " + model + " ***")

            target_r = meta['poses'][:, :, idx[0][0]][:, 0:3]
            target_t = np.array([meta['poses'][:, :, idx[0][0]][:, 3:4].flatten()])
            target_t = target_t.T

            try:
                pose = np.load(opt.save_path + '{0}_{1}.npy'.format(models[itemid[0]-1], now))

                R_est = quaternion_matrix(pose[3:])[:3,:3]
                T_est = pose[:3].reshape(3,1)

                model_adds, _ = adi(R_est, T_est, target_r, target_t, point_clouds[model], diameters[model] * 0.1)
            except:
                model_adds = np.inf
                # continue
            # if True in np.isnan(pose):
            #     continue
            adi_results[model].append(model_adds)

            if opt.visualization == True:
                draw_object(itemid[0], K, R_est, T_est.T, draw_box, sampled_clouds[model], corners[model])
        
        if opt.visualization == True:
            cv2.imshow("result", draw_box)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break

    mean_add_s = []
    for model in models:
        print("*** ", model)
        add_s = np.asarray(adi_results[model])
        if len(add_s) == 0:
            continue
        result = compute_pose_metrics(add_s)
        print(result)
        mean_add_s.extend(add_s)

    print()
    result = compute_pose_metrics(np.asarray(mean_add_s), max_pose_dist=0.02)
    print("Mean ADD-S (2cm < : ", result)
    result = compute_pose_metrics(np.asarray(mean_add_s), max_pose_dist=0.01)
    print("Mean ADD-S (1cm < : ", result)
    result = compute_pose_metrics(np.asarray(mean_add_s), max_pose_dist=0.005)
    print("Mean ADD-S (0.5cm < : ", result)

elif opt.dataset == "lmo":
    scene_gt = load_scene_gt('{0}/scene_gt.json'.format(dataset_root_dir))
    for now in tqdm(range(1214)):
        if opt.visualization == True:
            img = cv2.imread('{0}/rgb/{1}.png'.format(dataset_root_dir, '%06d' % now))
            draw_box = img.copy()
            draw_box_gt = img.copy()
        gt = scene_gt[now]
        for i, model in enumerate(models):
            id = models_id[i]
            for obj_gt in gt:
                if obj_gt['obj_id'] == id:
                    target_r = obj_gt['cam_R_m2c']
                    target_t = obj_gt['cam_t_m2c'] * cam_scale

            try:
                pose = np.load(opt.save_path + '{0}_{1}.npy'.format(model, now))
                
            except:
                continue

            R_est = quaternion_matrix(pose[3:])[:3,:3]
            T_est = pose[:3].reshape(3,1)

            if id == 11 or id == 10:
                adi_, correct = adi(R_est, T_est, target_r, target_t, point_clouds[model], diameters[model] * 0.1)
            else:
                adi_, correct = add(R_est, T_est, target_r, target_t, point_clouds[model], diameters[model] * 0.1)
            adi_results[model].append(adi_)
            add_01d[model].append(correct)

            if opt.visualization == True:
                draw_object(id, K, R_est, T_est.T, draw_box, sampled_clouds[model], corners[model])
                draw_object(id, K, target_r, target_t.T, draw_box_gt, sampled_clouds[model], corners[model])

        if opt.visualization == True:
            cv2.imshow("est", draw_box)
            cv2.imshow("draw_box_gt", draw_box_gt)
            key = cv2.waitKey(0)
            if key == ord('q'):
                    break

    avg = []
    for model in models:
            add_s = np.asarray(adi_results[model])
            print("*** ", model)
            try:
                result = compute_pose_metrics(add_s)
            except:
                continue
            print(result)
            print('ADD metric: {}'.format(np.mean(add_01d[model])))
            avg.append(np.mean(add_01d[model]))

    print("Mean ADD-0.1d : ", np.mean(np.asarray(avg)))

input_file = open(opt.save_path + "result.txt", mode='r')
lines = input_file.readlines()

time_sum = 0
for line in lines:
    t = line.split(': ')
    t = float(t[1][:-1])
    time_sum += t

print("average time : ", time_sum / len(lines))