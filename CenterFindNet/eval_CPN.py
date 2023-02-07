import os
import sys
import cv2
import time

libpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(libpath)
sys.path.append(libpath + '/../lib')

import torch
from torch.autograd import Variable
import numpy as np
import random
import cv2
from PIL import Image
import scipy.io as scio
import numpy.ma as ma
import open3d as o3d
import argparse
from utils import str2bool, calc_pts_diameter, load_scene_gt
from lib.centroid_prediction_network import CentroidPredictionNetwork
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='', help='dataset')
parser.add_argument('--dataset_root_dir', type=str, default='', help='dataset root dir')
parser.add_argument('--save_path', type=str, default='',  help='save results path')
parser.add_argument('--model_path', type=str, default='/trained_model/CPN_model_91_0.00023821471899932882.pth',  help='save results path')
parser.add_argument('--visualization', type=str2bool, default=False,  help='visualization')
opt = parser.parse_args()


""" Load Centroid Prediction Network """
model_path = libpath + opt.model_path
ycb_toolbox_dir = libpath + '/YCB_Video_toolbox'
dataset_config_dir = libpath + '/datasets/dataset_config'
cad_model_root_dir = libpath + '/../models/ycb/'

img_width = 480
img_length = 640
num_points = 1000

if opt.dataset == "ycb":
    testlist = []
    trainlist = []
    input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        if input_line[-1:] == '\n':
            input_line = input_line[:-1]
        testlist.append(input_line)
    input_file.close()
    
    cam_cx = 312.9869
    cam_cy = 241.3109
    cam_fx = 1066.778
    cam_fy = 1067.487
    cam_scale = 0.0001

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


elif opt.dataset == "lmo":
    scene_gt = load_scene_gt('{0}/scene_gt.json'.format(opt.dataset_root_dir))
    cam_cx = 325.2611
    cam_cy = 242.04899
    cam_fx = 572.4114
    cam_fy = 573.57043
    cam_scale = 0.001
    cad_model_root_dir = libpath + '/../models/lmo/'
    models = ['Ape', 'Can', 'Cat', 'Driller', 'Duck', 'Eggbox', 'Glue', 'Holepuncher']
    models_id = [1, 5, 6, 8, 9, 10, 11, 12]

K = [[cam_fx, 0, cam_cx],
     [0, cam_fy, cam_cy],
     [0, 0, 1]]
K = np.array(K)


estimator = CentroidPredictionNetwork(num_points = num_points)
estimator.cuda()
estimator.load_state_dict(torch.load(model_path))
estimator.eval()

xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])

cpn_error_sum = {}
num_of_obj = {}
surface_depth_error_sum = {}
mean_depth_error_sum = {}
model_points = {}
diameters = {}

cpn_error = []
surface_depth_error = []
mean_depth_error = []
# objs = 0

for model in models:
    print("*** " + model + " adding... ***")
    pcd_path = cad_model_root_dir + '{}/textured_simple.pcd'.format(model)
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd = np.asarray(pcd.points)

    if model == 'Eggbox':
        z = pcd[:,2].copy()
        pcd[:,2] = pcd[:,1]
        pcd[:,1] = z    


    dellist = [j for j in range(0, len(pcd))]
    dellist = random.sample(dellist, len(pcd) - num_points)
    model_point = np.delete(pcd, dellist, axis=0)
    model_point = torch.from_numpy(model_point.astype(np.float32))
    model_point = Variable(model_point).cuda()
    model_point = model_point.view(1, num_points, 3)
    model_points[model] = model_point
    diameters[model] = calc_pts_diameter(pcd)

    cpn_error_sum[model] = 0
    num_of_obj[model] = 0
    surface_depth_error_sum[model] = 0
    mean_depth_error_sum[model] = 0

def rmse(src, dst):
    return np.linalg.norm(src - dst) / np.sqrt(len(dst))

def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    c_b = cmax - cmin
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


def eval_CPN(model, img, depth_copy, mask_label, masked_depth, target_t, model_points, diameters, cv2_img):

    global cpn_error_sum
    global num_of_obj
    global surface_depth_error_sum
    global mean_depth_error_sum

    rmin, rmax, cmin, cmax = get_bbox(mask_label)
    # masked_depth_copy_2 = masked_depth.copy()
    masked_depth_copy = masked_depth.copy()
    masked_depth_copy = masked_depth_copy[masked_depth_copy > 0]

    # print(masked_depth_copy[:100])
    var = np.var(masked_depth_copy)
    mean = np.mean(masked_depth_copy)

    """ This line eliminates outlier """
    masked_depth[masked_depth -mean > var + diameters[model]] = 0

    choose = masked_depth[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
    if len(choose) > num_points:
        c_mask = np.zeros(len(choose), dtype=int)
        c_mask[:num_points] = 1
        np.random.shuffle(c_mask)
        choose = choose[c_mask.nonzero()]
    else:
        return cv2_img

    depth_masked = masked_depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    choose = np.array([choose])

    pt2 = depth_masked #* cam_scale
    pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
    cloud = np.concatenate((pt0, pt1, pt2), axis=1)

    cloud = torch.from_numpy(cloud.astype(np.float32))
    cloud = Variable(cloud).cuda()
    cloud = cloud.view(1, num_points, 3)

    center_x = (cmin + cmax) / 2.
    center_y = (rmin + rmax) / 2.

    center_of_roi_z = depth_copy[int(center_y), int(center_x)] * cam_scale

    min_depth = min(masked_depth[masked_depth > 0])
    if center_of_roi_z == 0:
        center_of_roi_z = min_depth

    center_of_roi_x = (center_x - cam_cx) * center_of_roi_z / cam_fx
    center_of_roi_y = (center_y - cam_cy) * center_of_roi_z / cam_fy
    surface_depth = np.array([center_of_roi_x, center_of_roi_y, center_of_roi_z])

    max_depth = max(masked_depth[masked_depth > 0])
    min_depth = min(masked_depth[masked_depth > 0])
    mean_depth = (max_depth + min_depth) / 2.

    mean_depth_x = (center_x - cam_cx) * mean_depth / cam_fx
    mean_depth_y = (center_y - cam_cy) * mean_depth / cam_fy
    mean_depth_xyz = np.array([mean_depth_x, mean_depth_y, mean_depth])
    
    centroid = estimator(cloud, model_points[model])
    centroid = centroid[0,0,:].cpu().data.numpy()

    distance_CPN = sum(pow(centroid - target_t, 2))
    distance_surface_depth = sum(pow(surface_depth - target_t, 2))
    distance_mean_depth = sum(pow(mean_depth_xyz - target_t, 2))

    mean_depth_xyz /= mean_depth_xyz[2]
    projected_mean_depth_x = int(cam_fx * mean_depth_xyz[0] + cam_cx)
    projected_mean_depth_y = int(cam_fy * mean_depth_xyz[1] + cam_cy)
    centroid /= centroid[2]
    projected_centroid_x = int(cam_fx * centroid[0] + cam_cx)
    projected_centroid_y = int(cam_fy * centroid[1] + cam_cy)
    target_t /= target_t[2]
    projected_target_x = int(cam_fx * target_t[0] + cam_cx)
    projected_target_y = int(cam_fy * target_t[1] + cam_cy)

    cv2.circle(cv2_img, (projected_mean_depth_x, projected_mean_depth_y), 4, (0,255,0), 4)
    cv2.circle(cv2_img, (projected_mean_depth_x, projected_mean_depth_y), 2, (0,255,0), 2)
    cv2.circle(cv2_img, (projected_centroid_x, projected_centroid_y), 4, (0,0,255), 4)
    cv2.circle(cv2_img, (projected_centroid_x, projected_centroid_y), 2, (0,0,255), 2)
    cv2.circle(cv2_img, (projected_target_x, projected_target_y), 4, (255,100,0), 4)
    cv2.circle(cv2_img, (projected_target_x, projected_target_y), 2, (255,100,0), 2)

    cpn_error_sum[model] += distance_CPN
    surface_depth_error_sum[model] += distance_surface_depth
    mean_depth_error_sum[model] += distance_mean_depth
    cpn_error.append(distance_CPN)
    surface_depth_error.append(distance_surface_depth)
    mean_depth_error.append(distance_mean_depth)
    num_of_obj[model] += 1

    return cv2_img

if __name__ == '__main__':
    
    if opt.dataset == "ycb":
        for now in tqdm(range(0, len(testlist))):
            img = cv2.imread('{0}/{1}-color.png'.format(opt.dataset_root_dir, testlist[now]))
            depth = np.array(Image.open('{0}/{1}-depth.png'.format(opt.dataset_root_dir, testlist[now])))

            label = np.array(Image.open('{0}/{1}-label.png'.format(opt.dataset_root_dir, testlist[now])))
            meta = scio.loadmat('{0}/{1}-meta.mat'.format(opt.dataset_root_dir, testlist[now]))

            obj = meta['cls_indexes'].flatten().astype(np.int32)
            cv2_img = img.copy()
            depth_copy = depth.copy()

            for idx in obj:
                itemid = idx
                model = models[itemid-1]
                indices = meta['cls_indexes']
                idx = np.where(indices == itemid)
                target_t = np.array(meta['poses'][:, :, idx[0][0]][:, 3:4].flatten())
                mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
                mask = mask_label * mask_depth
                masked_depth = mask * depth * cam_scale

                try:
                    cv2_img = eval_CPN(model, img, depth_copy, mask_label, masked_depth, target_t, model_points, diameters, cv2_img)
                except Exception as e:
                    print(e)
            
            if opt.visualization == True:
                try:
                    cv2.imshow("cv2_img", cv2_img)
                    key = cv2.waitKey(0)
                    if key == ord('q'):
                        break
                except:
                    pass

    
    if opt.dataset == "lmo":
        for now in tqdm(range(0, 1213, 1)):
            img = cv2.imread('{0}/rgb/{1}.png'.format(opt.dataset_root_dir, '%06d' % now))
            depth = np.array(Image.open('{0}/depth/{1}.png'.format(opt.dataset_root_dir, '%06d' % now)))
            gts = scene_gt[now]
            cv2_img = img.copy()
            depth_copy = depth.copy()

            for i, gt in enumerate(gts):
                label = np.array(Image.open(opt.dataset_root_dir + "/mask_visib/{0}_{1}.png".format('%06d' % now, '%06d' % i)))

                target_r = gt['cam_R_m2c']
                target_t = gt['cam_t_m2c'] * cam_scale
                model_id = gt['obj_id']
                target_t = np.asarray(target_t).squeeze()

                for j, obj_id in enumerate(models_id):
                    if obj_id == model_id:
                        model = models[j]
                        break

                mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                mask_label = ma.getmaskarray(ma.masked_equal(label, 255))
                mask = mask_label * mask_depth
                masked_depth = mask * depth * cam_scale

                try:
                    cv2_img = eval_CPN(model, img, depth_copy, mask_label, masked_depth, target_t, model_points, diameters, cv2_img)
                except Exception as e:
                    print(e)
            
            if opt.visualization == True:
                try:
                    cv2.imshow("cv2_img", cv2_img)
                    key = cv2.waitKey(0)
                    if key == ord('q'):
                        break
                except:
                    pass


    print("Surface depth error")
    for model in models:
        if surface_depth_error_sum[model] == 0:
            continue
        print(model + " : \t", np.sqrt(surface_depth_error_sum[model] / (num_of_obj[model])))

    print("Surface depth error : ", np.sqrt(sum(surface_depth_error) / (len(surface_depth_error))))
    print()
    print("="*40)
    print()

    print("Mean depth error")
    for model in models:
        if mean_depth_error_sum[model] == 0:
            continue
        print(model + " : \t", np.sqrt(mean_depth_error_sum[model] / (num_of_obj[model])))

    print("Mean depth error : ", np.sqrt(sum(mean_depth_error) / (len(mean_depth_error))))
    print()
    print("="*40)
    print()


    print("CenterFindNet")
    for model in models:
        if cpn_error_sum[model] == 0:
            continue
        print(model + " : \t", np.sqrt(cpn_error_sum[model] / (num_of_obj[model])))
    print("CPN error : ", np.sqrt(sum(cpn_error) / (len(cpn_error))))