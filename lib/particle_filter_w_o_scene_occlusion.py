
import os
import sys
import cv2
import time

libpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(libpath)
sys.path.append(libpath + '/../CenterFindNet/lib')
import render
import objloader
# import ctypes
from PIL import Image
import numpy as np
import scipy.io as scio
import random
import time
from transforms3d.quaternions import axangle2quat, mat2quat, qmult, qinverse
from transforms3d.euler import quat2euler, mat2euler, euler2quat
import matplotlib.pyplot as plt
import numpy.ma as ma
import open3d as o3d

from filterpy.monte_carlo import systematic_resample
from filterpy.monte_carlo import stratified_resample

from sample import *
from utils import quaternion_matrix, calc_pts_diameter, draw_object
from scipy import spatial

import torch
from torch.autograd import Variable
from centroid_prediction_network import CentroidPredictionNetwork

np.random.seed(0)

class ParticleFilter():
    def __init__(self, dataset="ycb", dataset_root_dir="", visualization=False, gaussian_std=0.1, max_iteration=20, tau=0.1, num_particles=180):
        if dataset == "ycb":
            self.cam_cx = 312.9869
            self.cam_cy = 241.3109
            self.cam_fx = 1066.778
            self.cam_fy = 1067.487
            self.cam_scale = 0.0001
            self.dataset_root_dir = dataset_root_dir
            self.ycb_toolbox_dir = libpath + '/../CenterFindNet/YCB_Video_toolbox'
            self.dataset_config_dir = libpath + '/../CenterFindNet/datasets/dataset_config'
            self.cad_model_root_dir = libpath + '/../models/ycb/'

            self.models = [
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

            # Same number of particles (180) for all objects.
            self.num_particles = [num_particles * 2, num_particles * 2, num_particles, num_particles * 2, num_particles* 2, num_particles*2,
            num_particles, num_particles*2, num_particles*2, num_particles, num_particles*2, num_particles*2, num_particles,
            int(num_particles/2), int(num_particles/2), num_particles*2, int(num_particles/2), num_particles, int(num_particles/2), int(num_particles/2), num_particles*2]

            # self.taus = [tau, tau * 2, tau, tau, tau, tau, tau, tau, tau, tau,
            #             tau, tau, tau, tau, tau * 2, tau, tau * 2, tau, tau * 2, tau * 2, tau]
            self.taus = [tau, tau * 1.5, tau, tau, tau, tau, tau, tau, tau, tau,
                        tau, tau, tau * 0.5, tau, tau * 1.5, tau, tau * 1.5, tau, tau * 2.0, tau * 2.0, tau]

            self.testlist = []
            input_file = open('{0}/test_data_list.txt'.format(self.dataset_config_dir))
            # input_file = open('{0}/small_test_data_list.txt'.format(self.dataset_config_dir))
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                self.testlist.append(input_line)
            input_file.close()
        
        elif dataset == "lmo":
            self.cam_cx = 325.2611
            self.cam_cy = 242.04899
            self.cam_fx = 572.4114
            self.cam_fy = 573.57043
            self.cam_scale = 0.001
            self.dataset_root_dir = dataset_root_dir
            self.cad_model_root_dir = libpath + '/../models/lmo/'
            # self.models = {1:'Ape', 5:'Can', 6:'Cat', 8:'Driller', 9:'Duck', 10:'Eggbox', 11:'Glue', 12:'Holepuncher'}
            self.models = ['Ape', 'Can', 'Cat', 'Driller', 'Duck', 'Eggbox', 'Glue', 'Holepuncher']
            self.models_id = [1, 5, 6, 8, 9, 10, 11, 12]
            # Same number of particles (720) for all objects.
            self.num_particles = [num_particles*2, num_particles*2, num_particles*2, num_particles*2, num_particles*2, num_particles*2, num_particles*2, num_particles*2]
            self.taus = [tau * 2, tau * 2, tau * 2, tau * 2, tau * 2, tau * 2, tau * 2, tau * 2]
        else:
            print("Write your own dataset config here.")
            exit(0)

        self.info = {'Height':480, 'Width':640, 'fx':self.cam_fx, 'fy':self.cam_fy, 'cx':self.cam_cx, 'cy':self.cam_cy}
        render.setup(self.info)

        self.visualization = visualization
        self.num_points = 1000
        # self.gaussian_std = gaussian_std
        self.gaussian_std = [0.1, 0.1, 0.1, 0.3, 0.2, 0.2, 0.1, 0.3, 0.05, 0.1, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1]
        self.max_iteration = max_iteration

        self.min_particles = 256
        
        self.K = [[self.cam_fx, 0, self.cam_cx],
            [0, self.cam_fy, self.cam_cy],
            [0, 0, 1]]
        self.K = np.array(self.K)

              
        """ Load Centroid Prediction Network """
        model_path = libpath + '/../CenterFindNet/trained_model/CPN_model_91_0.00023821471899932882.pth'
        # model_path = os.path.dirname(os.path.abspath(__file__)) +
        self.estimator = CentroidPredictionNetwork(num_points = self.num_points)
        self.estimator.load_state_dict(torch.load(model_path))
        self.estimator.eval()

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        
        self.context = {}
        self.diameters = {}
        self.model_points = {}
        self.np_model_points = {}
        self.corners = {}
        for model in self.models:
            # if model != self.models[3]:
            #     continue
            print("*** " + model + " adding... ***")
            cad_model_path = self.cad_model_root_dir + '{}/textured_simple.obj'.format(model)
            pcd_path = self.cad_model_root_dir + '{}/textured_simple.pcd'.format(model)

            V, F = objloader.LoadTextureOBJ_VF_only(cad_model_path)
            self.context[model] = render.SetMesh(V, F)

            pcd = o3d.io.read_point_cloud(pcd_path)
            pcd = np.asarray(pcd.points)

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
            self.corners[model] = corner

            self.diameters[model] = calc_pts_diameter(pcd)

            dellist = [j for j in range(0, len(pcd))]
            dellist = random.sample(dellist, len(pcd) - self.num_points)

            model_point = np.delete(pcd, dellist, axis=0)
            np_model_point = model_point.copy()

            model_point = torch.from_numpy(model_point.astype(np.float32)).view(1, self.num_points, 3)
            self.model_points[model] = model_point
            self.np_model_points[model] = np_model_point

        self.rotation_samples = {}
        for i, model in enumerate(self.models):
            self.rotation_samples[model] = get_rotation_samples(model, num_samples=self.num_particles[i])


    def mat2pdf_np(self, distance_matrix, mean, std):
        coeff = 1/(np.sqrt(2*np.pi) * std)
        pdf = coeff * np.exp(- (distance_matrix - mean)**2 / (2 * std**2))
        return pdf

    def mat2pdf(self, distance_matrix, mean, std):
        coeff = torch.ones_like(distance_matrix) * (1/(np.sqrt(2*np.pi) * std))
        mean = torch.ones_like(distance_matrix) * mean
        std = torch.ones_like(distance_matrix) * std
        pdf = coeff * torch.exp(- (distance_matrix - mean)**2 / (2 * std**2))
        return pdf

    def estimate(self, pos, weights):
        """returns mean and variance of the weighted particles"""
        mean = np.average(pos, weights=weights, axis=0)
        var  = np.average((pos - mean)**2, weights=weights, axis=0)
        return mean, var

    def start(self, itemid, now, img, depth, label, objects_region, dataset="ycb", posecnn_meta="", rois=[], target_t=None, target_r=None):
        # cv2.imshow("depth_i", depth.astype(np.float))
        # cv2.waitKey(0)
        # cv2.imshow("depth_f", depth.astype(np.float64))
        # cv2.waitKey(0)
        # exit(0)
        model = self.models[itemid-1]
        context = self.context[model]
        diameter = self.diameters[model]
        model_point = self.model_points[model]
        np_model_point = self.np_model_points[model]
        corner = self.corners[model]
        num_particles = self.num_particles[itemid-1]

        # Using ROI provided by PoseCNN.
        if dataset == "ycb":
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
            mask = mask_label * mask_depth
            masked_depth = mask * depth * self.cam_scale
            for rois in posecnn_meta['rois']:
                if rois[1] == itemid:
                    rmin = int(rois[3]) + 1
                    rmax = int(rois[5]) - 1
                    cmin = int(rois[2]) + 1
                    cmax = int(rois[4]) - 1

        masked_depth_copy = masked_depth.copy()
        masked_depth_copy = masked_depth_copy[masked_depth_copy > 0]

        var = np.var(masked_depth_copy)
        mean = np.mean(masked_depth_copy)
        masked_depth[masked_depth -mean > var + diameter] = 0

        # w/o Scene Occlusion
        other_objects_region = np.zeros((480, 640))

        # w/ Scene Occlusion
        # other_objects_region = objects_region.copy()
        # other_objects_region[mask_label] = 0
        # depth_zero_in_mask = np.logical_and(mask_label, np.logical_not(depth))
        # other_objects_region[depth_zero_in_mask] = 1

        render.setSrcDepthImage(self.info, masked_depth.copy(), other_objects_region.copy())
        threshold = 0.02
        render.setNumOfParticles(1, int(threshold * 1000000))

        transform_matrix = np.identity(4).astype(np.float32)
        # target_t[0] -= 0.008
        transform_matrix[:3,:3] = target_r
        transform_matrix[:3,3] = target_t.T
        render.render(context, transform_matrix)
        render.calcMatchingScore(self.info)
        matching_score = render.getMatchingScores(1)

        print("matching_score: ", matching_score)

        v = render.getDepth(self.info)
        visib_mask = v.copy()
        visib_mask[visib_mask > 0] = 1
        
        cv2.imshow("v", v)
        # cv2.imwrite("/home/user/GraspingResearch/thesis/related_work/v_hat.png", v.astype(np.float))
        # union = np.logical_or(visib_mask, mask_label)
        # union_O = union.copy().astype(np.float)
        # cv2.imshow("union_O", union_O)
        # union_O[other_objects_region > 0] = 0
        # inters = np.logical_and(visib_mask, mask_label)
        # inters_O = inters.copy().astype(np.float)
        # inters_O[other_objects_region > 0] = 0
        # v[inters_O == 0] = 0
        # cv2.imshow("v_1", v)
        masked_depth_copy = masked_depth.copy()
        # masked_depth_copy[inters_O == 0] = 0
        # cv2.imshow("masked_depth_1", masked_depth_copy)
        diff = v - masked_depth_copy
        diff *= 50
        # print(diff[diff>0])
        cv2.imshow("diff", diff.astype(np.float))
        # cv2.imwrite("/home/user/GraspingResearch/thesis/related_work/diff.png", diff.astype(np.float))
        # # print(union_O[union_O > 0])
        # # cv2.imshow("union", union)
        # cv2.imshow("union_O_1", union_O)
        # cv2.imshow("inters_O", inters_O)
        draw_box = img.copy()
        draw_object(itemid, self.K, transform_matrix[:3,:3], transform_matrix[:3,3], draw_box, np_model_point, corner)
        cv2.imshow("draw_box", draw_box)
        cv2.imwrite("/home/user/GraspingResearch/thesis/related_work/ground_truth.png", draw_box)

        key = cv2.waitKey(0)
        if key == ord('q'):
            exit(0)

        return
