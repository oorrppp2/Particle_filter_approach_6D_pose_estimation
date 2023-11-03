import os
import sys
import numpy as np
import open3d as o3d
import math

import torch
from graspnetAPI import GraspGroup, Grasp

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

# from graspnet import GraspNet, pred_decode
from transform import euler_matrix


num_point = 20000



def get_and_process_data(pcd_path):

    # pcd_path = '/home/user/python_projects/ycb_models/trans_tumbler/textured_simple.pcd'
    cloud = o3d.io.read_point_cloud(pcd_path)
    color = np.asarray([0.8, 0.8, 0.8])

    # cloud = o3d.geometry.PointCloud()
    cloud_masked = np.asarray(cloud.points)
    color_masked = color[None, :] * np.ones((cloud_masked.shape[0],))[:, None]

    
    # sample points
    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def vis_predefined_grasps(cloud, pcd_path):
    # score, width, height, depth, rotation_matrix, translation, object_id
    gg = GraspGroup()
    additional_g = []

    initial_pose_matrices = np.zeros((72,4,4))

# #""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # for i in range(72):
    for i in range(72):
        trans = np.asarray([-0.0, 0.1, 0.0])
        roll, pitch, yaw = [np.pi/2.0, (i*5)*math.pi/180.0, 0]
        rot = euler_matrix(roll, pitch, yaw)[:3,:3]
        additional_g.append(Grasp(1, 0.08, 0.0, 0.0, rot, trans, -1))
        initial_pose_matrices[i] = euler_matrix(roll, pitch, yaw)
        initial_pose_matrices[i,:3,3] = trans

    # trans = np.asarray([-0.0, -0.01, 0.02])
    # roll, pitch, yaw = [0, 0, np.pi/2.0]
    # rot = euler_matrix(roll, pitch, yaw)[:3,:3]
    # additional_g.append(Grasp(1, 0.08, 0.0, 0.0, rot, trans, -1))
    # initial_pose_matrices[1] = euler_matrix(roll, pitch, yaw)
    # initial_pose_matrices[1,:3,3] = trans
    
    # trans = np.asarray([0.01, 0.0, 0.02])
    # roll, pitch, yaw = [0, 0, np.pi]
    # rot = euler_matrix(roll, pitch, yaw)[:3,:3]
    # additional_g.append(Grasp(1, 0.08, 0.0, 0.0, rot, trans, -1))
    # initial_pose_matrices[2] = euler_matrix(roll, pitch, yaw)
    # initial_pose_matrices[2,:3,3] = trans
    
    # trans = np.asarray([-0.01, 0.0, 0.02])
    # roll, pitch, yaw = [0, 0, 0]
    # rot = euler_matrix(roll, pitch, yaw)[:3,:3]
    # additional_g.append(Grasp(1, 0.08, 0.0, 0.0, rot, trans, -1))
    # initial_pose_matrices[3] = euler_matrix(roll, pitch, yaw)
    # initial_pose_matrices[3,:3,3] = trans
    
    # trans = np.asarray([-0.045, 0.105, 0.0])
    # roll, pitch, yaw = [0, 0, -np.pi/2.0]

    # for i in range(1,4):
    #     trans[0] = -0.02 * np.sin(2 * np.pi / 4 * i)
    #     trans[1] = -0.02 * np.cos(2 * np.pi / 4 * i)
    #     yaw -= np.pi / 2

    #     rot = euler_matrix(roll, pitch, yaw)[:3,:3]
    #     additional_g.append(Grasp(1, 0.105, 0.0, 0.0, rot, trans, -1))
    #     initial_pose_matrices[i*2] = euler_matrix(roll, pitch, yaw)
    #     initial_pose_matrices[i*2,:3,3] = trans
        
    #     # pitch = np.pi / 4
    #     rot = euler_matrix(roll, np.pi / 4, yaw)[:3,:3]
    #     additional_g.append(Grasp(1, 0.105, 0.0, 0.0, rot, trans, -1))
    #     initial_pose_matrices[i*2+1] = euler_matrix(roll, pitch, yaw)
    #     initial_pose_matrices[i*2+1,:3,3] = trans

#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # np.save("/home/user/trans_tumbler.npy", initial_pose_matrices)
    # np.save(pcd_path.split('.')[0]+'npy', initial_pose_matrices)
    for g in additional_g:
        gg.add(g)
    # gg.add(additional_g)
    # print(gg)
    print("Number of grippers : ", len(gg))
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])


def demo(pcd_file_path):
    
    end_points, cloud = get_and_process_data(pcd_file_path)

    vis_predefined_grasps(cloud, pcd_file_path)

if __name__=='__main__':
    args = sys.argv[1:]
    demo(args[0])
