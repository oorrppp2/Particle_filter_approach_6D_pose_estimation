import os
import numpy as np
import cv2
from PIL import Image
import scipy.io as scio
import numpy.ma as ma
import argparse
from lib.utils import str2bool
from tqdm import tqdm
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='', help='dataset')
parser.add_argument('--dataset_root_dir', type=str, default='', help='dataset root dir')
parser.add_argument('--save_path', type=str, default='',  help='save results path')
parser.add_argument('--visualization', type=str2bool, default=False,  help='visualization')
parser.add_argument('--gaussian_std', type=str, default=None,  help='gaussian_std')
parser.add_argument('--max_iteration', type=int, default=20,  help='max_iteration')
parser.add_argument('--taus', type=str, default=None,  help='taus')
parser.add_argument('--num_particles', type=int, default=100,  help='num_particles')
parser.add_argument('--w_o_CPN', type=str2bool, default=False,  help='with out CPN')
parser.add_argument('--w_o_Scene_occlusion', type=str2bool, default=False,  help='with out secne occlusion')
opt = parser.parse_args()

if opt.taus is not None:
    opt.taus = list(map(float, opt.taus.split(' ')))

if opt.gaussian_std is not None:
    opt.gaussian_std = list(map(float, opt.gaussian_std.split(' ')))

import lib.particle_filter as particle_filter

if __name__ == '__main__':
    pf = particle_filter.ParticleFilter(opt.dataset, opt.dataset_root_dir, visualization=opt.visualization,
    gaussian_std=opt.gaussian_std, max_iteration=opt.max_iteration, taus=opt.taus, num_particles=opt.num_particles,
    w_o_CPN=opt.w_o_CPN, w_o_Scene_occlusion=opt.w_o_Scene_occlusion)
    
    processing_time_record = ""
    processing_time_for_an_object_record = ""
    if opt.dataset == "ycb":
        for now in tqdm(range(0, len(pf.testlist))):
            processing_time = time.time()
            img = cv2.imread('{0}/{1}-color.png'.format(pf.dataset_root_dir, pf.testlist[now]))
            depth = np.array(Image.open('{0}/{1}-depth.png'.format(pf.dataset_root_dir, pf.testlist[now])))
            posecnn_meta = scio.loadmat('{0}/results_PoseCNN_RSS2018/{1}.mat'.format(pf.ycb_toolbox_dir, '%06d' % (now)))
            label = np.array(posecnn_meta['labels'])

            posecnn_rois = np.array(posecnn_meta['rois'])
            meta = scio.loadmat('{0}/{1}-meta.mat'.format(pf.dataset_root_dir, pf.testlist[now]))
            indices = meta['cls_indexes']

            labels = label[label > 0]
            labels = np.unique(labels)

            for itemid in labels:
                if itemid not in posecnn_meta['rois'][:,1]:
                    labels = np.delete(labels, np.where(labels == itemid))

            objects_region = np.zeros((480,640))
            for labels_ in labels:
                label_region = ma.getmaskarray(ma.masked_equal(label, labels_))
                objects_region[label_region] = 1
                
            for itemid in labels:
                if itemid not in posecnn_meta['rois'][:,1] or itemid not in indices:
                    continue
                processing_time_for_an_object_start = time.time()
                best_score, pose = pf.start(itemid, img, depth, label, objects_region, dataset=opt.dataset, posecnn_meta=posecnn_meta)
                if best_score != 0:
                    processing_time_for_an_object = time.time() - processing_time_for_an_object_start
                    processing_time_for_an_object_str = "{0} Image {1} {2}".format(now, pf.models[itemid-1], processing_time_for_an_object)
                    processing_time_for_an_object_record += processing_time_for_an_object_str + '\n'
                    np.save(opt.save_path+pf.models[itemid-1]+"_"+str(now), pose)

            processing_time_record_str = "Finish No.{0} image, Processing time : {1}".format(now, time.time() - processing_time)
            processing_time_record += processing_time_record_str + '\n'

    write_str = ""
    if os.path.isfile(opt.save_path + "result.txt"):
        rf = open(opt.save_path + "result.txt")
        lines = rf.readlines()
        for line in lines:
            write_str += line
        write_str += processing_time_record
        rf.close()
    else:
        write_str += processing_time_record
    f = open(opt.save_path + "result.txt", mode='wt')
    f.write(write_str)
    f.close()



    write_str = ""
    if os.path.isfile(opt.save_path + "result_per_object.txt"):
        rf = open(opt.save_path + "result_per_object.txt")
        lines = rf.readlines()
        for line in lines:
            write_str += line
        write_str += processing_time_for_an_object_record
        rf.close()
    else:
        write_str += processing_time_for_an_object_record
    f = open(opt.save_path + "result_per_object.txt", mode='wt')
    f.write(write_str)
    f.close()

    info = pf.particle_filter_info
    f = open(opt.save_path + "info.txt", mode='wt')
    f.write(info)
    f.close()
