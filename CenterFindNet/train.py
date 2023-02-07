import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.dataset import ycb_dataset
from lib.centroid_prediction_network import CentroidPredictionNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root_dir', type=str, default = '', help='dataset root dir YCB_Video_Dataset')
parser.add_argument('--batch_size', type=int, default = 4, help='batch size')
parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.1, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.001, help='margin to decay lr & w')
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume_CPN', type=str, default = '',  help='resume CPN model')
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
opt = parser.parse_args()

def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    opt.num_points = 1000 #number of points on the input pointcloud
    opt.outf = 'CenterFindNet/trained_model' #folder to save trained models
    opt.weight_decay = 0.01

    estimator = CentroidPredictionNetwork(num_points = opt.num_points)
    estimator.cuda()

    if opt.resume_CPN != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_CPN)))

    opt.decay_start = False
    optimizer = optim.Adam(estimator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    # optimizer = optim.SGD(estimator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    dataset = ycb_dataset('train', opt.num_points, True, opt.dataset_root_dir, opt.noise_trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    test_dataset = ycb_dataset('test', opt.num_points, False, opt.dataset_root_dir, 0.0)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    criterion = nn.MSELoss()
    best_test = np.Inf
    st_time = time.time()

    for epoch in range(opt.start_epoch, opt.nepoch):
        print('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis_avg = 0.0
        estimator.train()
        optimizer.zero_grad()

        for i, data in enumerate(dataloader, 0):
            points, model_points, gt_centroid = data
            points, model_points, gt_centroid = Variable(points).cuda(), Variable(model_points).cuda(), Variable(gt_centroid).cuda()
            centroid = estimator(points, model_points)

            loss = criterion(centroid, gt_centroid)
            loss.backward()

            train_dis_avg += loss.item()
            train_count += 1

            print('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, train_count, train_count * opt.batch_size, loss.item()))
            optimizer.step()
            optimizer.zero_grad()
            train_dis_avg = 0

            if train_count != 0 and train_count % 1000 == 0:
                torch.save(estimator.state_dict(), '{0}/CPN_model_current.pth'.format(opt.outf))

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))

        print('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_count = 0
        estimator.eval()

        for j, data in enumerate(testdataloader, 0):
            points, model_points, gt_centroid = data
            points, model_points, gt_centroid = Variable(points).cuda(), Variable(model_points).cuda(), Variable(gt_centroid).cuda()
            centroid = estimator(points, model_points)
            loss = criterion(centroid, gt_centroid)

            test_dis += loss.item()
            print('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, loss))

            test_count += 1

        test_dis = test_dis / test_count
        print('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis))
        if test_dis <= best_test:
            best_test = test_dis
            torch.save(estimator.state_dict(), '{0}/CPN_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        if best_test < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
            # optimizer = optim.SGD(estimator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

if __name__ == '__main__':
    main()
