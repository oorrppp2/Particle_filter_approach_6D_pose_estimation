import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import torch.nn.functional as F

class GeometricEmbeddingNetwork(nn.Module):
    def __init__(self, num_points):
        super(GeometricEmbeddingNetwork, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 512, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.ap2 = torch.nn.AvgPool1d(num_points)
        self.ap3 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points
    def forward(self, x):
        x = F.relu(self.conv1(x))   # 64
        ap_x1 = self.ap1(x)
        x = F.relu(self.conv2(x))   # 128
        ap_x2 = self.ap2(x)
        x = F.relu(self.conv3(x))   # 256
        x = F.relu(self.conv4(x))

        ap_x3 = self.ap3(x)

        return torch.cat([ap_x1, ap_x2, ap_x3], 1)

class CentroidPredictionNetwork(nn.Module):
    def __init__(self, num_points):
        super(CentroidPredictionNetwork, self).__init__()
        self.num_points = num_points
        self.feat_input = GeometricEmbeddingNetwork(num_points)
        self.feat_pcd = GeometricEmbeddingNetwork(num_points)
        
        self.conv1_t = torch.nn.Conv1d(704, 256, 1)
        self.conv2_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(128, 3, 1) #translation

        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(128)

    def forward(self, x, model):
        x = x.transpose(2, 1).contiguous()
        model = model.transpose(2, 1).contiguous()
        ap_point = self.feat_input(x)
        ap_model = self.feat_pcd(model)

        tx = ap_point + ap_model

        tx = F.relu(self.bn1(self.conv1_t(tx)))
        tx = F.relu(self.bn2(self.conv2_t(tx)))
        tx = self.conv3_t(tx)

        out_tx = tx.contiguous().transpose(2, 1).contiguous()

        return out_tx
 
