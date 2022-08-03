import argparse
import os
import sys
import torch
import BACore
import numpy as np

from BAProblem.rotation import AngleAxisRotatePoint
from BAProblem.loss import UnitSphereReprojectionError
from BAProblem.io import LoadBALFromFile
from TorchLM.solver import Solve

from time import time


def ProcessBALDataset(points, cameras, features, ptIdx, camIdx):
    cameras[:, :3] = -cameras[:, :3]
    cameras[:, 3:6] = AngleAxisRotatePoint(cameras[:, :3], cameras[:, 3:6])
    weights = cameras[:, 6:7][camIdx.long()]
    features /= -weights
    weights *= ((features**2).sum(dim=1, keepdim=True) + 1).sqrt()
    cameras[:, 8] = cameras[:, 8] - 2*cameras[:, 7]**2
    cameras[:, 7] *= cameras[:, 6]**2
    cameras[:, 8] *= cameras[:, 6]**4
    cameras[:, 6] = 1
    points = -points
    return points, cameras, features, weights, ptIdx, camIdx


parser = argparse.ArgumentParser(description='Bundle adjuster')
parser.add_argument('--balFile', default='data/problem-1723-156502-pre.txt')
parser.add_argument('--device', default='cuda')  # cpu/cuda
args = parser.parse_args()

filename = args.balFile
device = args.device

# Load BA data
points, cameras, features, ptIdx, camIdx = LoadBALFromFile(filename)
points, cameras, features, weights, ptIdx, camIdx = ProcessBALDataset(
    points, cameras, features, ptIdx, camIdx)

# Optionally use CUDA
points, cameras, features, weights, ptIdx, camIdx = points.to(device),\
    cameras.to(device), features.to(
        device), weights.to(device), ptIdx.to(device), camIdx.to(device)

if device == 'cuda':
    torch.cuda.synchronize()

t1 = time()
# optimize
Solve(variables=[points, cameras],
      constants=[weights, features],
      indices=[ptIdx, camIdx],
      fn=UnitSphereReprojectionError,
      numIterations=15,
      numSuccessIterations=15)
t2 = time()

print("Time used %f secs." % (t2 - t1))
