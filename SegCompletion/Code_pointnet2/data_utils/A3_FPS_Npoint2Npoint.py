import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # cpu

    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)

    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)


    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)

        mask = dist < distance
        distance[mask] = dist[mask]#
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def RandomSample(input_part,partsumpoint):

    batchs,points,others= input_part.shape
    TempNumpy = np.zeros((batchs, partsumpoint, others), dtype=np.float32)

    for i in range(batchs):
        Temp_torch= torch.squeeze(input_part[i].cpu())
        Temp_torch = Temp_torch.detach().numpy()

        n = np.random.choice(len(Temp_torch), partsumpoint, replace=False)  # s随机采500个数据，这种随机方式也可以自己定义
        TempNumpy[i] = Temp_torch[n]

    return torch.as_tensor(TempNumpy)  # numpy   to  tensor


def sample_and_group(xyz ,npoint ):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]

    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()


    return new_xyz

if __name__ == '__main__':


    with open('./test_data/532.pts') as file_obj:
        contents = file_obj.readlines();

    print(type(contents))

    #######################################################
    i = 0
    landmarks = []
    for line in contents:
        TT = line.strip("\n")
        if i > 2 and i < 71:
            # print TT
            #TT_temp = TT.split(" ")
            # x = float(TT_temp[0])
            # y = float(TT_temp[1].strip("\r"))  # \r :
            landmarks.append(TT)
        i += 1
    print(landmarks)
    ################################################################


    path = "./FDSFSD.txt"
    data = np.loadtxt(path)
    sdfsd = torch.from_numpy(data).float()

    print(sdfsd.shape)
    tor_arr = torch.unsqueeze(sdfsd, dim=0)
    print(tor_arr.shape)
    print(type(tor_arr), tor_arr.dtype, sep = ' , ')

    new_xyz = sample_and_group(tor_arr, 1024)
    new_xyz = torch.squeeze(new_xyz, dim=0)
    print(new_xyz.shape)
    np.savetxt('output/111.txt' , new_xyz,fmt='%1.5f')
