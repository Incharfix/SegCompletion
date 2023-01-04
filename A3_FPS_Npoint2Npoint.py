import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np


#主题： 对已有点，指定下采样（就是 稀释点云的意思）


############ 公用的 东西  ########################
#################################################


# 点云归一化：       pc_normalize（）
# 最远点采样：        farthest_point_sample(xyz， Num)   return 那个点是中心点
# 抽离出512点tensor：index_points(points, idx):
# 512点-半径形成512个组 ：query_ball_point（）
#原点与采点的矩阵值：  square_distance（）      #一个距离矩阵：原始所有点到 每个下采样点的距离 [1024行  512列]

# sample_and_group_all

#####################################



def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # GPU

    # device = xyz.device
    # B, N, C = xyz.shape
    # centroids = torch.zeros(B, npoint, dtype=torch.long).to(device) #8*512 的tensor
    #
    # distance = torch.ones(B, N).to(device) * 1e10       #距离 8*1024 # 一个空壳
    # farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)#第一个点随机，后边的是依据第一个
    # batch_indices = torch.arange(B, dtype=torch.long).to(device)

    # cpu

    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device) #8*512 的tensor

    distance = torch.ones(B, N).to(device) * 1e10       #距离 8*1024 # 一个空壳
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)#第一个点随机，后边的是依据第一个
    batch_indices = torch.arange(B, dtype=torch.long).to(device)




    for i in range(npoint):        #第一个采样点选随机初始化的索引，后边需要for-512次
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)# 得到当前采样点的坐标 B*3
        dist = torch.sum((xyz - centroid) ** 2, -1)            # 计算当前采样点与其他点的距离

        mask = dist < distance                                 # 选择距离最近的来更新距离（更新维护这个表）
        distance[mask] = dist[mask]#
        farthest = torch.max(distance, -1)[1]#重新计算得到最远点索引（在更新的表中选择距离最大的那个点）
    return centroids

# 作用： 将 之前有索引的点---抽出来 --- 成一个完整的 tensor格式（1024点 抽出512点-缩影的意思）
#输入： 点，但是带索引。
#输出： 实际得到的点 tensor 格式
#举例： id为[5，566，100，2000]， 在2048个点中 将这四个点(序号的值)抽出来，组成一个新的tensor
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

    return torch.as_tensor(TempNumpy)  # NUMPY   to  tensor




# 这里 与原作者 相比， 做了改善
#  只取我自己的 下采样就行
# 参数： 输入所有点，需要几个点，  输出：得到下采样的点

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

    # idx = query_ball_point(radius, nsample, xyz, new_xyz)
    # torch.cuda.empty_cache()
    # grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    # torch.cuda.empty_cache()
    # grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    # torch.cuda.empty_cache()

    # if points is not None:
    #     grouped_points = index_points(points, idx)
    #     new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    # else:
    #     new_points = grouped_xyz_norm
    # if returnfps:
    #     return new_xyz, new_points, grouped_xyz, fps_idx
    # else:
    #     return new_xyz
    return new_xyz

if __name__ == '__main__':


    with open('./test_data/532.pts') as file_obj:
        contents = file_obj.readlines();

    print(type(contents))
    # print(contents)

    #######################################################
    i = 0
    landmarks = []
    for line in contents:
        TT = line.strip("\n")  # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        if i > 2 and i < 71:
            # print TT
            #TT_temp = TT.split(" ")
            # x = float(TT_temp[0])
            # y = float(TT_temp[1].strip("\r"))  # \r :回车
            landmarks.append(TT)
        i += 1
    print(landmarks)
    ################################################################


    path = "./FDSFSD.txt"
    data = np.loadtxt(path)
    sdfsd = torch.from_numpy(data).float()  #torch.tensor(data)  # torch.float64,即double类型。

    print(sdfsd.shape)
    tor_arr = torch.unsqueeze(sdfsd, dim=0)  # 指定位置增加维度
    print(tor_arr.shape)
    print(type(tor_arr), tor_arr.dtype, sep = ' , ')

    new_xyz = sample_and_group(tor_arr, 1024)  # 输入需要三个维度
    new_xyz = torch.squeeze(new_xyz, dim=0)  # 指定位置增加维度
    print(new_xyz.shape)
    np.savetxt('output/111.txt' , new_xyz,fmt='%1.5f')
