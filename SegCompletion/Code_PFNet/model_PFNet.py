#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np

# paper中 CMLP，对原样本的 2048 ，1024，512三个标本都会进来一次
# 对一个3维的点，进行卷积，池化，最后形成一个特征图

class Convlayer(nn.Module):
    def __init__(self,point_scales):
        super(Convlayer,self).__init__()
        self.point_scales = point_scales
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.conv5 = torch.nn.Conv2d(256, 512, 1)
        self.conv6 = torch.nn.Conv2d(512, 1024, 1)
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(1024)
    def forward(self,x):
        x = torch.unsqueeze(x,1)  # [8,1,2048,3] # 成为2维图像的卷积
        x = F.relu(self.bn1(self.conv1(x))) #输入channel-1， 卷积核 1*3 ，输出channel-64
                                                            #  # [8,64,2048,1] # 成为2维图像的卷积
        x = F.relu(self.bn2(self.conv2(x)))   #(64, 64, 1)
                                                             # [8,64,2048,1]，原样再卷一次

        x_128 = F.relu(self.bn3(self.conv3(x)))        # (64, 128, 1) # [8,128,2048,1]，channel 增加
        x_256 = F.relu(self.bn4(self.conv4(x_128)))      #(128, 256, 1) # [8,256,2048,1]，channel 增加
        x_512 = F.relu(self.bn5(self.conv5(x_256)))      # (256, 512, 1) # [8,512,2048,1]，channel 增加
        x_1024 = F.relu(self.bn6(self.conv6(x_512)))   # (512, 1024, 1)  # [8,1024,2048,1]，channel 增加

        x_128 = torch.squeeze(self.maxpool(x_128),2)  # [8,128,2048,1] -> [8,128,1,1]  池化核point_scales 2048*1
        x_256 = torch.squeeze(self.maxpool(x_256),2) # [8,256,2048,1] -> [8,256,1,1]  池化核point_scales 2048*1
        x_512 = torch.squeeze(self.maxpool(x_512),2)  # [8,512,2048,1] -> [8,512,1,1]  池化核point_scales 2048*1
        x_1024 = torch.squeeze(self.maxpool(x_1024),2) # [8,1024,2048,1] -> [8,1024,1,1]  池化核point_scales 2048*1

        L = [x_1024,x_512,x_256,x_128]
        x = torch.cat(L,1)

        return x #  [8,1920,1,1]


#  三个原样，特征提取后，最后拼接成一个，
class Latentfeature(nn.Module):
    def __init__(self,num_scales,each_scales_size,point_scales_list):
        super(Latentfeature,self).__init__()
        self.num_scales = num_scales
        self.each_scales_size = each_scales_size
        self.point_scales_list = point_scales_list
        self.Convlayers1 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[0]) for i in range(self.each_scales_size)])
        self.Convlayers2 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[1]) for i in range(self.each_scales_size)])
        self.Convlayers3 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[2]) for i in range(self.each_scales_size)])
        self.conv1 = torch.nn.Conv1d(3,1,1)       
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self,x):

        outs = []                # 准备一个list， 进行特征拼接到一起
        #,x[0].shape , x[1].shape , x[2].shape)  # #之前的降采样得到的三种尺度 2048，1024 ，512

        for i in range(self.each_scales_size): outs.append(self.Convlayers1[i](x[0]))
        for j in range(self.each_scales_size): outs.append(self.Convlayers2[j](x[1]))
        for k in range(self.each_scales_size): outs.append(self.Convlayers3[k](x[2]))


        latentfeature = torch.cat(outs,2)              # [8, 1920, 3]
        latentfeature = latentfeature.transpose(1,2)   # [8, 3, 1920]  # transpose 指定维度调换
        latentfeature = F.relu(self.bn1(self.conv1(latentfeature))) # 卷积核(3,1,1) 输入channel-3，输出-1 ，卷积核1*1
                                                                    #               [8, 3, 1920]-> [8, 1, 1920]
        latentfeature = torch.squeeze(latentfeature,1)              # 再压缩 [8, 1, 1920] -> [8, 1920]
        #('Latentfeature 结束。三种尺度特征图拼接，维度增加，卷积后，再压缩维度，。结果：',latentfeature.shape)

        return latentfeature


class PointcloudCls(nn.Module):
    def __init__(self,num_scales,each_scales_size,point_scales_list,k=40):
        super(PointcloudCls,self).__init__()
        self.latentfeature = Latentfeature(num_scales,each_scales_size,point_scales_list)
        self.fc1 = nn.Linear(1920, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.latentfeature(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))        
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

# _netG 判别器G： 生成器
# MouldnetG
class MouldnetG(nn.Module):
    def  __init__(self,num_scales,each_scales_size,point_scales_list,crop_point_num):
        super(MouldnetG,self).__init__()
        self.crop_point_num = crop_point_num
        self.latentfeature = Latentfeature(num_scales,each_scales_size,point_scales_list)
        self.fc1 = nn.Linear(1920,1024)  # 1920*1  拉成 1024*1
        self.fc2 = nn.Linear(1024,512)  # 1024*1  拉成 512*1
        self.fc3 = nn.Linear(512,256)     # 512*1  拉成 256*1
        
        self.fc1_1 = nn.Linear(1024,128*512)
        self.fc2_1 = nn.Linear(512,64*128)#nn.Linear(512,64*256) !
        self.fc3_1 = nn.Linear(256,64*3)   # 256*1  拉成 192*1 （64*3）#

#        
        self.conv1_1 = torch.nn.Conv1d(512,512,1)#torch.nn.Conv1d(256,256,1) !
        self.conv1_2 = torch.nn.Conv1d(512,256,1)
        self.conv1_3 = torch.nn.Conv1d(256,int((self.crop_point_num*3)/128),1)
        self.conv2_1 = torch.nn.Conv1d(128,6,1)#torch.nn.Conv1d(256,12,1) !
        
#        self.bn1_ = nn.BatchNorm1d(512)
#        self.bn2_ = nn.BatchNorm1d(256)
        
    def forward(self,x):

        x = self.latentfeature(x)  #  三个原样，特征提取后，最后拼接成一个，

        # 以上：三个原样，卷积后形成1个特征图
        # 以下： 从1个特征图中，上采样形成三层的特征图
        x_1 = F.relu(self.fc1(x))    #1024   #形成三种特征金字塔
        x_2 = F.relu(self.fc2(x_1))  #512
        x_3 = F.relu(self.fc3(x_2))  #256
        
        
        pc1_feat = self.fc3_1(x_3)    #Linear： 256*1 拉成 192*1 ，  192是 64个点的意思
        pc1_xyz = pc1_feat.reshape(-1,64,3) # [8,192] -> [8,64,3]   # -1 保留 # 特征向量还原 拉成立体
        
        pc2_feat = F.relu(self.fc2_1(x_2))      # #Linear：拉成..
        pc2_feat = pc2_feat.reshape(-1,128,64)  # [8,8192] -> [8,128,64]
        pc2_xyz =self.conv2_1(pc2_feat)        # [8,128,64] -> [8,6,64]   # nn.Conv1d(128,6,1)
        
        pc3_feat = F.relu(self.fc1_1(x_1))      #  # #Linear：拉成..
        pc3_feat = pc3_feat.reshape(-1,512,128)
        pc3_feat = F.relu(self.conv1_1(pc3_feat))
        pc3_feat = F.relu(self.conv1_2(pc3_feat))
        pc3_xyz = self.conv1_3(pc3_feat)


        # 以上： 从1个特征图中，上采样形成三层的特征图
        # 以下： 三个特征图再卷积，之后再拼接，  还原出立体形
        pc1_xyz_expand = torch.unsqueeze(pc1_xyz,2)

        pc2_xyz = pc2_xyz.transpose(1,2)  # [8,6,64]   [8,64，6]
        pc2_xyz = pc2_xyz.reshape(-1,64,2,3)     # [8,64，6]  [8,64，2,3]   # 为了做加法， 做维度操作
        pc2_xyz = pc1_xyz_expand+pc2_xyz          #  [8,64，2,3]  = [8,64，2,3] +  [8,64,1，3]
        pc2_xyz = pc2_xyz.reshape(-1,128,3)       #  [8,64，2,3] ->  #  [8,128，3]
        pc2_xyz_expand = torch.unsqueeze(pc2_xyz,2)  #  [8,128，3]-> [8,128，1,3]


        pc3_xyz = pc3_xyz.transpose(1,2)   #   -> [8,128,12]
        pc3_xyz = pc3_xyz.reshape(-1,128,int(self.crop_point_num/128),3)  # [8,128,12]  -> [8,128,4,3]

        pc3_xyz = pc2_xyz_expand+pc3_xyz                 #   [8,128,4,3]  =  [8,128,1,3]  +  [8,128,4,3]
        pc3_xyz = pc3_xyz.reshape(-1,self.crop_point_num,3) #  [8,128,4,3]-> [8,512,3]
        return pc1_xyz,pc2_xyz,pc3_xyz #center1 ,center2 ,fine

#  point_netD   判别器
class MouldnetD(nn.Module):
    def __init__(self,crop_point_num):
        super(MouldnetD,self).__init__()
        self.crop_point_num = crop_point_num
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.maxpool = torch.nn.MaxPool2d((self.crop_point_num, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(448,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,16)
        self.fc4 = nn.Linear(16,1)
        self.bn_1 = nn.BatchNorm1d(256)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x_64 = F.relu(self.bn2(self.conv2(x)))        # 卷积
        x_128 = F.relu(self.bn3(self.conv3(x_64)))
        x_256 = F.relu(self.bn4(self.conv4(x_128)))


        x_64 = torch.squeeze(self.maxpool(x_64))   # 池化
        x_128 = torch.squeeze(self.maxpool(x_128))
        x_256 = torch.squeeze(self.maxpool(x_256))

        Layers = [x_256,x_128,x_64]

        x = torch.cat(Layers,1)           # 拼接在一起


        x = F.relu(self.bn_1(self.fc1(x)))           # 全连接 ， 由于 128-> 64 -> 1 这样最后得到1个值
        x = F.relu(self.bn_2(self.fc2(x)))
        x = F.relu(self.bn_3(self.fc3(x)))
        x = self.fc4(x)                       # print(x.shape)   torch.Size([2, 1])

        return x   # 做二分类任务 print(x) # tensor([[ 0.0561], [-0.1079]],   tensor([[-0.0046], [-0.0467]],

if __name__=='__main__':
    input1 = torch.randn(64,2048,3)
    input2 = torch.randn(64,512,3)
    input3 = torch.randn(64,256,3)
    input_ = [input1,input2,input3]
    netG=MouldnetG(3,1,[2048,512,256],1024)
    output = netG(input_)
    print(output)
