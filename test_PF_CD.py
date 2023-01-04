#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import numpy as np
import argparse
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import Code_PFNet.utils
from Code_PFNet.utils import PointLoss_test
from Code_PFNet.utils import distance_squre
import Code_PFNet.ModelNet40Loader
from Code_PFNet.shapenet_part_loader  import PartDataset
from Code_PFNet.model_PFNet import MouldnetD,MouldnetG
from arguments_test import  parse_args
from torch.autograd import Variable




#备注： 在我们的论文中显示倒角距离和两个指标。
#个人理解：  把这一类的class_choice给算出值是多大。
'''
--dataroot  ./data/shapenet_part_Airplane50/
--class_choice Airplane
--netG Pth_FPnet/point_netG48.pth
'''




def distance_squre1(p1,p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2

def OninitData(args):
    test_dset = PartDataset(root=args.dataroot, classification=True,
                                                 class_choice=args.class_choice, npoints=args.pnum, split='test')
    test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=args.batchSize,
                                                  shuffle=True, num_workers=int(args.workers))
    length = len(test_dataloader)
    return test_dset , test_dataloader,length

def Inintdevice():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def InitPFnet(args,device):
    point_netG = MouldnetG(args.num_scales, args.each_scales_size, args.point_scales_list, args.crop_point_num)
    point_netG = torch.nn.DataParallel(point_netG)
    point_netG.to(device)
    point_netG.load_state_dict(torch.load(args.netG, map_location=lambda storage, location: storage)['state_dict'])
    point_netG.eval()

    criterion_PointLoss = PointLoss_test().to(device)

    return point_netG,criterion_PointLoss

def RandomSample(input_part,partsumpoint):

    batchs,points,others= input_part.shape
    TempNumpy = np.zeros((batchs, partsumpoint, others), dtype=np.float32)

    for i in range(batchs):
        Temp_torch= torch.squeeze(input_part[i].cpu())
        Temp_torch = Temp_torch.detach().numpy()

        n = np.random.choice(len(Temp_torch), partsumpoint, replace=False)  # s随机采500个数据，这种随机方式也可以自己定义
        TempNumpy[i] = Temp_torch[n]

    return torch.as_tensor(TempNumpy)  # NUMPY   to  tensor



def GetInputlackList(args, input_cropped1, device):

    input_cropped1 = torch.squeeze(input_cropped1, 1)
    input_cropped2 =  RandomSample(input_cropped1, args.point_scales_list[1])
    input_cropped3 =  RandomSample(input_cropped1, args.point_scales_list[2])

    input_cropped2 = input_cropped2.to(device)
    input_cropped3 = input_cropped3.to(device)
    input_cropped = [input_cropped1, input_cropped2, input_cropped3]

    return input_cropped


def Getinput_cropped_partial(args,device,batch_size,real_point,distance_order):

    input_cropped_partial = torch.FloatTensor(batch_size, 1, args.pnum - args.crop_point_num, 3)

    crop_num_list = []
    for num_ in range(args.pnum - args.crop_point_num):
        crop_num_list.append(distance_order[num_ + args.crop_point_num][0])
    indices = torch.LongTensor(crop_num_list)
    input_cropped_partial[0, 0] = torch.index_select(real_point[0, 0], 0, indices)
    input_cropped_partial = torch.squeeze(input_cropped_partial, 1)
    input_cropped_partial = input_cropped_partial.to(device)



    return input_cropped_partial


def Getreal_center(args,IDX,batch_size,real_point):

    choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]), torch.Tensor([-1, 0, 0]),
              torch.Tensor([-1, 1, 0])]
    index = choice[IDX - 1]
    IDX = IDX + 1
    if IDX % 5 == 0:
        IDX = 0
    distance_list = []
    #    p_center  = real_point[0,0,index]
    p_center = index
    for num in range(args.pnum):
        distance_list.append(distance_squre(real_point[0, 0, num], p_center))
    distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])


    real_center = torch.FloatTensor(batch_size, 1, args.crop_point_num, 3)
    input_cropped1 = torch.FloatTensor(args.batchSize, 1, args.pnum, 3)
    input_cropped1.resize_(real_point.size()).copy_(real_point)

    for sp in range(args.crop_point_num):
        input_cropped1.data[0, 0, distance_order[sp][0]] = torch.FloatTensor([0, 0, 0])
        real_center.data[0, 0, sp] = real_point[0, 0, distance_order[sp][0]]

    real_center = torch.squeeze(real_center, 1)

    return IDX ,input_cropped1, real_center,distance_order  # input_crop  和 input_drop
################################
def GetGnetPoint(input_part_listA,point_netG):
    fake_center1, fake_center2, fake = point_netG(input_part_listA)
    return fake


# input_cropped_partial
def GetLossdist(length,device,criterion_PointLoss,input_cropped_partial,real_center, fake,real_point,CD,Gt_Pre,Pre_Gt):

    # fake_whole = torch.cat((input_cropped_partial, fake), 1)
    # fake_whole = fake_whole.to(device)
    # real_point = real_point.to(device)

    real_center = real_center.to(device)
    dist_all, dist1, dist2 = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(real_center,1))
               # +0.1*criterion_PointLoss(torch.squeeze(fake_part,1),torch.squeeze(real_center,1))

    dist_all = dist_all.cpu().detach().numpy()
    dist1 = dist1.cpu().detach().numpy()
    dist2 = dist2.cpu().detach().numpy()

    # CD = CD + dist_all / length
    # Gt_Pre = Gt_Pre + dist1 / length
    # Pre_Gt = Pre_Gt + dist2 / length
    # print(CD, Gt_Pre, Pre_Gt)

    return dist_all, dist1, dist2



if __name__ == '__main__':
    args = parse_args()
    test_dset,test_dataloader,length = OninitData(args)
    device = Inintdevice()
    point_netG,criterion_PointLoss = InitPFnet(args,device)

    ####################################

    errG_min = 100
    n = 0
    CD = 0
    Gt_Pre = 0
    Pre_Gt = 0
    IDX = 1
    Aloss = []
    Bloss = []
    Closs = []


    for i, data in enumerate(test_dataloader, 0):

        real_point, target = data
        real_point = torch.unsqueeze(real_point, 1)
        batch_size = real_point.size()[0]

        IDX ,input_cropped1, real_center,distance_order =  Getreal_center(args,IDX,batch_size,real_point)
        input_cropped_partial =  Getinput_cropped_partial(args,device,batch_size,real_point,distance_order)
        input_croppedlist = GetInputlackList(args, input_cropped1, device)

        fake = GetGnetPoint(input_croppedlist,point_netG)
        CD, Gt_Pre, Pre_Gt= GetLossdist(length,device,criterion_PointLoss,input_cropped_partial,real_center, fake,real_point,CD,Gt_Pre,Pre_Gt)

        Aloss.append(CD)
        Bloss.append(Gt_Pre)
        Closs.append(Pre_Gt)


    avgepoch_Aloss = sum(Aloss) / len(Aloss)  # 所有的损失 / 摊平到每一个上
    CD = float('%.5f' % avgepoch_Aloss)
    avgepoch_Aloss = sum(Bloss) / len(Bloss)  # 所有的损失 / 摊平到每一个上
    Gt_Pre = float('%.5f' % avgepoch_Aloss)
    avgepoch_Aloss = sum(Closs) / len(Closs)  # 所有的损失 / 摊平到每一个上
    Pre_Gt = float('%.5f' % avgepoch_Aloss)

    print(CD, Gt_Pre, Pre_Gt)
    print("CD:{} , Gt_Pre:{} , Pre_Gt:{}".format(float(CD), float(Gt_Pre), float(Pre_Gt)))


