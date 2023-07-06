

import torch
import os
import numpy as np
import random
from torch.autograd import Variable
from Code_PFNet.utils import distance_squre
import Code_PFNet.utils as utils_PF
from Code_PFNet.utils import farthest_point_sample,index_points

def TestAndOpenRPM(i,epoch, _args, train_len,val_loader,datakind, errD, errG_D, errG_l2, errG, CD_LOSS,PointLoss,device,PFnetG):
    # print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f / %.4f/ %.4f'
    #       % (epoch, _args.epochs, i, train_len, errD.data, errG_D.data, errG_l2, errG, CD_LOSS))

    f = open('loss_PFNet.txt', 'a')
    f.write('\n' + '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f / %.4f /%.4f'
            % (epoch, _args.epochs, i, train_len, errD.data, errG_D.data, errG_l2, errG, CD_LOSS))

    if i % 50 == 0:
        #print('After, ', i, '-th batch')
        f.write('\n' + 'After, ' + str(i) + '-th batch')

        #########################
        # 经典的半路验证
        #   ：将数据传入到  生成网络中，计算CD损失
        ###########################

        # for i, data in enumerate(val_loader, 0):
        for data in val_loader:

            point_gt, input_part1, point_drop, = GetRPMSplitPoint(_args, data,datakind, device)  # 分离数据
            input_lack_list = GetInputlackList(_args, input_part1, device, False)        # #三个层次缺失点云

            ###################
            # 以上是将原材料得到 三种不同的尺度
            # 以下是进入到 G 网络中 生成器 生成点云，计算CD 损失
            ###################
            PFnetG.eval()
            fake_center1, fake_center2, fake = PFnetG(input_lack_list)  # 跳入该函数中  MouldnetG
            CD_loss = PointLoss(torch.squeeze(fake, 1), torch.squeeze(point_drop, 1))
            #print('test result:', CD_loss)
            f.write('\n' + 'test result:  %.4f' % (CD_loss))
            break
    f.close()

    return PFnetG

#  @ inchar
# 这里是从PFNet 中搬过来的一些函数

def TestAndOpen(i,epoch, _args, dataloader,val_loader, errD, errG_D, errG_l2, errG, CD_LOSS,PointLoss,device,PFnetG):
    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f / %.4f/ %.4f'
          % (epoch, _args.epochs, i, len(dataloader), errD.data, errG_D.data, errG_l2, errG, CD_LOSS))

    f = open('loss_PFNet.txt', 'a')
    f.write('\n' + '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f / %.4f /%.4f'
            % (epoch, _args.epochs, i, len(dataloader), errD.data, errG_D.data, errG_l2, errG, CD_LOSS))

    if i % 10 == 0:
        print('After, ', i, '-th batch')
        f.write('\n' + 'After, ' + str(i) + '-th batch')

        #########################
        # 经典的半路验证
        #   ：将数据传入到  生成网络中，计算CD损失
        ###########################

        for i, data in enumerate(val_loader, 0):
            point_gt, target = data

            point_gt, input_part1, point_drop, = GetSplitPoint(_args, point_gt, device)  # 分离数据
            input_lack_list = GetInputlackList(_args, input_part1, device, False)        # #三个层次缺失点云

            ###################
            # 以上是将原材料得到 三种不同的尺度
            # 以下是进入到 G 网络中 生成器 生成点云，计算CD 损失
            ###################
            PFnetG.eval()
            fake_center1, fake_center2, fake = PFnetG(input_lack_list)  # 跳入该函数中  MouldnetG
            CD_loss = PointLoss(torch.squeeze(fake, 1), torch.squeeze(point_drop, 1))
            print('test result:', CD_loss)
            f.write('\n' + 'test result:  %.4f' % (CD_loss))
            break
    f.close()

    return PFnetG


def  savetestpoint(savename, tempdata):
    savepath = './output_testpoint/'
    if(os.path.exists(savepath) == False):
        os.makedirs(savepath)
    tempdata = torch.squeeze(tempdata.cpu()) # torch.squeeze(gt.cpu())
    np.savetxt(savepath + savename +'.txt', tempdata.detach().numpy())

def  printdictKey(dictval):
    for key, value in dictval.items():
        # print('{key}:{value}'.format(key=key, value=value))
        print('key:','{key}:'.format(key=key),type(value),value.shape)

def  GetRPMSplitPoint (_args,data,datakind,device):

    # points_src  points_ref
    point_gt = data['points_srcFPgt'][:,:,0:3]
    input_part1 = data['points_srcFPpart'][:,:,0:3]
    point_drop =  data['points_srcFPdrop'][:,:,0:3]

    # 这样写，防止变量未定义
    if datakind == "points_ref":
        point_gt = data['points_refFPgt'][:,:,0:3]
        input_part1 = data['points_refFPpart'][:,:,0:3]
        point_drop =  data['points_refFPdrop'][:,:,0:3]


    # input_part1  # 补0
    channel1, temp_dropnum, channel3 = point_drop.shape
    dropTensor = torch.zeros([channel1, temp_dropnum, channel3])
    input_part1 = torch.cat((input_part1, dropTensor), axis=1)  # 补0



    point_gt = torch.unsqueeze(point_gt, 1)  # torch.Size([2, 1, 2048, 3]) #延展
    input_part1 = torch.unsqueeze(input_part1, 1)  # torch.Size([2, 1, 2048, 3]) #延展
    point_drop = torch.unsqueeze(point_drop, 1)  # torch.Size([2, 1, 2048, 3]) #延展

    point_gt = point_gt.to(device)  # 源文件值2048个  torch.Size([2, 1, 2048, 3])
    input_part1 = input_part1.to(device)  # 经过造假后的点 torch.Size([2, 1, 2048, 3])  #距离近的index，赋值为0,0,0，默认是丢失的。
    point_drop = point_drop.to(device)  # 源文件值512个  torch.Size([2, 1, 512, 3])

    return point_gt, input_part1, point_drop



def  GetSplitPoint (_args,point_gt,device):

    batch_size,_,_ = point_gt.shape

    input_part1 = torch.FloatTensor(batch_size, _args.num_points, 3)  # 新建 torch.Size([2, 2048, 3])，一个部分真实的点云
    input_part1 = input_part1.data.copy_(point_gt)  # torch.Size([2, 2048, 3])   # copy point_gt
    point_drop = torch.FloatTensor(batch_size, 1, _args.point_drop_num, 3)  # 新建 torch.Size([2, 2048, 3])


    point_gt = torch.unsqueeze(point_gt, 1)  # torch.Size([2, 1, 2048, 3]) #延展
    input_part1 = torch.unsqueeze(input_part1, 1)  # torch.Size([2, 1, 2048, 3]) #延展

     ######### 以上 初步得到 [point_gt 的延展]  [input_part1 复制]    [point_drop 的空壳]  [label 真假值]
     #

    if _args.cropmethod == 'random_center':
        # Set viewpoints
        choice = [torch.Tensor([1, -1, 0]),torch.Tensor([1, 0, 1]),torch.Tensor([1, 1, -1]),

                torch.Tensor([0, 0, -1]),torch.Tensor([0, -1, 1]),torch.Tensor([0, 0, 1]),

                torch.Tensor([-1, 1, -1]),torch.Tensor([-1, 0, 0]),torch.Tensor([-1, 1, 0])]  # 可以理解为， 字符串数组， 日后随机取出其中一个 “字符串”
            # 作用： 丢失的点，在那个区域丢失

        for m in range(batch_size):
            index = random.sample(choice, 1)  # Random choose one of the element
            distance_list = []
            p_center = index[0]  # 随机取出 choice 中的一个元素

            for n in range(_args.num_points):  # 点云的总点数， 公认1024个
                distance_list.append(distance_squre(point_gt[m, 0, n], p_center))  # batch的第0个， 共2048点
            distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])
            # print(distance_order)     # index 和 距离值
            # 以上的作用： 以某个点为中心，和2048个点进行比较，，遍历哪个点是最近的， 就把那个512个点缺失掉。

            for sp in range(_args.point_drop_num):  # 把一部分点去掉，并且保存其标签
                input_part1.data[m, 0, distance_order[sp][0]] = torch.FloatTensor([0, 0, 0])  # 距离近的index，赋值为0,0,0
                point_drop.data[m, 0, sp] = point_gt[m, 0, distance_order[sp][0]]             # 保存index下的实际距离

    point_gt = point_gt.to(device)  # 源文件值2048个  torch.Size([2, 1, 2048, 3])
    input_part1 = input_part1.to(device)  # 经过造假后的点 torch.Size([2, 1, 2048, 3])  #距离近的index，赋值为0,0,0，默认是丢失的。
    point_drop = point_drop.to(device)  # 源文件值512个  torch.Size([2, 1, 512, 3])



    return point_gt,input_part1,point_drop

# 在训练的时候需要， 在测试的时候不需要，所以单独拿出写
def SaveGanLabel(batch_size,device):

    label = torch.FloatTensor(batch_size)
    label.resize_([batch_size, 1]).fill_(1)  # 默认填充 true
    label = label.to(device)  # torch.Size([2, 1])
    return  label



def SaveFPNetEpoch(epoch,PFnetG,PFnetD):
    savedir = './Code_PFNet/Checkpoint'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    if epoch % 50 == 0:  # 原先10
        torch.save({'epoch': epoch + 1, 'state_dict': PFnetG.state_dict()}, savedir +'/PFnetG' + str(epoch) + '.pth')
        torch.save({'epoch': epoch + 1, 'state_dict': PFnetD.state_dict()}, savedir +'/PFnetD' + str(epoch) + '.pth')

def  PF_epoch_alpha(epoch):
    if epoch < 30:
        alpha1 = 0.01
        alpha2 = 0.02
    elif epoch < 80:
        alpha1 = 0.05
        alpha2 = 0.1
    else:
        alpha1 = 0.1
        alpha2 = 0.2
    return alpha1,alpha2

def RandomSample(input_part,partsumpoint):

    batchs,points,others= input_part.shape
    TempNumpy = np.zeros((batchs, partsumpoint, others), dtype=np.float32)

    for i in range(batchs):
        Temp_torch= torch.squeeze(input_part[i].cpu())
        Temp_torch = Temp_torch.detach().numpy()

        n = np.random.choice(len(Temp_torch), partsumpoint, replace=False)  # s随机采500个数据，这种随机方式也可以自己定义
        TempNumpy[i] = Temp_torch[n]

    return torch.as_tensor(TempNumpy)  # NUMPY   to  tensor


def  Get_PointDrop_Level(point_drop,device):
    # point_drop 分支
    point_drop = Variable(point_drop, requires_grad=True)  # variable可以反向传播
    point_drop = torch.squeeze(point_drop, 1)

    point_drop1 =  RandomSample(point_drop, 64)
    point_drop2 =  RandomSample(point_drop, 128)

    # point_drop1 = utils_PF.get_farthest_point(point_drop, 64, RAN=False)
    #point_drop2 = utils_PF.get_farthest_point(point_drop, 128, RAN=True)



    point_drop1 = Variable(point_drop1, requires_grad=True)
    point_drop2 = Variable(point_drop2, requires_grad=True)

    point_drop1 = point_drop1.to(device)
    point_drop2 = point_drop2.to(device)

    return point_drop,point_drop1,point_drop2


def  GetInputlackList(_args,input_part1,device,requires_grad_01):
    # input_part1  cropped2  cropped3 分支
    input_part1 = torch.squeeze(input_part1, 1)
    # input_part2 = utils_PF.get_farthest_point(input_part1, _args.point_scales_list[1], RAN=True)
    # input_part3 = utils_PF.get_farthest_point(input_part1, _args.point_scales_list[2], RAN=False)

    input_part2 =  RandomSample(input_part1, _args.point_scales_list[1])
    input_part3 =  RandomSample(input_part1, _args.point_scales_list[2])

    input_part1 = Variable(input_part1, requires_grad_01)  # torch.Size([2, 2048, 3])
    input_part2 = Variable(input_part2, requires_grad_01)  # torch.Size([2, 1024, 3])
    input_part3 = Variable(input_part3, requires_grad_01 )  # torch.Size([2, 512, 3])

    input_part2 = input_part2.to(device)
    input_part3 = input_part3.to(device)

    input_lack_list = [input_part1, input_part2, input_part3]

    return input_lack_list



# def  preparePFNet(train_data,_device):
#     print(_args.num_points)
#     print(_args.partial)
#
#     # points_src 上升到1024
#     temp_points_src = train_data['points_src']
#     temp_dropnum =  int((1 - _args.partial[0])*(_args.num_points))
#
#     channel1,channel2,channel3= temp_points_src.shape
#     dropTensor= torch.zeros([channel1,temp_dropnum, channel3])
#     train_data['points_src'] = torch.cat((temp_points_src, dropTensor), axis=1)
#
#     #缺失点云(1024) 进行下采样三种尺度
#
#
#     # input_part1  cropped2  cropped3 分支
#     points_src1 = train_data['points_src'][:,:,:3]
#     points_src2_idx = farthest_point_sample(points_src1, _args.point_scales_list[1], RAN=True)
#     points_src3_idx = farthest_point_sample(points_src1, _args.point_scales_list[2], RAN=False)
#     points_src2 = index_points(points_src1, points_src2_idx)
#     points_src3 = index_points(points_src1, points_src3_idx)
#
#     # points_src1 = Variable(points_src1, requires_grad=True)  # torch.Size([2, 2048, 3])
#     # points_src2 = Variable(points_src2, requires_grad=True)  # torch.Size([2, 1024, 3])
#     # points_src3 = Variable(points_src3, requires_grad=True)  # torch.Size([2, 512, 3])
#     points_src1 = points_src1.to(_device)
#     points_src2 = points_src2.to(_device)
#     points_src3 = points_src3.to(_device)
#     points_src_PFnetList = [points_src1, points_src2, points_src3]
#     print(points_src1.shape,points_src2.shape,points_src3.shape)
#
#
#
#    #points_ref
#     temp_points_ref = train_data['points_ref']
#     temp_dropnum =  int((1 - _args.partial[1])*(_args.num_points))
#
#     channel1,channel2,channel3= temp_points_ref.shape
#     dropTensor= torch.zeros([channel1,temp_dropnum, channel3])
#     train_data['points_ref'] = torch.cat((temp_points_ref, dropTensor), axis=1)
#
#     #缺失点云(1024) 进行下采样三种尺度
#     # input_part1  cropped2  cropped3 分支
#     points_ref1 = train_data['points_ref'][:,:,:3]
#     points_ref2_idx = farthest_point_sample(points_ref1, _args.point_scales_list[1], RAN=True)
#     points_ref3_idx = farthest_point_sample(points_ref1, _args.point_scales_list[2], RAN=False)
#     points_ref2 = index_points(points_ref1, points_ref2_idx)
#     points_ref3 = index_points(points_ref1, points_ref3_idx)
#
#     # points_ref1 = Variable(points_src1, requires_grad=True)  # torch.Size([2, 2048, 3])
#     # points_ref2 = Variable(points_ref2, requires_grad=True)  # torch.Size([2, 1024, 3])
#     # points_ref3 = Variable(points_ref3, requires_grad=True)  # torch.Size([2, 512, 3])
#     points_ref1 = points_ref1.to(_device)
#     points_ref2 = points_ref2.to(_device)
#     points_ref3 = points_ref3.to(_device)
#
#     points_ref_PFnetList = [points_ref1, points_ref2, points_ref3]
#     print(points_ref1.shape,points_ref2.shape,points_ref3.shape)
#
#
#     # 其中一个打印出来查看
#     # print(temp_points_src.shape)
#     # print(dropTensor.shape)
#     # print(train_data['points_src'].shape)
#     # savetestpoint('points_src',train_data['points_src'])
#     # savetestpoint('temp_points_src',temp_points_src)
#     # print(111111111111)
#     return train_data,points_src_PFnetList,points_ref_PFnetList


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Conv1d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)