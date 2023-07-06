import os
import sys
import argparse
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import utils
from utils import PointLoss
from utils import distance_squre
import data_utils as d_utils
import ModelNet40Loader
import shapenet_part_loader
from model_PFNet import _netlocalD,_netG  # _netG 生成G： generator     _netlocalD 判别器D： discriminator
import numpy as np

dset_root =      './dataset/shapenet_part_Airplane50/'      #split='train'
test_dset_root = './dataset/shapenet_part_Airplane50/' # split='test'



"""
问题： Dimension out of range (expected to be in range of [-1, 0], but got 1)

解决1： 剪裁数据集 让它的大小适合批处理大小 eg：  batchsize:2 , train_n: 48 val_n: 10

解决2：将代码添加到shapenet_part_loader.py：（生成self.datapath之后）
      for i in range(len(self.datapath)%32):
                  self.datapath.pop()

train_n:     54 val_n: 8 test_n: 6

Airplane50   train_n: 48 val_n: 6 test_n: 6
Airplane150  train_n: 144 val_n: 13 test_n: 13
Airplane300  train_n: 320 val_n: 10 test_n: 10

Airplane	 train_n: 2400 val_n: 145 test_n: 145
Chair       train_n: 3344 val_n: 201 test_n: 201
Earphone     train_n: 48 val_n: 10 test_n: 10
Table  train_n: 4736 val_n: 265 test_n: 265
"""

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot',  default='dataset/train', help='path to dataset')
parser.add_argument('--workers', type=int,default=0, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=2, help='input batch size') #设置 即num_epochs/batch_size的值是一个正整数。
parser.add_argument('--pnum', type=int, default=2048, help='the point number of a sample')
parser.add_argument('--crop_point_num',type=int,default=512,help='0 means do not use else use with this weight')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default= 2, help='number of epochs to train for') # default=201
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type = bool, default = False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--D_choose',type=int, default=1, help='0 not use D-net,1 use D-net')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop',type=float,default=0.2)
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
parser.add_argument('--point_scales_list',type=list,default=[2048,1024,512],help='number of points in each scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument('--wtl2',type=float,default=0.95,help='0 means do not use else use with this weight')
parser.add_argument('--cropmethod', default = 'random_center', help = 'random|center|random_center')
opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USE_CUDA = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
point_netG = _netG(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num)
point_netD = _netlocalD(opt.crop_point_num)
cudnn.benchmark = True
resume_epoch=0

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

if USE_CUDA:       
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    point_netG = torch.nn.DataParallel(point_netG)
    point_netD = torch.nn.DataParallel(point_netD)
    point_netG.to(device) 
    point_netG.apply(weights_init_normal)
    point_netD.to(device)
    point_netD.apply(weights_init_normal)
if opt.netG != '' :
    point_netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netG)['epoch']
if opt.netD != '' :
    point_netD.load_state_dict(torch.load(opt.netD,map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netD)['epoch']

        
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)


transforms = transforms.Compose(
    [
        d_utils.PointcloudToTensor(),
    ]
)


########################################
######   DataLoader  加载数据  ##############
###############################################

dset = shapenet_part_loader.PartDataset( root= dset_root,classification=True, class_choice=None, npoints=opt.pnum, split='train')
print(len(dset))
# assert dset
dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize, shuffle=True,num_workers = int(opt.workers))

#print(dataloader)

test_dset = shapenet_part_loader.PartDataset( root=test_dset_root,classification=True, class_choice=None, npoints=opt.pnum, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize, shuffle=True,num_workers = int(opt.workers))

print(len(test_dataloader))

#dset = ModelNet40Loader.ModelNet40Cls(opt.pnum, train=True, transforms=transforms, download = False)
#assert dset
#dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize, shuffle=True,num_workers = int(opt.workers))
#
#
#test_dset = ModelNet40Loader.ModelNet40Cls(opt.pnum, train=False, transforms=transforms, download = False)
#test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize, shuffle=True,num_workers = int(opt.workers))

#pointcls_net.apply(weights_init)


criterion = torch.nn.BCEWithLogitsLoss().to(device)
criterion_PointLoss = PointLoss().to(device)

# setup optimizer
optimizerD = torch.optim.Adam(point_netD.parameters(), lr=0.0001,betas=(0.9, 0.999),eps=1e-05,weight_decay=opt.weight_decay)
optimizerG = torch.optim.Adam(point_netG.parameters(), lr=0.0001,betas=(0.9, 0.999),eps=1e-05 ,weight_decay=opt.weight_decay)
schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=40, gamma=0.2)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=40, gamma=0.2)

real_label = 1
fake_label = 0

crop_point_num = int(opt.crop_point_num)
input_cropped1 = torch.FloatTensor(opt.batchSize, opt.pnum, 3)
label = torch.FloatTensor(opt.batchSize)


num_batch = len(dset) / opt.batchSize


def SaveOnePoint(OneData,saveFile):
    #np.savetxt(saveFile,OneData,fmt='%1.5f')
    pass

###########################
#  G-NET and T-NET
#  取数据
##########################

if opt.D_choose == 1:
    for epoch in range(resume_epoch,opt.niter):
        if epoch<30:
            alpha1 = 0.01
            alpha2 = 0.02
        elif epoch<80:
            alpha1 = 0.05
            alpha2 = 0.1
        else:
            alpha1 = 0.1
            alpha2 = 0.2

        # 调试时候，默认当batchsize = 2
        for i, data in enumerate(dataloader, 0):
            real_point, target = data                                      # 取数据 torch.Size([2, 2048, 3])
            
    
            batch_size = real_point.size()[0]
            real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)  # 新建 torch.Size([2, 2048, 3])
            input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)             # 新建 torch.Size([2, 2048, 3])，一个部分真实的点云
            input_cropped1 = input_cropped1.data.copy_(real_point)  # torch.Size([2, 2048, 3])   # copy real_point

            real_point = torch.unsqueeze(real_point, 1)             # torch.Size([2, 1, 2048, 3]) #延展
            input_cropped1 = torch.unsqueeze(input_cropped1,1)     # torch.Size([2, 1, 2048, 3]) #延展

            p_origin = [0,0,0]
            if opt.cropmethod == 'random_center':
                #Set viewpoints
                choice = [torch.Tensor([1,0,0]),
                          torch.Tensor([0,0,1]),
                          torch.Tensor([1,0,1]),
                          torch.Tensor([-1,0,0]),
                          torch.Tensor([-1,1,0])]        # 可以理解为， 字符串数组， 日后随机取出其中一个 “字符串”
                                                         # 作用： 丢失的点，在那个区域丢失

                for m in range(batch_size):
                    index = random.sample(choice,1)# Random choose one of the element
                    distance_list = []
                    p_center = index[0]                      # 随机取出 choice 中的一个元素

                    for n in range(opt.pnum):               # 点云的总点数， 公认1024个
                        distance_list.append(distance_squre(real_point[m,0,n],p_center)) #batch的第0个， 共2048点
                    distance_order = sorted(enumerate(distance_list), key  = lambda x:x[1])
                    #print(distance_order)     # index 和 距离值
                    #以上的作用： 以某个点为中心，和2048个点进行比较，，遍历哪个点是最近的， 就把那个512个点缺失掉。


                    for sp in range(opt.crop_point_num):#把一部分点去掉，并且保存其标签
                        input_cropped1.data[m,0,distance_order[sp][0]] = torch.FloatTensor([0,0,0])  #距离近的index，赋值为0,0,0
                        real_center.data[m,0,sp] = real_point[m,0,distance_order[sp][0]]             # 保存index下的实际距离

            label.resize_([batch_size,1]).fill_(real_label)
            real_point = real_point.to(device)               # torch.Size([2, 1, 2048, 3])
            real_center = real_center.to(device)             # torch.Size([2, 1, 512, 3])
            input_cropped1 = input_cropped1.to(device)       # torch.Size([2, 1, 2048, 3])  #距离近的index，赋值为0,0,0，默认是丢失的。
            label = label.to(device)                         # torch.Size([2, 1])


            ############################    小结 ###############################################################
            # torch.Size([2, 1, 2048, 3])     real_point ：源文件            input_cropped1： 经过造假后的点
            # torch.Size([2, 1, 512, 3])      real_center： 源文件值512个
            #  torch.Size([2, 1])             real_label                   label
            ###########################################################################################



            ############################
            # 以上准备好残缺的数据
            # 以下  3个下采样的点， 分支进行初步准备提特征（未进入神经网络）
            ###########################
            # real_center 分支
            real_center = Variable(real_center,requires_grad=True)  # variable可以反向传播
            real_center = torch.squeeze(real_center,1)

            real_center_key1_idx = utils.farthest_point_sample(real_center,64,RAN = False)
            real_center_key1 = utils.index_points(real_center,real_center_key1_idx)
            real_center_key1 =Variable(real_center_key1,requires_grad=True)
            real_center_key2_idx = utils.farthest_point_sample(real_center,128,RAN = True)
            real_center_key2 = utils.index_points(real_center,real_center_key2_idx)
            real_center_key2 =Variable(real_center_key2,requires_grad=True)
            #print(real_center_key2.shape)


            # input_cropped1  cropped2  cropped3 分支
            input_cropped1 = torch.squeeze(input_cropped1,1)

            input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
            input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)

            input_cropped3_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)
            input_cropped3     = utils.index_points(input_cropped1,input_cropped3_idx)

            input_cropped1 = Variable(input_cropped1,requires_grad=True)  #torch.Size([2, 2048, 3])
            input_cropped2 = Variable(input_cropped2,requires_grad=True)  #torch.Size([2, 1024, 3])
            input_cropped3 = Variable(input_cropped3,requires_grad=True)  #torch.Size([2, 512, 3])

            input_cropped2 = input_cropped2.to(device)
            input_cropped3 = input_cropped3.to(device)

            input_cropped  = [input_cropped1,input_cropped2,input_cropped3]
            ###### 自己添加： 临时： 保存数值 ###########
            # save_real = torch.squeeze(real_point,1)       #存在着 无故升维的情况
            # save_real = torch.squeeze(save_real,0)        #存在着batchsize维度
            # save_cropped1 = torch.squeeze(input_cropped1, 0) #存在着batchsize维度
            #
            #
            # SaveOnePoint(save_real, './try/real_2048.txt')
            # SaveOnePoint(save_cropped1, './try/crop_2048.txt')
            ####################


            ############################
            # 以上  3个下采样的点， 分支进行初步准备提特征（未进入神经网络）
            # 以下  进入 D network

            #  Train Discriminator  判别器
            # 此阶段判别器为主角 ，期望实际的值是1， 虚假的值是0， 判别器才准确
            # 在判别器中， 假图像进行计算损失时， 携带 fake.detach()  不参与反向传播

            # point_netD(real_center)   判别网络 +  criterion 损失(真点vs真值) + 损失backward

            # point_netG(input_cropped)  生成网络 +
            # point_netD(fake.detach())  判别网络  + criterion 损失(假点vs假值)  + 损失backward

            # 阶段总损失： errD = errD_real + errD_fake  无 backward
            # optimizerD.step()
            ###########################
            point_netG = point_netG.train()
            point_netD = point_netD.train()




            point_netD.zero_grad()
            real_center = torch.unsqueeze(real_center,1)
            output = point_netD(real_center)  # 进入神经网络   #real_center : torch.Size([2, 512, 3])
            errD_real = criterion(output,label)
            errD_real.backward()



            # # 跳入该函数中  _netG ，生成器的结果，所有是假的，fake
            fake_center1,fake_center2,fake  =point_netG(input_cropped)  # input_cropped : torch.Size([2, 2048, 3]) torch.Size([2, 1024, 3]
            fake = torch.unsqueeze(fake,1)
            label.data.fill_(fake_label)              #  tensor([[0.],[0]


            # 判别器  point_netD
            print('fake',fake.shape)
            output = point_netD(fake.detach())   #  在判别器中 该数值不进行 反向传播 # fake ： torch.Size([2, 1, 512, 3])
            errD_fake = criterion(output, label)  # 常规的损失值
            errD_fake.backward()                  # 常规的损失反向
            errD = errD_real + errD_fake
            optimizerD.step()                    # 常规的损失梯度优化


            ############################
            #  Train Generator  生成器
            #  此阶段是生成器为主角， 生成的希望和真的一样。 不断接近真的（这样生成器会提高）

            # point_netD(fake)  # 判别器 假点   损失（假点vs真值）
            #  errG_D   # 生成--到--判别中
            # 阶段总损失  errG 中含有 CD_LOSS  errG_D
            ###########################
            point_netG.zero_grad()
            label.data.fill_(real_label)
            output = point_netD(fake)
            errG_D = criterion(output, label)

            errG_l2 = 0
            CD_LOSS = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real_center,1))
            errG_l2 = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real_center,1))\
            +alpha1*criterion_PointLoss(fake_center1,real_center_key1)\
            +alpha2*criterion_PointLoss(fake_center2,real_center_key2)
            
            errG = (1-opt.wtl2) * errG_D + opt.wtl2 * errG_l2
            errG.backward()
            optimizerG.step()


            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f / %.4f/ %.4f'
                  % (epoch, opt.niter, i, len(dataloader),   errD.data, errG_D.data,errG_l2,errG,CD_LOSS))

            f=open('loss_PFNet.txt','a')
            f.write('\n'+'[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f / %.4f /%.4f'
                  % (epoch, opt.niter, i, len(dataloader),   errD.data, errG_D.data,errG_l2,errG,CD_LOSS))
            
            #########################
            # 经典的半路验证
            #   ：将数据传入到  生成网络中，计算CD损失
            ###########################

            if i % 10 ==0:
                print('After, ',i,'-th batch')
                f.write('\n'+'After, '+str(i)+'-th batch')
                for i, data in enumerate(test_dataloader, 0):
                    print()
                    real_point, target = data
                    
            
                    batch_size = real_point.size()[0]
                    real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)
                    input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)
                    input_cropped1 = input_cropped1.data.copy_(real_point)
                    real_point = torch.unsqueeze(real_point, 1)
                    input_cropped1 = torch.unsqueeze(input_cropped1,1)
                    
                    p_origin = [0,0,0]
                    
                    if opt.cropmethod == 'random_center':
                        choice = [torch.Tensor([1,0,0]),
                                  torch.Tensor([0,0,1]),
                                  torch.Tensor([1,0,1]),
                                  torch.Tensor([-1,0,0]),
                                  torch.Tensor([-1,1,0])]
                        
                        for m in range(batch_size):
                            index = random.sample(choice,1)
                            distance_list = []
                            p_center = index[0]
                            for n in range(opt.pnum):
                                distance_list.append(distance_squre(real_point[m,0,n],p_center))
                            distance_order = sorted(enumerate(distance_list), key  = lambda x:x[1])                         
                            for sp in range(opt.crop_point_num):
                                input_cropped1.data[m,0,distance_order[sp][0]] = torch.FloatTensor([0,0,0])
                                real_center.data[m,0,sp] = real_point[m,0,distance_order[sp][0]]

                    real_center = real_center.to(device)
                    real_center = torch.squeeze(real_center,1)
                    input_cropped1 = input_cropped1.to(device) 
                    input_cropped1 = torch.squeeze(input_cropped1,1)
                    input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
                    input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)
                    input_cropped3_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)
                    input_cropped3     = utils.index_points(input_cropped1,input_cropped3_idx)
                    input_cropped1 = Variable(input_cropped1,requires_grad=False)
                    input_cropped2 = Variable(input_cropped2,requires_grad=False)
                    input_cropped3 = Variable(input_cropped3,requires_grad=False)
                    input_cropped2 = input_cropped2.to(device)
                    input_cropped3 = input_cropped3.to(device)      
                    input_cropped  = [input_cropped1,input_cropped2,input_cropped3]

                    ###################
                    # 以上是将原材料得到 三种不同的尺度
                    # 以下是进入到 G 网络中 生成器 生成点云，计算CD 损失
                    ###################
                    point_netG.eval()
                    fake_center1,fake_center2,fake  =point_netG(input_cropped)   # 跳入该函数中  _netG
                    CD_loss = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real_center,1))
                    print('test result:',CD_loss)
                    f.write('\n'+'test result:  %.4f'%(CD_loss))
                    break
            f.close()

        schedulerD.step()
        schedulerG.step()
        if epoch% 50 == 0:     # 原先10
            torch.save({'epoch':epoch+1,
                        'state_dict':point_netG.state_dict()},
                        './Checkpoint/point_netG'+str(epoch)+'.pth' )
            torch.save({'epoch':epoch+1,
                        'state_dict':point_netD.state_dict()},
                        './Checkpoint/point_netD'+str(epoch)+'.pth' )

#
#############################
## ONLY G-NET
############################
else:
    for epoch in range(resume_epoch,opt.niter):
        if epoch<30:
            alpha1 = 0.01
            alpha2 = 0.02
        elif epoch<80:
            alpha1 = 0.05
            alpha2 = 0.1
        else:
            alpha1 = 0.1
            alpha2 = 0.2

        for i, data in enumerate(dataloader, 0):

            real_point, target = data


            batch_size = real_point.size()[0]
            real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)
            input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)
            input_cropped1 = input_cropped1.data.copy_(real_point)
            real_point = torch.unsqueeze(real_point, 1)
            input_cropped1 = torch.unsqueeze(input_cropped1,1)
            p_origin = [0,0,0]
            if opt.cropmethod == 'random_center':
                choice = [torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]
                for m in range(batch_size):
                    index = random.sample(choice,1)
                    distance_list = []
                    p_center = index[0]
                    for n in range(opt.pnum):
                        distance_list.append(distance_squre(real_point[m,0,n],p_center))
                    distance_order = sorted(enumerate(distance_list), key  = lambda x:x[1])

                    for sp in range(opt.crop_point_num):
                        input_cropped1.data[m,0,distance_order[sp][0]] = torch.FloatTensor([0,0,0])
                        real_center.data[m,0,sp] = real_point[m,0,distance_order[sp][0]]
            real_point = real_point.to(device)
            real_center = real_center.to(device)
            input_cropped1 = input_cropped1.to(device)
            ############################
            # (1) data prepare
            ###########################
            real_center = Variable(real_center,requires_grad=True)
            real_center = torch.squeeze(real_center,1)
            real_center_key1_idx = utils.farthest_point_sample(real_center,64,RAN = False)
            real_center_key1 = utils.index_points(real_center,real_center_key1_idx)
            real_center_key1 =Variable(real_center_key1,requires_grad=True)

            real_center_key2_idx = utils.farthest_point_sample(real_center,128,RAN = True)
            real_center_key2 = utils.index_points(real_center,real_center_key2_idx)
            real_center_key2 =Variable(real_center_key2,requires_grad=True)

            input_cropped1 = torch.squeeze(input_cropped1,1)
            input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
            input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)
            input_cropped3_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)
            input_cropped3     = utils.index_points(input_cropped1,input_cropped3_idx)
            input_cropped1 = Variable(input_cropped1,requires_grad=True)
            input_cropped2 = Variable(input_cropped2,requires_grad=True)
            input_cropped3 = Variable(input_cropped3,requires_grad=True)
            input_cropped2 = input_cropped2.to(device)
            input_cropped3 = input_cropped3.to(device)
            input_cropped  = [input_cropped1,input_cropped2,input_cropped3]
            point_netG = point_netG.train()
            point_netG.zero_grad()
            fake_center1,fake_center2,fake  =point_netG(input_cropped)
            fake = torch.unsqueeze(fake,1)
            ############################
            # (3) Update G network: maximize log(D(G(z)))
            ###########################

            CD_LOSS = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real_center,1))

            errG_l2 = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real_center,1))\
            +alpha1*criterion_PointLoss(fake_center1,real_center_key1)\
            +alpha2*criterion_PointLoss(fake_center2,real_center_key2)

            errG_l2.backward()
            optimizerG.step()
            print('[%d/%d][%d/%d] Loss_G: %.4f / %.4f '
                  % (epoch, opt.niter, i, len(dataloader),
                      errG_l2,CD_LOSS))
            f=open('loss_PFNet.txt','a')
            f.write('\n'+'[%d/%d][%d/%d] Loss_G: %.4f / %.4f '
                  % (epoch, opt.niter, i, len(dataloader),
                      errG_l2,CD_LOSS))
            f.close()
        schedulerD.step()
        schedulerG.step()

        if epoch% 50 == 0:
            torch.save({'epoch':epoch+1,
                        'state_dict':point_netG.state_dict()},
                        'Checkpoint/point_netG'+str(epoch)+'.pth' )
 

    
        
