import os
import random
import torch.nn.parallel
import torch.utils.data
from Code_PFNet.utils import PointLoss
import Code_PFNet.data_utils as d_utils
from Code_PFNet.shapenet_part_loader  import PartDataset
from Code_PFNet.model_PFNet import MouldnetD,MouldnetG  # _netG 生成G： generator     MouldnetD 判别器D： discriminator
import Code_PFNet.utils_otherdef as PFotherdef
import numpy as np



def OneBatchVal():

    pass

def SavePthLog(epoch,point_netG):
    if epoch % 25 == 0:  # 原先10
        torch.save({'epoch': epoch + 1,
                    'state_dict': point_netG.state_dict()},
                   './Pth_FPnet/point_netG' + str(epoch) + '.pth')
        # torch.save({'epoch': epoch + 1,
        #             'state_dict': point_netD.state_dict()},
        #            './FPnetPth/point_netD' + str(epoch) + '.pth')


def  savePONIT(NAME,train_onedata):
    savepath = './output_testpoint/'
    if(os.path.exists(savepath) == False):
        os.makedirs(savepath)
    temp_gt = torch.squeeze(train_onedata[0].cpu()) # torch.squeeze(gt.cpu())
    np.savetxt(savepath  + NAME +'.txt', temp_gt.detach().numpy(),fmt="%0.3f")



# 统一   gt原始真实   part整体的的部分   drop  整体丢失的
# if _args.D_choose == 1:

def RunOnebatch(_args, device, train_len, train_onedata,val_loader, datakind,epoch,PFnetG, PFnetD,i,
                criterion, cri_PointLoss, optimizerD, optimizerG,alpha1,alpha2,     G_treegan, D_treegan,
                                                                                                   optimizerG_treegan,
                                                                                                   optimizerD_treegan,
                                                                                                   GP, knn_loss,
                                                                                                   w_train,
                                                                                                   epoch_d_loss,
                                                                                                   epoch_g_loss):


         # point_gt, target = data                                       # 取数据 torch.Size([2, 2048, 3])
        point_gt,input_part1,point_drop = PFotherdef.GetSplitPoint(_args,train_onedata,device) #分离数据
         # # PFotherdef.printdictKey(data)
        # point_gt,input_part1,point_drop, = PFotherdef.GetRPMSplitPoint(_args,train_onedata,datakind,device) #分离数据

        batch_size, _, _ = train_onedata.shape
        label = PFotherdef.SaveGanLabel(batch_size, device)                    # label
        input_part_list = PFotherdef.GetInputlackList(_args,input_part1,device,True)            #三个层次缺失点云

        point_drop, point_drop1,point_drop2= PFotherdef.Get_PointDrop_Level(point_drop,device)         #三个层次丢失的点云

        # CD_LOSS = cri_PointLoss(torch.squeeze(point_drop, 1), torch.squeeze(point_drop, 1))
        # return point_gt, input_part1, point_drop, point_drop,  CD_LOSS, epoch_g_loss
        #############################################################################
        # -------------------- Discriminator -------------------- #
        D_treegan.zero_grad()
        for d_iter in range(_args.D_iter):
            with torch.no_grad():
                output_drop1,output_drop2,output_drop3  =PFnetG(input_part_list)

            D_real, _ = D_treegan(point_drop)   #  drop_gt
            D_fake, _ = D_treegan(output_drop3) #drop_ fake
            gp_loss = GP(D_treegan, point_drop.data, output_drop3.data)
            savePONIT('real2048', point_drop)
            savePONIT('fake2048', output_drop3)

            D_realm = D_real.mean()
            D_fakem = D_fake.mean()
            d_loss = -D_realm + D_fakem
            d_loss_gp = d_loss + gp_loss
            # times weight before backward
            d_loss *= w_train
            d_loss_gp.backward()
            optimizerD_treegan.step()

        epoch_d_loss.append(d_loss.item())
    # ---------------------- Generator ---------------------- #

    #############################################################
         ############################
         #  Train Generator  生成器
         #  此阶段是生成器为主角， 生成的希望和真的一样。 不断接近真的（这样生成器会提高）
         ###########################
        PFnetG.zero_grad()
        output_drop1, output_drop2, output_drop3 = PFnetG(input_part_list)

        CD_LOSS = cri_PointLoss(torch.squeeze(output_drop3,1),torch.squeeze(point_drop,1))
        errG_l2 = cri_PointLoss(torch.squeeze(output_drop3,1),torch.squeeze(point_drop,1))\
         +alpha1*cri_PointLoss(output_drop1,point_drop1)\
         +alpha2*cri_PointLoss(output_drop2,point_drop2)

        knn_loss = knn_loss(output_drop3)
        g_loss =   _args.knn_scalar * knn_loss
        g_loss *= w_train

        # label.data.fill_(1)  # real_label
        # output = D_treegan(output_drop3.detach())
        # errG_D = criterion(output, label)
        # xxxxxxx = (1-_args.wtl2) * errG_D

        errG =  _args.wtl2 * errG_l2 + g_loss

        errG.backward()
        optimizerG.step()

       #  #和RPM联合之后
       # # PFnetG = PFotherdef.TestAndOpenRPM(i,epoch,_args,train_len,val_loader,datakind,errD,errG_D,errG_l2,errG,CD_LOSS,cri_PointLoss,device,PFnetG)

        return point_gt,input_part1,point_drop,output_drop3,CD_LOSS,epoch_g_loss



#############################
## ONLY G-NET  这里暂时省
############################

def InitPFnet(_args,device):


    PFnetG = MouldnetG(_args.num_scales, _args.each_scales_size, _args.point_scales_list, _args.point_drop_num)
    PFnetD = MouldnetD(_args.point_drop_num)

    USE_CUDA = True
    if USE_CUDA:
        PFnetG = torch.nn.DataParallel(PFnetG,device_ids=_args.gpu_ids) # 也可以是两个 device_ids=[1,2]
        PFnetD = torch.nn.DataParallel(PFnetD,device_ids=_args.gpu_ids) # 也可以是两个
        PFnetG.to(device)
        PFnetG.apply(PFotherdef.weights_init_normal)
        PFnetD.to(device)
        PFnetD.apply(PFotherdef.weights_init_normal)

    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    cri_PointLoss = PointLoss().to(device)

    # setup optimizer
    optimizerD = torch.optim.Adam(PFnetD.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-05,
                                  weight_decay=_args.weight_decay)
    optimizerG = torch.optim.Adam(PFnetG.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-05,
                                  weight_decay=_args.weight_decay)
    schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=40, gamma=0.2)
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=40, gamma=0.2)

    return PFnetG,PFnetD,criterion,cri_PointLoss,optimizerD,optimizerG,schedulerD,schedulerG

#
# def OwnFunction():  # 当函数独立的时候， 这里才能用的上   当和RPM合并时， 这里用不上
#     dset_root = './datasets/shapenet_part_Airplane50/'  # split='train'
#     test_dset_root = './datasets/shapenet_part_Airplane50/'  # split='test'
#
#     parser = rpmnet_train_arguments()
#     _args = parser.parse_args()
#
#     ########################################
#     ######   DataLoader  加载数据  ##############
#     ###############################################
#
#     dset = PartDataset(root=dset_root, classification=True, class_choice=None, npoints=_args.num_points, split='train')
#     dataloader = torch.utils.data.DataLoader(dset, batch_size=_args.train_batch_size, shuffle=True,
#                                              num_workers=int(_args.num_workers))
#
#     test_dset = PartDataset(root=test_dset_root, classification=True, class_choice=None, npoints=_args.num_points,
#                             split='test')
#     val_loader = torch.utils.data.DataLoader(test_dset, batch_size=_args.train_batch_size, shuffle=True,
#                                                   num_workers=int(_args.num_workers))
#     ###################
#     if _args.manualSeed is None:
#         _args.manualSeed = random.randint(1, 10000)
#     print("Random Seed: ", _args.manualSeed)
#     random.seed(_args.manualSeed)
#     torch.manual_seed(_args.manualSeed)
#     if _args.cuda:
#         torch.cuda.manual_seed_all(_args.manualSeed)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#
#     return _args,device,dataloader,val_loader

if __name__ == '__main__':

    _args,device,dataloader,val_loader = OwnFunction()

    PFnetG,PFnetD,\
    criterion,cri_PointLoss,optimizerD,optimizerG,schedulerD,schedulerG = InitPFnet(_args,device)



    for epoch in range(0,_args.epochs):
        alpha1, alpha2 = PFotherdef.PF_epoch_alpha(epoch)
        for i, data in enumerate(dataloader, 0):

            PFnetG = RunOnebatch(_args, device, dataloader, val_loader, epoch,PFnetG, PFnetD,i,data,
                criterion, cri_PointLoss, optimizerD, optimizerG,alpha1,alpha2)

        schedulerD.step()
        schedulerG.step()
        PFotherdef.SaveFPNetEpoch(epoch, PFnetG, PFnetD)


