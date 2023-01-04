

import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np

from Code_pointnet2.data_utils.Amain_onetree_abstractleaf import SplitLeafTest,SplitLeaf
import test_partseg
import test_PF_CD
from  A3_FPS_Npoint2Npoint import sample_and_group as FPS
# from Chamfer3D_L1L2.loss_utils import get_loss  as get_lossl1l2



def avgloss(Aloss,Bloss,Closs):
    avgepoch_Aloss = sum(Aloss) / len(Aloss)  # 所有的损失 / 摊平到每一个上
    CD = float('%.5f' % avgepoch_Aloss)
    avgepoch_Aloss = sum(Bloss) / len(Bloss)  # 所有的损失 / 摊平到每一个上
    Gt_Pre = float('%.5f' % avgepoch_Aloss)
    avgepoch_Aloss = sum(Closs) / len(Closs)  # 所有的损失 / 摊平到每一个上
    Pre_Gt = float('%.5f' % avgepoch_Aloss)
    print(CD, Gt_Pre, Pre_Gt)
    print("CD:{} , Gt_Pre:{} , Pre_Gt:{}".format(float(CD), float(Gt_Pre), float(Pre_Gt)))

def  saveFPpoint(epoch,name, data):
    savepath = './output_test/'

    if len(data.shape) == 4:
        data = data.squeeze(1)

    Newdata = data[0].cpu().detach().numpy()
    np.savetxt(savepath + str(epoch) + name + '.txt', Newdata, fmt="%0.3f")

def WriteTxtLine(filename , context):

    with open(filename, 'a+',encoding='utf-8') as fp:
        fp.write('\n'+ context )
        fp.close()


def saveresult(i, temp1536, fake):

    saveFPpoint(i, "_3_test" + 'real_center', temp1536)
    saveFPpoint(i, "_4_test" + 'fake', fake)

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)  #求中心，对pc数组的每行求平均值，通过这条函数最后得到一个1×3的数组[x_mean,y_mean,z_mean];
    pc = pc - centroid  #点云平移  或  # 求得每一点到中点的绝对距离
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))  # 将同一列的元素取平方相加得(x^2+y^2+z^2)，再开方，取最大，得最大标准差
    #pc ** 2 平移后的点云求平方   #np.sum(pc ** 2, axis=1)：每列求和
    pc = pc / m   # 归一化，这里使用的是Z-score标准化方法，即为(x-mean)/std
    return pc
def pc_normalize_saveage(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)              #对pc数组的每行求平均值;
    pc = pc - centroid                          # 求得每一点到中点的绝对距离
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))  #得最大标准差
    pc = pc / m                                # 归一化Z-score标准化，即为(x-mean)/std
    return pc,m,centroid
# https://blog.csdn.net/qq_36387467/article/details/122365148
# 点云的归一化与反归一化
def pc_normalize_reverse(pc,m,centroid):
    ret = pc * m + centroid
    return ret

def del_tensor_0_cloumn(Cs):

    idx = torch.all(Cs[..., :] == 0, axis=1)

    index=[]
    for i in range(idx.shape[0]):
        if not idx[i].item():
            index.append(i)


    index=torch.tensor(index)

    Cs = torch.index_select(Cs, 0, index)
    return Cs



if __name__ == '__main__':

    ##########  pointnet++
    args = test_partseg.parse_args()
    experiment_dir,seg_classes,seg_label_to_cat = test_partseg.onint(args)
    testDataLoader = test_partseg.load_data(args)
    #PointNetModel = test_partseg.MoudleLoad(args,experiment_dir)
    # PointWithno_grad(args,seg_classes,seg_label_to_cat,testDataLoader,PointNetModel)

    ##########  FPnet
    device = test_PF_CD.Inintdevice()
    point_netG,criterion_PointLoss = test_PF_CD.InitPFnet(args,device)



    with torch.no_grad():

        ##########  pointnet++
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(args.num_part)]
        total_correct_class = [0 for _ in range(args.num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}

        ##########  FPnet
        errG_min = 100
        n = 0
        CD = 0
        Gt_Pre = 0
        Pre_Gt = 0
        IDX = 1

        Aloss = []
        Bloss = []
        Closs = []

        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):

            #####################################################################
            # ##########  pointnet++
            # target,cur_pred_val_logits = test_partseg.Get_PreTarget(args,points,label,target,PointNetModel)
            # cur_pred_val, target,shape_ious = test_partseg.Get_ShapeIous(args, points, seg_classes, seg_label_to_cat, target,
            #                            cur_pred_val_logits,total_correct, total_seen, total_seen_class, total_correct_class, shape_ious)

            #batchtree_abstractleaf = SplitLeafTest(args,batch_id,cur_pred_val,points)
            batchtree_abstractleaf,_ ,batchtree_bar= SplitLeafTest(args,batch_id,target,points)
            # print(batchtree_abstractleaf.shape)
            #####################################################################


            onetreeleafake = []
            onetreeleafreal = []
            #####################################################################
            for leafidx in range(batchtree_abstractleaf.shape[0]):

                real_point = batchtree_abstractleaf[leafidx:leafidx + 1, :, :]

                templist = []

                for i in range(real_point.shape[0]):
                    temptorch = batchtree_abstractleaf[i, :, :]
                    temptorch = temptorch.data.numpy()
                    temptorch,m_onesave,cen_onesave = pc_normalize_saveage(temptorch)
                    temptorch = temptorch.reshape(-1, 1024, 3)
                    templist.append(temptorch)
                real_point = np.concatenate(templist, 0)
                real_point = torch.tensor(real_point).float()
                ################################################################################

                real_point = torch.unsqueeze(real_point, 1)
                batch_size = real_point.size()[0]

                IDX, input_cropped1, real_center, distance_order = test_PF_CD.Getreal_center(args,IDX,batch_size,real_point)
                input_cropped_partial = test_PF_CD.Getinput_cropped_partial(args,device, batch_size,real_point, distance_order)
                input_croppedlist = test_PF_CD.GetInputlackList(args, input_cropped1, device)

                #####################################################################
                # https://blog.csdn.net/lichaoqi1/article/details/121498578
                # pytorch删除行全为0的元素
                xcxvcxvc = input_cropped1.squeeze(0)
                xcxvcxvc = xcxvcxvc.squeeze(0)
                temp1536 = del_tensor_0_cloumn(xcxvcxvc)
                temp1536 = temp1536.unsqueeze(0)
                #####################################################################


                fake = test_PF_CD.GetGnetPoint(input_croppedlist,point_netG)
                CD, Gt_Pre, Pre_Gt = test_PF_CD.GetLossdist(0,device,criterion_PointLoss,input_cropped_partial, real_center, fake, real_point, CD, Gt_Pre, Pre_Gt)

                # ##############################################
                ## 1 片叶子的合成
                temp1536 = pc_normalize_reverse(temp1536[:, :, :],m_onesave,cen_onesave)
                fake = pc_normalize_reverse(fake[:, :, :],m_onesave,cen_onesave)
                oneleaf = torch.cat((temp1536,fake ), 1)
                onetreeleafake.append(oneleaf)
                real_point = real_point.squeeze(0)
                onetreeleafreal.append(real_point)
                #saveresult(batch_id, temp1536, fake)
                # ##############################################
                # #############################################
                print(CD)
                Aloss.append(CD)
                Bloss.append(Gt_Pre)
                Closs.append(Pre_Gt)

                #############################################

            real = torch.cat(onetreeleafreal, 1)
            fake = torch.cat(onetreeleafake, 1)
            endreal = torch.cat((real, batchtree_bar[:,:,0:3]), 1)
            endfake = torch.cat((fake, batchtree_bar[:,:,0:3]), 1)
            endreal = FPS(endreal, 2048)  # 输入需要三个维度
            endfake = FPS(endfake, 2048)  # 输入需要三个维度
            #                 real_center = real_center.cuda()
            xxxxxx = get_lossl1l2(endfake, endreal, sqrt=True)  # False True


            # print(CD,Gt_Pre,Pre_Gt)
            # saveresult(batch_id,real_point,input_cropped1,real_center,fake,float(CD))
            # print(111)
            #############################################
        avgloss(Aloss,Bloss,Closs) # 计算并打印 损失

