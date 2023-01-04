

import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np

from Code_pointnet2.data_utils.Amain_onetree_abstractleaf import SplitLeafTest,SplitLeaf
from  Code_pointnet2.data_utils.A3_FPS_Npoint2Npoint import sample_and_group as FPS
import test_partseg
import test_PF_CD





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
    #Newdata = data
    np.savetxt(savepath + str(epoch) + name + '.txt', Newdata, fmt="%0.3f")

def WriteTxtLine(filename , context):

    with open(filename, 'a+',encoding='utf-8') as fp:
        fp.write('\n'+ context )
        fp.close()


def saveresult(i,leafidx, real_point, input_cropped1, real_center, fake,CD):
    cd_val= format(CD, '.5f')
    cd_str = str(cd_val)

    # saveFPpoint(i, "test_1" + 'real_point', real_point)
    # saveFPpoint(i, "test_2" + 'input_cropped1', input_cropped1)
    saveFPpoint(i, str(leafidx)+ "test_3" + 'drop', real_point)
    saveFPpoint(i, str(leafidx)+ "test_4" + 'fake', fake)

    # WriteTxtLine('./output_test/result.txt', str(i) + " " + cd_str)



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

if __name__ == '__main__':

    ##########  pointnet++
    args = test_partseg.parse_args()
    experiment_dir,seg_classes,seg_label_to_cat = test_partseg.onint(args)
    testDataLoader = test_partseg.load_data(args)
   # PointNetModel = test_partseg.MoudleLoad(args,experiment_dir)
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
            ##########  pointnet++
            # target,cur_pred_val_logits = test_partseg.Get_PreTarget(args,points,label,target,PointNetModel)
            # cur_pred_val, target,shape_ious = test_partseg.Get_ShapeIous(args, points, seg_classes, seg_label_to_cat, target,
            #                            cur_pred_val_logits,total_correct, total_seen, total_seen_class, total_correct_class, shape_ious)

            #batchtree_abstractleaf = SplitLeafTest(args,batch_id,cur_pred_val,points)
            batchtree_abstractleaf,_ = SplitLeafTest(args,batch_id,target,points)
            # print(batchtree_abstractleaf.shape)
            #####################################################################

            templist = []
            for i in range(batchtree_abstractleaf.shape[0]):
                temptorch = batchtree_abstractleaf[i, :, :]
                temptorch = temptorch.data.numpy()
                temptorch, m, centroid = pc_normalize_saveage(temptorch)
                temptorch = temptorch.reshape(-1, 1024, 3)
                templist.append(temptorch)
            batchtree_abstractleaf = np.concatenate(templist, 0)
            batchtree_abstractleaf = torch.tensor(batchtree_abstractleaf).float()
            ################################################################################

            #####################################################################
            #real_point = batchtree_abstractleaf

            for leafidx in range(batchtree_abstractleaf.shape[0]):
                real_point = batchtree_abstractleaf[leafidx:leafidx+1,:,:]

                real_point = FPS(real_point,1024-256)
                x = torch.zeros([1,256,3],dtype=torch.float)
                real_point = torch.cat((real_point,x),1)
                #
                # templist = []
                # for i in range(real_point.shape[0]):
                #     temptorch = batchtree_abstractleaf[i, :, :]
                #     temptorch = temptorch.data.numpy()
                #     temptorchpc,m,centroid = pc_normalize_saveage(temptorch)
                #     temptorch = temptorch.reshape(-1, 1024, 3)
                #     templist.append(temptorch)
                # real_point = np.concatenate(templist, 0)
                # real_point = torch.tensor(real_point).float()
                # # ################################################################################


                real_point = torch.unsqueeze(real_point, 1)

                # IDX, input_cropped1, real_center, distance_order = test_PF_CD.Getreal_center(args,IDX,batch_size,real_point)
                # input_cropped_partial = test_PF_CD.Getinput_cropped_partial(args,device, batch_size,real_point, distance_order)
                real_center = real_point
                input_cropped1 = real_point
                input_cropped_partial = real_point
                input_croppedlist = test_PF_CD.GetInputlackList(args, input_cropped1, device)

                fake = test_PF_CD.GetGnetPoint(input_croppedlist,point_netG)
                CD, Gt_Pre, Pre_Gt = test_PF_CD.GetLossdist(0,device,criterion_PointLoss,input_cropped_partial, real_center, fake, real_point, CD, Gt_Pre, Pre_Gt)

                # #############################################
                print(CD)
                Aloss.append(CD)
                Bloss.append(Gt_Pre)
                Closs.append(Pre_Gt)

                # #############################################
                # fake = fake.cpu().detach().numpy()
                # fake  = pc_normalize_reverse(fake[0, :, :],m,centroid)
                saveresult(batch_id,leafidx,real_point,input_cropped1,real_center,fake,float(CD))
                # #############################################

        avgloss(Aloss,Bloss,Closs) # 计算并打印 损失

