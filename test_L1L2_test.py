import os.path

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
from Chamfer3D_L1L2.loss_utils import get_loss  as get_lossl1l2


from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
from Chamfer3D_L1L2.loss_utils import get_loss  as get_lossl1l2

chamfer_dist = chamfer_3DDist()
def chamfer(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


def chamfer_single_side(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(d1)
    return d1


def chamfer_single_side_sqrt(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(torch.sqrt(d1))
    return d1


def get_loss(pcds_pred, gt, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
    else:
        CD = chamfer

    P3 = pcds_pred
    cd3 = CD(P3, gt)

    return  cd3* 1e3


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def pc_normalize_saveage(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc,m,centroid

# Normalization and anti-normalization of point clouds
def pc_normalize_reverse(pc,m,centroid):
    ret = pc * m + centroid
    return ret


if __name__ == '__main__':

    ##########  pointnet++
    args = test_partseg.parse_args()
    experiment_dir,seg_classes,seg_label_to_cat = test_partseg.onint(args)
    testDataLoader = test_partseg.load_data(args)
    PointNetModel = test_partseg.MoudleLoad(args,experiment_dir)

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
        Dloss_L_CD_true = []
        Dloss_L_CD_false = []


        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):

            #####################################################################
            # ##########  pointnet++
            target,cur_pred_val_logits = test_partseg.Get_PreTarget(args,points,label,target,PointNetModel)
            cur_pred_val, target, shape_ious, shape_ious_Airplane= test_partseg.Get_ShapeIous(args, points, seg_classes,
                                                                                              seg_label_to_cat, target,cur_pred_val_logits,total_correct, total_seen, total_seen_class, total_correct_class, shape_ious)

            print("point++",shape_ious_Airplane)
            # args, epoch, points, OutbatchLabel
            batchtree_abstractleaf,_ ,batchtree_bar= SplitLeafTest(args,0,points,cur_pred_val)
            #####################################################################
            onetreeleafgt      = []
            onetreeleafpart    = []
            onetreeleafdropgt  = []
            onetreeleafdropfake= []
            onetreeleafake     = []

            #####################################################################
            length = batchtree_abstractleaf.shape[0]
            print('length',length)

            for leafidx in range(batchtree_abstractleaf.shape[0]): #  Here are N leaves from a family of plants

                real_point = batchtree_abstractleaf[leafidx:leafidx + 1, :, :]
                # print(leafidx)

                # ######  Get normalized  ###############
                templist = []
                temptorch = batchtree_abstractleaf[leafidx, :, :]
                temptorch = temptorch.data.numpy()
                temptorch,m_onesave,cen_onesave = pc_normalize_saveage(temptorch)
                temptorch = temptorch.reshape(-1, 1024, 3)
                templist.append(temptorch)
                #################################

                # print(m_onesave,cen_onesave)
                real_point = np.concatenate(templist, 0)
                real_point = torch.tensor(real_point).float()
                ################################################################################

                real_point = torch.unsqueeze(real_point, 1)
                batch_size = real_point.size()[0]
                length = batch_size

                ############
                #  real_point = input_partial3584 + real_center
                #  fake_leaf  =  input_partial3584 + fake
                #######
                IDX, input_cropped1, real_center, distance_order = test_PF_CD.Getreal_center(args,IDX,batch_size,real_point)
                input_partial3584 = test_PF_CD.Getinput_cropped_partial(args,device, batch_size,real_point, distance_order)
                input_croppedlist = test_PF_CD.GetInputlackList(args, input_cropped1, device)

                fake = test_PF_CD.GetGnetPoint(input_croppedlist,point_netG)  # PFnet 网络

                ############
                #  real_point = input_partial3584 + real_center
                #  fake_leaf  =  input_partial3584 + fake
                # ###########
                #  1 leaf synthesis
                input_partial3584 = input_partial3584.cuda()
                real_center = real_center.cuda()
                endgt      = torch.cat((input_partial3584, real_center.detach()), 1) #1024
                endpart    = input_partial3584
                enddropgt  = real_center.detach()
                enddropfake= fake.detach()
                endfake    = torch.cat((input_partial3584, fake.detach()), 1) # 1024

                # Inverse normalization
                endgt      =pc_normalize_reverse(endgt,      torch.tensor(m_onesave).cuda(),torch.tensor(cen_onesave).cuda())
                endpart    =pc_normalize_reverse(endpart,    torch.tensor(m_onesave).cuda(),torch.tensor(cen_onesave).cuda())
                enddropgt  =pc_normalize_reverse(enddropgt,  torch.tensor(m_onesave).cuda(),torch.tensor(cen_onesave).cuda())
                enddropfake=pc_normalize_reverse(enddropfake,torch.tensor(m_onesave).cuda(),torch.tensor(cen_onesave).cuda())
                endfake    =pc_normalize_reverse(endfake,    torch.tensor(m_onesave).cuda(),torch.tensor(cen_onesave).cuda())

                onetreeleafgt.append(endgt)
                onetreeleafpart.append(endpart)
                onetreeleafdropgt.append(enddropgt)
                onetreeleafdropfake.append(enddropfake)
                onetreeleafake.append(endfake)

            batchtree_bar = batchtree_bar[:, :, 0:3].cuda()


            allleafgt  = torch.cat(onetreeleafgt, 1)
            allleafpart = torch.cat(onetreeleafpart, 1)
            allleafdropgt = torch.cat(onetreeleafdropgt, 1)
            allleafdropfake = torch.cat(onetreeleafdropfake, 1)
            allleafake = torch.cat(onetreeleafake, 1)

            labelsum0  = 20480 - batchtree_bar.shape[1]  # 内存不够用，暂时用这里的折半
            allleafgt       =FPS(torch.cat((allleafgt,allleafgt), 1), labelsum0)
            allleapart      =FPS(torch.cat((allleafpart,allleafpart), 1), labelsum0)
            allleafdropgt   =FPS(torch.cat((allleafdropgt,allleafdropgt), 1), labelsum0)
            allleafdropgt = FPS(torch.cat((allleafdropgt, allleafdropgt), 1), labelsum0)
            allleafdropfake =FPS(torch.cat((allleafdropfake,allleafdropfake), 1), labelsum0)
            allleafdropfake =FPS(torch.cat((allleafdropfake,allleafdropfake), 1), labelsum0)
            allleafake      =FPS(torch.cat((allleafake,allleafake), 1), labelsum0)

             #########################################################
            endgt      = torch.cat((batchtree_bar,allleafgt), 1)
            endpart    = torch.cat((batchtree_bar,allleapart), 1)
            enddropgt  = allleafdropgt
            enddropfake= allleafdropfake
            endfake    = torch.cat((batchtree_bar,allleafake), 1)

            tempval = chamfer(endfake, endgt).item() * 1e3
            Aloss.append(float(tempval))

            # method 1 ： routine method
            temp_true = get_lossl1l2(endfake, endgt, sqrt=True)  # False True
            temp_false = get_lossl1l2(endfake, endgt, sqrt=False)  # False True
            Dloss_L_CD_true.append(float(temp_true))
            Dloss_L_CD_false.append(float(temp_false))

            # method 2： PFnet
            dist_all = dist1 = dist2 = Gt_Pre = Pre_Gt
            tempval = chamfer(endfake, endgt).item() * 1e3
            Aloss.append(tempval)
            print(dist_all * 1e3, temp_true, temp_false)

        avgepoch_L_CD_true = sum(Dloss_L_CD_true) / len(Dloss_L_CD_true)
        avgepoch_L_CD_false = sum(Dloss_L_CD_false) / len(Dloss_L_CD_false)
        avgepoch_Aloss = sum(Aloss) / len(Aloss)
        print(avgepoch_L_CD_true, avgepoch_L_CD_false)

