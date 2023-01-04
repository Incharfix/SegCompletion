
from tqdm import tqdm
import numpy as np
import torch
import train_partseg

import arguments

import Code_treegan.train_treegan as file_treegan
import Train_PFNet as file_PFNet
import Code_PFNet.utils_otherdef as PFotherdef
import open3d as o3d
from Code_pointnet2.data_utils.Amain_onetree_abstractleaf import SplitLeaf
import os



# @ 20200809 调整 数据集 -  这里是能够正常运行的
"""
训练所需设置参数：
--log_dir pointnet2_part_seg_msg
--root ../data_ours/trainPoint/
"""



def WriteTxtLine(filename , context):

    with open(filename, 'a+',encoding='utf-8') as fp:
        fp.write('\n'+ context )
        fp.close()
def  saveFPpoint(epoch,name, data):
    savepath = './output_testpoint/'

    if len(data.shape) == 4:
        data = data.squeeze(1)

    Newdata = data[0].cpu().detach().numpy()
    np.savetxt(savepath + str(epoch) + name + '.txt', Newdata, fmt="%0.3f")



def noise_Gaussian(points, std):
    noise = np.random.normal(0, std, points.shape)
    out = points + noise
    return out


def addPointnumber(Nowpoint,needpoint,data):

    count_times = int(needpoint / Nowpoint)


    target_np_noised = data
    for  i  in range(count_times +1 ):

        temp = noise_Gaussian(data, 0.00000001) # 增加的上采样的点
        target_np_noised = np.vstack([temp, target_np_noised])

    # print('add  ',target_np_noised.shape)
    return  target_np_noised

def RandomSample(input_part,partsumpoint):

    batchs,points,others= input_part.shape
    TempNumpy = np.zeros((batchs, partsumpoint, others), dtype=np.float32)

    for i in range(batchs):
        Temp_torch= torch.squeeze(input_part[i].cpu())
        Temp_torch = Temp_torch.detach().numpy()

        n = np.random.choice(len(Temp_torch), partsumpoint, replace=False)  # s随机采500个数据，这种随机方式也可以自己定义
        TempNumpy[i] = Temp_torch[n]

    return torch.as_tensor(TempNumpy)  # NUMPY   to  tensor


def addnoise(real_point,opt):
    temp_real_point = RandomSample(real_point, int(opt.npoint/4))

    concat = []
    for i in range(len(temp_real_point)):

        newone = temp_real_point[i].squeeze()
        Array_data_np = addPointnumber(len(newone), opt.npoint, newone) #         #上采样后 再下采样， 增加复杂性
        onedata = torch.from_numpy(Array_data_np).float()
        np_3D=onedata.unsqueeze(0)
        concat.append(np_3D)
    real_point = torch.cat(concat, 0)

    if (real_point.shape[1] >= opt.npoint):
        real_point = RandomSample(real_point, opt.npoint)  # 输入需要三个维度
    return real_point


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)  #求中心，对pc数组的每行求平均值，通过这条函数最后得到一个1×3的数组[x_mean,y_mean,z_mean];
    pc = pc - centroid  #点云平移  或  # 求得每一点到中点的绝对距离
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))  # 将同一列的元素取平方相加得(x^2+y^2+z^2)，再开方，取最大，得最大标准差
    #pc ** 2 平移后的点云求平方   #np.sum(pc ** 2, axis=1)：每列求和
    pc = pc / m   # 归一化，这里使用的是Z-score标准化方法，即为(x-mean)/std
    return pc


if __name__ == '__main__':


    args = arguments.parse_args()
    trainDataLoader, testDataLoader = train_partseg.load_data(args)


    #point++
    device,experiment_dir,checkpoints_dir,seg_classes,seg_label_to_cat = train_partseg.onint(args)
    PointNetModel, criterion_point, optimizer, start_epoch = train_partseg.OninitNetwork(args,device,experiment_dir)


    # FPNet
    PFnetG,PFnetD,\
    criterion,cri_PointLoss,optimizerD,optimizerG,schedulerD,schedulerG = file_PFNet.InitPFnet(args,device)


    #treegan
    G_treegan,D_treegan,optimizerG_treegan,optimizerD_treegan,GP,knn_loss,w_train_ls = file_treegan.InitTreeGan(args,device)


###################    进入 for  batch 循环  ####################
################################################################
    # poinitnet for 之前 必须存在
    best_acc = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0
    train_len =0  #len(train_loader) # 训练中的测试使用  暂时0
    pcd_vector = o3d.geometry.PointCloud()  # 建立一个空的open3d点云 # 防止重复建立
    lastPointAcu = 0


   ##############  tree gan  #################
    loss_log = {'G_loss': [], 'D_loss': []}
    loss_legend = list(loss_log.keys())
    metric = {'FPD': []}



    for epoch in range(start_epoch, args.epoch):

        PointNetModel = train_partseg.PointNet_adjustepoch(args, epoch, PointNetModel, optimizer)
        mean_correct = []

        # FPNet
        alpha1, alpha2 = PFotherdef.PF_epoch_alpha(epoch)
        Aloss = []

        ##############  tree gan  #################
        epoch_g_loss = []
        epoch_d_loss = []
        #epoch_time = time.time()
        w_train = w_train_ls[min(3, int(epoch / 500))]


        '''learning one epoch'''
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):

            if lastPointAcu <= 0.9999:  #
            #if epoch > 10000:  #
                points, label, target = train_partseg.OneBatchData(device,data, pcd_vector)
                # points, label, target = data
                recorrect, OutbatchLabel = train_partseg.OneBatchTrain(args, optimizer, PointNetModel, points, label,
                                                                       target, criterion_point)
                mean_correct.append(recorrect.item() / (args.batch_size * args.npoint))


            else:

                ################################################################################
                points, label, target = train_partseg.OneBatchData(device,data,pcd_vector)

                recorrect, OutbatchLabel = train_partseg.OneBatchTrain(args,optimizer,PointNetModel, points, label, target,criterion_point)
                mean_correct.append(recorrect.item() / (args.batch_size * args.npoint))
                batchtree_abstractleaf,concat = SplitLeaf(args, epoch, points, OutbatchLabel) # 如果 想要的label数，少于 xx 则不进行分类

                if concat >= 2:
                    print("epoch 该株棉花数量为", epoch, '_', batchtree_abstractleaf.shape[0])

                    templist = []
                    for i in range(batchtree_abstractleaf.shape[0]):
                        temptorch = batchtree_abstractleaf[i,:, :]
                        temptorch = temptorch.data.numpy()
                        temptorch = pc_normalize(temptorch)
                        temptorch = temptorch.reshape(-1,1024,3)
                        templist.append(temptorch)
                    batchtree_abstractleaf = np.concatenate(templist,0)
                    batchtree_abstractleaf = torch.tensor(batchtree_abstractleaf).float()
                    ################################################################################
                    # # print(batchtree_abstractleaf.shape)   # FP net 中不能使用batch = 1
                    # points, label, target = data
                    # batchtree_abstractleaf = points
                    ################################################################################


                    if batchtree_abstractleaf.shape[0]<=10 :
                        inputFp = batchtree_abstractleaf
                    else:
                        inputFp = batchtree_abstractleaf[0:8, :, :]
                        inputFp2 = batchtree_abstractleaf[8:, :, :]

                    point_gt, input_part1, point_drop, output_drop3, PF_CD, \
                        epoch_g_loss = file_PFNet.RunOnebatch(args, device, train_len,
                                                              inputFp, inputFp, "points_ref",
                                                              epoch, PFnetG, PFnetD, i, criterion, cri_PointLoss,
                                                              optimizerD, optimizerG, alpha1, alpha2,
                                                              G_treegan,D_treegan, optimizerG_treegan,optimizerD_treegan,
                                                              GP, knn_loss,w_train,
                                                              epoch_d_loss,epoch_g_loss)

                    if batchtree_abstractleaf.shape[0] >=11:
                        point_gt, input_part1, point_drop, output_drop3, PF_CD, \
                            epoch_g_loss = file_PFNet.RunOnebatch(args, device, train_len,
                                                                  inputFp2, inputFp2, "points_ref",
                                                                  epoch, PFnetG, PFnetD, i, criterion, cri_PointLoss,
                                                                  optimizerD, optimizerG, alpha1, alpha2,
                                                                  G_treegan,D_treegan, optimizerG_treegan,optimizerD_treegan,
                                                                  GP, knn_loss,w_train,
                                                                  epoch_d_loss,epoch_g_loss)


                    Aloss.append(float(PF_CD))  # 临时

                    ###############################################################################
                    # FPdrop_gt,FPdrop_fake,fake_point,epoch_d_loss,epoch_g_loss =file_treegan.RunOnebatch(args,G_treegan,D_treegan,
                    #                                                                                     optimizerG_treegan,optimizerD_treegan,GP,knn_loss,w_train,point_drop,output_drop3,epoch_d_loss,epoch_g_loss)

                    if epoch % 25 == 0:
                        saveFPpoint(epoch, 'FPpoint_gt', point_gt)
                        saveFPpoint(epoch, 'FPinput_part1', input_part1)
                        saveFPpoint(epoch, 'FPpoint_drop', point_drop)
                        saveFPpoint(epoch, 'FPoutput_drop3', output_drop3)

        ################################################################################
        ########  point++  测试 并保存  #############
        ###################################################
        train_instance_acc = np.mean(mean_correct)
        WriteTxtLine(args.TxtLinePoint, str(epoch) + " " + str(train_instance_acc))
        print('Train accuracy is: %.3f' % train_instance_acc)
        print("epoch ", epoch)
        lastPointAcu = train_instance_acc

        test_metrics = train_partseg.OneBatchVal(args,seg_classes,testDataLoader,PointNetModel)

        train_partseg.SavePthLog(checkpoints_dir, epoch, optimizer, PointNetModel,
                                 train_instance_acc, test_metrics, best_acc,best_class_avg_iou, best_inctance_avg_iou)
        ###################################################

        ################################################################################
        ######## FPnet  测试 并保存  #############
        ###################################################

        file_PFNet.OneBatchVal()
        file_PFNet.SavePthLog(epoch,PFnetG)


        schedulerD.step()
        schedulerG.step()


        ################################################################################
        ########    tree gan   测试 并保存   ###############################
        ################################################################
        # # # ---------------- Epoch everage loss   --------------- #
        # d_loss_mean = np.array(epoch_d_loss).mean()
        # g_loss_mean = np.array(epoch_g_loss).mean()
        #print("g_loss_mean",g_loss_mean)

        ################################################################################
        ######## 自己保存的数据  #############
        ###################################################

        if len(Aloss) != 0:
            avgepoch_Aloss = sum(Aloss) / len(Aloss)  # 所有的损失 / 摊平到每一个上
            avgepoch_Aloss = float('%.5f' % avgepoch_Aloss)
            WriteTxtLine(args.TxtLineFP, str(epoch) + " " + str(avgepoch_Aloss))
            print("epoch FPloss1",epoch, avgepoch_Aloss)









