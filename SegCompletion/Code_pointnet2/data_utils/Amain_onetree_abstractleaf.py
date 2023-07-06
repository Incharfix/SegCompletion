import numpy as np
import open3d as o3d
from  Code_pointnet2.data_utils.A3_FPS_Npoint2Npoint import RandomSample as RandomSample
from  Code_pointnet2.data_utils.A3_FPS_Npoint2Npoint import sample_and_group as FPS
import torch
import os
import hdbscan



def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def noise_Gaussian(points, std):
    noise = np.random.normal(0, std, points.shape)
    out = points + noise
    return out


def addPointnumber(Nowpoint,needpoint,data):

    count_times = int(needpoint / Nowpoint)
    target_np_noised = data

    for  i  in range(count_times +1 ):

        temp = noise_Gaussian(data, 0.00000001)
        target_np_noised = np.vstack([temp, target_np_noised])

    #print('add之后 ',target_np_noised.shape)
    return  target_np_noised

def OneTree_GetAllLeaf(data):
    pos=np.where(data[:,3] >=0.8  )
    All_leaf = data[pos]

    pos=np.where(All_leaf[:,3] <= 1.2  )
    All_leaf = All_leaf[pos]


    pos=np.where(data[:,3] <= 0.5  )
    all_bar = data[pos]

    return All_leaf, all_bar

#  DBSCAN
def OneTree_GetLeafLabel(All_leaf):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(All_leaf[:,:3])
    label_npList = np.array(pcd.cluster_dbscan(eps=0.07, min_points=600, print_progress=False))
    label_np2D= label_npList.reshape(-1,1) # 每个点都被编辑成1个类别

    data_pointlabel = np.concatenate([All_leaf[:,:3], label_np2D], 1)

    return label_npList,label_np2D,data_pointlabel


# HDBSCAN
def getInstancelabel_HDBSCAN(data):

    cluster_labels = hdbscan.HDBSCAN(min_cluster_size=400).fit_predict(data)# points[:, :3]

    max_label = np.max(cluster_labels)
    # print('获得的最大类别',max(cluster_labels))
    # print('type(labels)',type(cluster_labels))
    Newlabels= cluster_labels.reshape(-1,1)
    resultdata = np.concatenate([data, Newlabels], 1)

    return resultdata,max_label

def getleaf(index,data):

    if index ==0 :
        pos = np.where(data[:, 3] >= 0)
        tempdata = data[pos]
        temppos = np.where(tempdata[:, 3] <= 0.3)
    elif index >= 1:
        pos = np.where(data[:, 3] >= index-0.2)
        tempdata = data[pos]
        temppos = np.where(tempdata[:, 3] <= index+0.2)


    return tempdata,temppos

def getmedian(x):
    length = len(x)
    x.sort()
    print(length,x)
    if (length % 2)== 1:
        z=length // 2
        y = x[z]
    else:
        y = (x[length//2]+x[length//2-1])/2
    return y


def presigleleaf(templistnum,templistnpy):

    templistnpy_save = []
    if len(templistnum) >= 3:
        m_median = getmedian(templistnum)
        print(m_median,templistnum)
        for j, m_data in enumerate(templistnpy):
            if m_data.shape[0] >= m_median * 0.4 and m_data.shape[0] <= m_median * 1.9:
                templistnpy_save.append(m_data)
            else:
                print('drop：', m_data.shape[0])
    else:
        templistnpy_save = templistnpy

    return templistnpy_save


def handle_nor_FPS(sigleleaf_All,needpoint):

    concat = []
    for index,oneleaf  in  enumerate(sigleleaf_All):
        if (len(oneleaf) < needpoint):
            oneleaf = addPointnumber(len(oneleaf), needpoint, oneleaf)  #

        oneleaf = torch.from_numpy(oneleaf).float()
        torch_3D = torch.unsqueeze(oneleaf, dim=0)

        if (torch_3D.shape[1] >= needpoint):
            torch_3D = FPS(torch_3D[:, :, :3], needpoint)
            #torch_3D = RandomSample(torch_3D[:,:,:3], needpoint)
            concat.append(torch_3D)
            np.savetxt('./output_test/'  + str(index)  + 'leafone.txt', torch_3D.squeeze(0), fmt='%1.5f')

    OnebatchLeaf = torch.cat(concat, 0)  # [batch,2048,1]

    return  OnebatchLeaf



# resultdata,max_label

# 需要的部件点数
def OneTree_SplitLeaf(resultdata, max_label,needpoint):

    if max_label <= 0:
        print("Number of clusters")
        return 1,1

    concat = []
    print("Number of clusters：",max_label)

    templistnum = []
    templistnpy = []

    # 1颗棉花植株
    for i in range(max_label+1):

        tempdata, temppos = getleaf(i, resultdata)
        point_first = tempdata[temppos]
        templistnum.append(point_first.shape[0])
        templistnpy.append(point_first[:,:3])

    sigleleaf=presigleleaf(templistnum,templistnpy)

    OnebatchLeaf = handle_nor_FPS(sigleleaf,needpoint)

    return OnebatchLeaf,OnebatchLeaf.shape[0]




########################
# Data and label fusion
# Then cut the data
########################

def SplitLeaf(args,epoch,inputbatchdata,OutbatchLabel): # [batch,2048,3]  [batch,2048,6]

    batchs,_,_= OutbatchLabel.shape
    #label 具体预测出来

    concat = []
    for i in range(batchs):
        seg_pred = OutbatchLabel[i].contiguous().view(-1, args.num_part)
        pred_choice = seg_pred.data.max(1)[1]  # max(1)：每1行的最大值
        tempdata = torch.reshape(pred_choice, ((1,-1, 1)))
        concat.append(tempdata)
    OutbatchLabel = torch.cat(concat, 0)  #[batch,2048,1]

    inputbatchdata = inputbatchdata.permute(0, 2, 1)
    inputbatchdata = inputbatchdata[:,:,:3]
    DataCatLabel= torch.cat([inputbatchdata.float(),OutbatchLabel.float()],dim=2)

    if epoch %100 == 0:
        savebatchpoint(str(epoch) +"_1Onetree_", DataCatLabel)

    ###################################
    # Plants of 1 family were sampled as leaves respectively
    ##################################
    # concat = []
    for i in range(batchs):
        onetree = torch.squeeze(DataCatLabel[i])          #[1,x,y]   batch=1，
        onetree1 = onetree.cpu().numpy()                  # .cpu().numpy()

        All_leaf, all_bar   = OneTree_GetAllLeaf(onetree1)

        # label_npList,label_np2D,data_PoLab = OneTree_GetLeafLabel(All_leaf) #  DBSCAN
        resultdata, max_label = getInstancelabel_HDBSCAN(All_leaf[:, :3])     # HDBSCAN  #
        print(max_label)

        abstractleaf, leafNum = OneTree_SplitLeaf(resultdata, max_label,  args.FPnetNeed_point)  # needpoint

    return abstractleaf,leafNum


def SplitLeafTest(args,epoch,inputbatchdata,OutbatchLabel): # [batch,2048,3]  [batch,2048,6]

    batchs = OutbatchLabel.shape[0]

    OutbatchLabel = np.expand_dims(OutbatchLabel,axis=2)
    OutbatchLabel = torch.from_numpy(OutbatchLabel)
    DataCatLabel = torch.cat([inputbatchdata, OutbatchLabel], dim=2)

    if epoch % 100 == 0:
        savebatchpoint(str(epoch) + "_1Onetree_", DataCatLabel)

    ###################################
    # Plants of 1 family were sampled as leaves respectively
    ##################################
    # concat = []
    for i in range(batchs):
        onetree = torch.squeeze(DataCatLabel[i])  # [1,x,y] 这里的batch=1，
        onetree1 = onetree.cpu().numpy()  # .cpu().numpy()

        All_leaf, all_bar = OneTree_GetAllLeaf(onetree1)
        np.savetxt(  './temp_test.txt', All_leaf, fmt='%.4f')

        # label_npList,label_np2D,data_PoLab = OneTree_GetLeafLabel(All_leaf)  #  DBSCAN
        resultdata, max_label = getInstancelabel_HDBSCAN(All_leaf[:, :3])      # HDBSCAN  #

        print(max_label)
        # savebatchpointNocpu(str(epoch) + "HdnscanLeaf", resultdata)
        abstractleaf, leafNum = OneTree_SplitLeaf(resultdata, max_label, args.FPnetNeed_point)  # needpoint

        all_bar = np.expand_dims(all_bar, axis=0)
        all_bar = torch.from_numpy(all_bar)
    return abstractleaf, leafNum,all_bar



