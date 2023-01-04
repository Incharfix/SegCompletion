"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
from Code_pointnet2.data_utils.ShapeNetDataLoader import PartNormalDataset
from arguments_test import  parse_args




# @ 20200809 调整 数据集 -  这里是能够正常运行的
"""
训练所需设置参数：

--log_dir pointnet2_part_seg_msg
--dataroot  ./dataset/shapenetcore_partanno/
--class_choice Airplane
--netG Pth_FPnet/point_netG150.pth

"""

def Get_ShapeIous(args,points,seg_classes,seg_label_to_cat,target,cur_pred_val_logits,total_correct,total_seen,total_seen_class,total_correct_class,shape_ious):



    cur_batch_size, num_point, _ = points.size()
    cur_pred_val = np.zeros((cur_batch_size, num_point)).astype(np.int32)



    for i in range(cur_batch_size):
        cat = seg_label_to_cat[target[i, 0]]
        logits = cur_pred_val_logits[i, :, :]
        cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

    correct = np.sum(cur_pred_val == target)
    total_correct += correct
    total_seen += (cur_batch_size * num_point)

    for l in range(args.num_part):
        total_seen_class[l] += np.sum(target == l)
        total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

    for i in range(cur_batch_size):
        segp = cur_pred_val[i, :]
        segl = target[i, :]
        cat = seg_label_to_cat[segl[0]]
        part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
        for l in seg_classes[cat]:
            if (np.sum(segl == l) == 0) and (
                    np.sum(segp == l) == 0):  # part is not present, no prediction as well
                part_ious[l - seg_classes[cat][0]] = 1.0
            else:
                part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                    np.sum((segl == l) | (segp == l)))
        shape_ious[cat].append(np.mean(part_ious))

    return cur_pred_val, target,shape_ious,shape_ious['Airplane']



def Get_PreTarget(args,points,label,target,PointNetModel):
    cur_batch_size, num_point, _ = points.size()
    points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
    #points, label, target = points.float(), label.long(), target.long()
    points = points.transpose(2, 1)

    PointNetModel = PointNetModel.eval()
    vote_pool = torch.zeros(target.size()[0], target.size()[1], args.num_part).cuda()
    #vote_pool = torch.zeros(target.size()[0], target.size()[1], args.num_part)
    for _ in range(args.num_votes):
        seg_pred, _ = PointNetModel(points, to_categorical(label, args.num_classes))
        vote_pool += seg_pred

    seg_pred = vote_pool / args.num_votes
    cur_pred_val_logits =seg_pred.cpu().data.numpy()
    target = target.cpu().data.numpy()

    return  target,cur_pred_val_logits

# 暂时 被拆解
def PointWithno_grad(args,seg_classes,seg_label_to_cat,testDataLoader,PointNetModel):

    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(args.num_part)]
        total_correct_class = [0 for _ in range(args.num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}


        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):

            target,cur_pred_val_logits = Get_PreTarget(args,points,label,target,PointNetModel)
            cur_pred_val, target,shape_ious = Get_ShapeIous(args, points, seg_classes, seg_label_to_cat, target,
                                       cur_pred_val_logits,total_correct, total_seen, total_seen_class, total_correct_class, shape_ious)


        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        print("最重要的指标 Airplane ：", str(shape_ious['Airplane']) )


        # mean_shape_ious = np.mean(list(shape_ious.values()))
        # test_metrics['accuracy'] = total_correct / float(total_seen)
        # test_metrics['class_avg_accuracy'] = np.mean(
        #     np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        #
        # for cat in sorted(shape_ious.keys()):
        #     print('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        #
        # test_metrics['class_avg_iou'] = mean_shape_ious
        # test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

    # print('Accuracy is: %.5f'%test_metrics['accuracy'])
    # print('Class avg accuracy is: %.5f'%test_metrics['class_avg_accuracy'])
    # print('Class avg mIOU is: %.5f'%test_metrics['class_avg_iou'])
    # print('Inctance avg mIOU is: %.5f'%test_metrics['inctance_avg_iou'])


#############################################################
################################################################
##################################################################

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def onint(args):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = BASE_DIR
    sys.path.append(os.path.join(ROOT_DIR, 'models'))

    seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                   'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                   'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                   'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15],
                   'Knife': [22, 23]}
    # seg_classes = {'Airplane': [0, 1, 2, 3]}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat


    '''HYPER PARAMETER'''
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'pth_pointdir/part_seg/' + args.log_dir



    return experiment_dir,seg_classes,seg_label_to_cat

def load_data(args):

    TEST_DATASET = PartNormalDataset(root = args.root, npoints=args.npoint, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,shuffle=False, num_workers=4)
    print("The number of test data is: %d" %  len(TEST_DATASET))

    return testDataLoader

def MoudleLoad(args,experiment_dir):



    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(args.num_part, normal_channel=args.normal).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    '''MODEL LOADING'''
    # model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    # MODEL = importlib.import_module(model_name)
    # GPU
    #     PointNetModel = MODEL.ModelNet(args.num_part, normal_channel=args.normal)#.cuda()
    # checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth',map_location='cpu')
    # PointNetModel.load_state_dict(checkpoint['model_state_dict'],False)

    return classifier



def main():

    args = parse_args()
    experiment_dir,seg_classes,seg_label_to_cat = onint(args)
    testDataLoader = load_data(args)

    PointNetModel = MoudleLoad(args,experiment_dir)
    PointWithno_grad(args,seg_classes,seg_label_to_cat,testDataLoader,PointNetModel)




if __name__ == '__main__':
    main()

