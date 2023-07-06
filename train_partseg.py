import argparse
import os
from Code_pointnet2.data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
from Code_pointnet2.provider  import random_scale_point_cloud ,shift_point_cloud
import numpy as np
import open3d as o3d


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def load_data(args):

    TRAIN_DATASET = PartNormalDataset(root = args.root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size,shuffle=True, num_workers=4)
    TEST_DATASET = PartNormalDataset(root = args.root, npoints=args.npoint, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,shuffle=False, num_workers=4)

    return trainDataLoader,testDataLoader


def onint(args):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = BASE_DIR
    sys.path.append(os.path.join(ROOT_DIR, 'models'))

    seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                   'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                   'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
                   'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}


    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    if args.gpu_ids[0] >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_environ
        #_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

        strTemp = 'cuda:' + str(args.gpu_ids[0])
        device = torch.device(strTemp) if torch.cuda.is_available() else torch.device('cpu')

    else:
        device = torch.device('cpu')


    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./pth_pointdir/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('part_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('PARAMETER ...')
    logger.info(args)

    return device,experiment_dir,checkpoints_dir,seg_classes,seg_label_to_cat


def OninitNetwork(args,device,experiment_dir):

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(args.num_part, normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()
    criterion.to(device)

    start_epoch = 0
    classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),eps=1e-08, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    return  classifier,criterion,optimizer,start_epoch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum

def PointNet_adjustepoch(args,epoch,PointNetModel,optimizer):
    lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)),
             args.learning_rate_clip)  # '''Adjust learning rate and BN momentum'''

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    momentum = args.momentum_original * (args.momentum_deccay ** (epoch // args.step_size))
    if momentum < 0.01:
        momentum = 0.01

    PointNetModel = PointNetModel.apply(lambda x: bn_momentum_adjust(x, momentum))

    return PointNetModel


def OneBatchData(device,data,pcd_vector):
    points, label, target = data


    # 增加 噪点等
    points = points.data.numpy()
    points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
    points[:, :, 0:3] = shift_point_cloud(points[:, :, 0:3])

    points = torch.tensor(points).float()
    label = torch.tensor(label).long()
    target = torch.tensor(target).long()

    points = points.transpose(2, 1)

    points = points.to(device)
    label = label.to(device)
    target = target.to(device)

    return points, label, target




def SavePthLog(checkpoints_dir,epoch,optimizer,PointNetModel,train_instance_acc,test_metrics,best_acc,best_class_avg_iou,best_inctance_avg_iou):

    if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
        savepath = str(checkpoints_dir) + '/best_model.pth'
        state = {
            'epoch': epoch,
            'train_acc': train_instance_acc,
            'test_acc': test_metrics['accuracy'],
            'class_avg_iou': test_metrics['class_avg_iou'],
            'inctance_avg_iou': test_metrics['inctance_avg_iou'],
            'model_state_dict': PointNetModel.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)
        # log_string('Saving model....')

    if test_metrics['accuracy'] > best_acc:
        best_acc = test_metrics['accuracy']
    if test_metrics['class_avg_iou'] > best_class_avg_iou:
        best_class_avg_iou = test_metrics['class_avg_iou']
    if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
        best_inctance_avg_iou = test_metrics['inctance_avg_iou']

    return best_acc,best_class_avg_iou,best_inctance_avg_iou


def OneBatchVal(args,seg_classes,testDataLoader,PointNetModel):
    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(args.num_part)]
        total_correct_class = [0 for _ in range(args.num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                      smoothing=0.9):
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            PointNetModel = PointNetModel.eval()
            seg_pred, _ = PointNetModel(points, to_categorical(label, args.num_classes))
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()
            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

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

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class))

        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

    return  test_metrics


def OneBatchTrain(args,optimizer,PointNetModel, points, label, target,criterion ):

    optimizer.zero_grad()
    PointNetModel = PointNetModel.train()

    seg_pred, trans_feat = PointNetModel(points, to_categorical(label, args.num_classes))
    OutbatchLabel= seg_pred

    seg_pred = seg_pred.contiguous().view(-1, args.num_part)
    target = target.view(-1, 1)[:, 0]
    pred_choice = seg_pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).sum()
    recorrect = correct.detach()

    loss = criterion(seg_pred, target, trans_feat)
    loss.backward()
    optimizer.step()

    return recorrect,OutbatchLabel


def main():


    args = parse_args()
    logger,experiment_dir,checkpoints_dir,seg_classes,seg_label_to_cat = onint(args)
    trainDataLoader, testDataLoader = load_data(args)
    PointNetModel, criterion, optimizer, start_epoch = OninitNetwork(args, logger, experiment_dir)


    pcd_vector = o3d.geometry.PointCloud()


    for epoch in range(start_epoch,args.epoch):

        PointNetModel = PointNet_adjustepoch(args,epoch,PointNetModel,optimizer)
        mean_correct = []

        '''learning one epoch'''
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):

            points, label, target = OneBatchData(data,pcd_vector)
            recorrect, OutbatchLabel = OneBatchTrain(args, optimizer, PointNetModel, points, label,
                                                                   target, criterion)

            mean_correct.append(recorrect.item() / (args.batch_size * args.npoint))
            batchtree_abstractleaf = SplitLeaf(args,epoch,points,OutbatchLabel)
        train_instance_acc = np.mean(mean_correct)
        print('Train accuracy is: %.5f' % train_instance_acc)

if __name__ == '__main__':

    main()

