
import argparse


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing [default: 24]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--root', type=str, default='../data_ours/TrainPoint/', help='Adam or SGD [default: Adam]')
    #  TrainPoint  DrawPoint
    parser.add_argument('--netG', default='./Pth_FPnet/point_netG200.pth',help="path to netG (to continue training)")

    # Point++
    parser.add_argument('--npoint', type=int,  default=20480, help='Point Number [default: 2048]')
    # FPnet
    parser.add_argument('--num_points', default=1024, type=int,metavar='N', help='points in point-cloud (default: 1024)')
    parser.add_argument('--FPnetNeed_point', type=int,  default=1024, help='每片你叶子需要的点数  Point Number [default: 2048]')

    parser.add_argument('--point_scales_list', type=list, default=[1024,512,256],
                        help='number of points in each scales')
    parser.add_argument('--crop_point_num', type=int, default=256, help='0 means do not use else use with this weight')

    parser.add_argument('--log_dir', type=str, default='pointnet2_part_seg_msg', help='Experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate segmentation scores with voting [default: 3]')


   # point++
    parser.add_argument('--num_classes', type=int,  default=16, help='')
    parser.add_argument('--num_part', type=int,  default=50, help='')


   # PFnet
    # parser.add_argument('--dataset',  default='ModelNet40', help='ModelNet10|ModelNet40|ShapeNet')
    parser.add_argument('--dataroot', default='dataset/train', help='path to dataset')
    parser.add_argument('--class_choice', default='Airplane',
                        help='choice something to learn for Gan')
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")

    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--pnum', type=int, default=1024, help='the point number of a sample')

    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--cuda', type=bool, default=False, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--drop', type=float, default=0.2)
    parser.add_argument('--num_scales', type=int, default=3, help='number of scales')

    parser.add_argument('--each_scales_size', type=int, default=1, help='each scales size')
    parser.add_argument('--wtl2', type=float, default=0.9, help='0 means do not use else use with this weight')
    parser.add_argument('--cropmethod', default='random_center', help='random|center|random_center')
    parser.parse_args()
    return parser.parse_args()