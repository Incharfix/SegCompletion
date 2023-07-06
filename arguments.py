
import argparse


def parse_args():
    parser = argparse.ArgumentParser('Model')

    parser.add_argument('--gpu_environ', default='0', type=str, metavar='DEVICE', help='GPU to use, ')
    parser.add_argument('--gpu_ids', default=[0], type=list, metavar='DEVICE', help='GPU to use, ')
    parser.add_argument('--gpu_env', type=str, default='0', help='default= str(0,1,2) ')
    parser.add_argument('--gpu_list', type=list, default=[0], help='default=[0,1,2] ')


    parser.add_argument('--root', type=str, default='../data_ours/TrainPoint/', help='Adam or SGD [default: Adam]')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=301, type=int, help='Epoch to run [default: 251]')


    # Point++
    parser.add_argument('--npoint', type=int,  default=20480, help='Point Number [default: 2048]')
    # FPnet
    parser.add_argument('--num_points', default=1024, type=int,metavar='N', help='points in point-cloud (default: 1024)')
    parser.add_argument('--FPnetNeed_point', type=int,  default=1024, help=' Point Number [default: 2048]')

    parser.add_argument('--point_scales_list', type=list, default=[1024,512,256],
                        help='number of points in each scales')  # default [2048,1024,512]
    parser.add_argument('--point_drop_num', type=int, default=256,
                        help='0 means do not use else use with this weight')  # default [512]



    parser.add_argument('--model', type=str, default='pointnet2_part_seg_msg', help='model name [default: pointnet2_part_seg_msg]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')

    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')

    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--step_size', type=int,  default=20, help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.5, help='Decay rate for lr decay [default: 0.5]')
    parser.add_argument('--log_dir', type=str, default='pointnet2_part_seg_msg', help='Experiment root')

    parser.add_argument('--num_classes', type=int,  default=16, help='')
    parser.add_argument('--num_part', type=int,  default=50, help='') # s 所有的物体被分为多少类别


    parser.add_argument('--learning_rate_clip', type=int,  default=1e-5, help='LEARNING_RATE_CLIP')
    parser.add_argument('--momentum_original', type=int,  default=0.1, help='MOMENTUM_ORIGINAL')
    parser.add_argument('--momentum_deccay', type=int,  default=0.5, help='MOMENTUM_DECCAY')

    # PFnet
    parser.add_argument('--TxtLinePoint', default='./output_testpoint/TxtLinePoint.txt', type=str)
    parser.add_argument('--TxtLineFP', default='./output_testpoint/TxtLineFP.txt', type=str)

    parser.add_argument('--num_scales', type=int, default=3, help='number of scales')
    parser.add_argument('--each_scales_size', type=int, default=1, help='each scales size')
    parser.add_argument('--wtl2', type=float, default=0.95, help='0 means do not use else use with this weight')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--D_choose', type=int, default=1, help='0 not use D-net,1 use D-net')
    parser.add_argument('--cropmethod', default='random_center', help='random|center|random_center')



    ##############################################
    ###  tree gan
    ##############################################

    ### general training related

    parser.add_argument('--lr', type=float, default=1e-4, help='Float value for learning rate.')
    parser.add_argument('--lambdaGP', type=int, default=10, help='Lambda for GP term.')

    parser.add_argument('--D_iter', type=int, default=5, help='Number of iterations for discriminator.')
    parser.add_argument('--w_train_ls', type=float, default=[1], nargs='+', help='train loss weightage')

    ### uniform losses related
    # PatchVariance
    parser.add_argument('--knn_loss', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--knn_k', type=int, default=30)
    parser.add_argument('--knn_n_seeds', type=int, default=100)
    parser.add_argument('--knn_scalar', type=float, default=0.2)
    # PU-Net's uniform loss
    parser.add_argument('--krepul_loss', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--krepul_k', type=int, default=10)
    parser.add_argument('--krepul_n_seeds', type=int, default=20)
    parser.add_argument('--krepul_scalar', type=float, default=1)
    parser.add_argument('--krepul_h', type=float, default=0.01)
    # MSN's Expansion-Penalty
    parser.add_argument('--expansion_penality', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--expan_primitive_size', type=int, default=2)
    parser.add_argument('--expan_alpha', type=float, default=1.5)
    parser.add_argument('--expan_scalar', type=float, default=0.1)


    ### TreeGAN architecture related
    parser.add_argument('--DEGREE', type=int, default=[1, 2, 2, 2, 2, 2, 64], nargs='+',
                              help='Upsample degrees for generator.')
    parser.add_argument('--G_FEAT', type=int, default=[96, 256, 256, 256, 128, 128, 128, 3], nargs='+',
                              help='Features for generator.')
    parser.add_argument('--D_FEAT', type=int, default=[3, 64, 128, 256, 256, 512], nargs='+',
                              help='Features for discriminator.')
    parser.add_argument('--support', type=int, default=10, help='Support value for TreeGCN loop term.')
    parser.add_argument('--loop_non_linear', default=False, type=lambda x: (str(x).lower() == 'true'))



    ### ohters
    parser.add_argument('--ckpt_path0', type=str, default='./checkpoints', help='Checkpoint path.')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/chair', help='Checkpoint path.')
    parser.add_argument('--ckpt_save', type=str, default='tree_ckpt_', help='Checkpoint name to save.')
    parser.add_argument('--eval_every_n_epoch', type=int, default=25, help='0 means never eval')
    parser.add_argument('--save_every_n_epoch', type=int, default=100, help='save models every n epochs')

    return parser.parse_args()