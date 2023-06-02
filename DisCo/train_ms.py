import os
import glob
import argparse
import builtins
import math
import random
import shutil
import time
import warnings


import mindspore as ms
from mindspore import nn, Tensor, Model, ParallelMode
from mindspore.communication import init, get_rank, get_group_size

from dataset_ms import create_dataset
from ResNet_ms import resnet50, resnet18
from DisCo_ms import DisCo


parser = argparse.ArgumentParser(description='MindSpore ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('--teacher_arch', default='resnet50', type=str,
                    help='teacher architecture')
parser.add_argument('--teacher', default='', type=str, metavar='PATH',
                    help='path to teacher checkpoint (default: none)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                    help='mini-batch size (default: 32), this is the '
                         'batch size of each GPU on the current node')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='./ckpt', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: ./ckpt)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use MindSpore for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true', help='use mlp head')
parser.add_argument('--aug-plus', action='store_true', help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')


parser.add_argument('--nmb_prototypes', default=0, type=int, help='num prototype')
parser.add_argument('--only-mse', action='store_true', help='only use mse loss')


def get_last_checkpoint(checkpoint_dir):
    all_ckpt = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_moco_*.ckpt'))
    if all_ckpt:
        all_ckpt = sorted(all_ckpt)
        return all_ckpt[-1]
    else: return ''


def get_lr(steps_per_epoch, args):
    """ generate learning rate array """
    lr_each_step = []
    for epoch in range(args.epochs):
        lr = args.lr
        if args.cos:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        else:  # stepwise lr schedule
            for milestone in args.schedule:
                lr *= 0.1 if epoch >= milestone else 1.
        for step in range(steps_per_epoch):
            lr_each_step.append(lr)

    lr_each_step = Tensor(lr_each_step, ms.float32)
    return lr_each_step
 

class NetWithLossCell(nn.Cell):
    def __init__(self, network, loss,mse_loss, only_mse):
        super(NetWithLossCell, self).__init__()
        self.network = network
        self.loss = loss
        self.mse_loss = mse_loss
        self.only_mse = only_mse

    def construct(self, data_x, data_y, label):
        logits, labels, student_q, teacher_q, student_qk, teacher_qk = self.network(data_x, data_y)
        loss = self.loss(logits, labels)
        mse_loss = self.mse_loss(teacher_q, student_q)
        mse_loss_qk = self.mse_loss(teacher_qk, student_qk)
        if self.only_mse:
            loss = 0.0 * loss + 1.0 * mse_loss + 1.0 * mse_loss_qk
        else: loss = 1.0 * loss + 1.0 * mse_loss + 1.0 * mse_loss_qk
        return loss


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3"
    args = parser.parse_args()

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    if args.distributed: # GPU target
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")
        init("nccl")
        ms.set_auto_parallel_context(device_num=get_group_size(),
                                     parallel_mode=ParallelMode.DATA_PARALLEL,
                                     parameter_broadcast=True, gradients_mean=True)
    else: ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU", device_id=args.gpu)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    train_dataset = create_dataset(traindir, args.batch_size, args.aug_plus, args.distributed)
    steps_per_epoch = train_dataset.get_dataset_size()
    # print(steps_per_epoch)

    # create model
    print("=> creating model '{}'".format(args.teacher_arch))
    if args.teacher_arch == 'resnet50': teacher_model = resnet50
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'resnet18': base_model = resnet18
    model = DisCo(base_model, teacher_model, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, args)

    # # optionally resume from a checkpoint
    # if args.resume:
    #     checkpoint_path = get_last_checkpoint(args.resume)
    #     if os.path.isfile(checkpoint_path):
    #         print("=> loading checkpoint '{}'".format(checkpoint_path))
    #         param_dict = ms.load_checkpoint(checkpoint_path)
    #         ms.load_param_into_net(model, param_dict)
    #         args.start_epoch = int(checkpoint_path[-8:-5])
    #         print("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, args.start_epoch))
    #     else: print("=> no checkpoint found at '{}'".format(args.resume))
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss(reduction='sum')
    optimizer = nn.SGD(model.trainable_params(), learning_rate= get_lr(steps_per_epoch, args), 
                       momentum=args.momentum, weight_decay=args.weight_decay)

    net_loss = NetWithLossCell(model, criterion, criterion_mse, args.only_mse)
    train_net = nn.TrainOneStepCell(net_loss, optimizer)
    model = Model(train_net)

    config_ck = ms.CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=args.epochs)
    ckpoint_cb = ms.ModelCheckpoint(prefix="checkpoint_moco_%s"%(get_rank()), directory=args.resume, config=config_ck)
    model.train(args.epochs-args.start_epoch, train_dataset, 
                callbacks=[ms.TimeMonitor(steps_per_epoch), ckpoint_cb, ms.LossMonitor(args.print_freq)])#, dataset_sink_mode=True)