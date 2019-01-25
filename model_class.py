import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter
from dataset import TSNDataSet
from models import TSN
from transforms import *
import sys
import torch.utils.model_zoo as model_zoo
from torch.nn.init import constant_, xavier_uniform_
from ops.utils import class_recall
import argparse
parser=argparse.ArgumentParser(description="test")
parser.add_argument('val_list',type=str)
parser.add_argument('model_path',type=str)
parser.add_argument('num_segments',type=int)
parser.add_argument('pretrained_parts',type=str)
best_prec1 = 0
args=parser.parse_args()

def main():

    global args, best_prec1
    num_class=4
    rgb_read_format="{:d}.jpg"

    model = TSN(num_class, args.num_segments, args.pretrained_parts, 'RGB',
                base_model='ECO',
                consensus_type='identity', dropout=0.3, partial_bn=True)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std

    # Optimizer s also support specifying per-parameter options.
    # To do this, pass in an iterable of dict s.
    # Each of them will define a separate parameter group,
    # and should contain a params key, containing a list of parameters belonging to it.
    # Other keys should match the keyword arguments accepted by the optimizers,
    # and will be used as optimization options for this group.
    policies = model.get_optim_policies()

    train_augmentation = model.get_augmentation()

    model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()

    model_dict = model.state_dict()

    print("pretrained_parts: ", args.pretrained_parts)

    model_dir=args.model_path
    new_state_dict=torch.load(model_dir)['state_dict']

    un_init_dict_keys = [k for k in model_dict.keys() if k not in new_state_dict]
    print("un_init_dict_keys: ", un_init_dict_keys)
    print("\n------------------------------------")

    for k in un_init_dict_keys:
        new_state_dict[k] = torch.DoubleTensor(model_dict[k].size()).zero_()
        if 'weight' in k:
            if 'bn' in k:
                print("{} init as: 1".format(k))
                constant_(new_state_dict[k], 1)
            else:
                print("{} init as: xavier".format(k))
                xavier_uniform_(new_state_dict[k])
        elif 'bias' in k:
            print("{} init as: 0".format(k))
            constant_(new_state_dict[k], 0)

    print("------------------------------------")

    model.load_state_dict(new_state_dict)


    cudnn.benchmark = True

    # Data loading code
    normalize = GroupNormalize(input_mean, input_std)
    data_length = 1


    val_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality='RGB',
                   image_tmpl=rgb_read_format,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=True),
                       ToTorchFormatTensor(div=False),
                       normalize,
                   ])),
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)


    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    model.eval()
    for i,(input,target) in enumerate(val_loader):
        target=target.cuda()
        input_var=input
        target_var=target
        output=model(input_var)
        _,pred=output.data.topk(1,1,True,True)
        print(pred,target)
    print('done')









if __name__ == '__main__':
    main()
