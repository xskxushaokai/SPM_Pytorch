import os
import argparse

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from dataset.SpmDataset import SpmDataset
from nets import network
from config.center_config import center_config

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    min_val_loss = 9999

    model = network.CPN101(center_config['out_shape'], center_config['joints']*2+1, pretrained = True)
    model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        if args.reset:
            args.start_epoch = 0
        else:
            args.start_epoch = checkpoint['epoch']

    criterion_global_root = torch.nn.MSELoss().cuda()
    criterion_global_joint = torch.nn.SmoothL1Loss().cuda()
    criterion_refine_root = torch.nn.MSELoss().cuda()
    criterion_refine_joint = torch.nn.SmoothL1Loss(reduce=False).cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=center_config['lr'],
                                 weight_decay=center_config['weight_decay'])

    cudnn.benchmark = True
    print('    Total params: %.2fMB' % (sum(p.numel() for p in model.parameters()) / (1024 * 1024) * 4))

    train_loader = torch.utils.data.DataLoader(
        SpmDataset(center_config, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        SpmDataset(center_config, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(center_config['lr'], optimizer, epoch)
        train_loss = train(train_loader, model, (criterion_global_root, criterion_global_joint, criterion_refine_root, criterion_refine_joint), optimizer)
        print('epoch: {} train_loss: {}'.format(epoch, train_loss))

        val_loss = validation(val_loader, model, (criterion_refine_root, criterion_global_joint))
        print('epoch: {} validation loss:  {}'.format(epoch, val_loss))

        if val_loss < min_val_loss:
            min_val_loss = val_loss

            save_model({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'min_val_loss': min_val_loss
            }, checkpoint=args.checkpoint, filename="SPM_resnet101_576_320_person_20191130")


def train(train_loader, model, criterions, optimizer):
    def ohkm(loss, top_k):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(sub_loss, k=top_k, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / top_k
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    criterion_global_root, criterion_global_joint, criterion_refine_root, criterion_refine_joint = criterions

    losses = AverageMeter()

    model.train()

    for i, (inputs, root_map, kps_map, kps_weight) in enumerate(train_loader):
        input_var = torch.autograd.Variable(inputs.cuda())
        root_map_var = torch.autograd.Variable(root_map.cuda(async=True))
        kps_map_var = torch.autograd.Variable(kps_map.cuda(async =True))
        kps_weight_var = torch.autograd.Variable(kps_weight.cuda(async =True))

        global_outputs, refine_output = model(input_var)

        loss = 0.
        global_loss_record_root = 0.
        global_loss_record_joint = 0.
        refine_loss_record_root = 0.
        refine_loss_record_joint = 0.

        for global_output in global_outputs:
            root_map_pred = torch.unsqueeze(global_output[:, 0, :, :], dim=1)    # 0 channel位置
            kps_map_pred = global_output[:, 1:, :, :]    # 1:end channel位置 x1,y1,x2,y2,......
            global_root_loss = criterion_global_root(root_map_pred, root_map_var)
            loss += global_root_loss * 1
            global_loss_record_root += global_root_loss.data.item()
            global_joint_loss = criterion_global_joint(kps_map_pred.mul(kps_weight_var), kps_map_var)    # 不存在的关节点预测值手动置零
            loss += global_joint_loss * 500
            global_loss_record_joint += global_joint_loss.data.item()
        root_map_pred = torch.unsqueeze(refine_output[:, 0, :, :], dim=1)
        kps_map_pred = refine_output[:, 1:, :, :]
        refine_root_loss = criterion_refine_root(root_map_pred, root_map_var)
        loss += refine_root_loss * 10
        refine_loss_record_root = refine_root_loss.data.item()
        refine_joint_loss = criterion_refine_joint(kps_map_pred.mul(kps_weight_var), kps_map_var)
        refine_joint_loss = refine_joint_loss.mean(dim=3).mean(dim=2)
        refine_joint_loss = ohkm(refine_joint_loss, 16)
        loss += refine_joint_loss * 5000
        refine_loss_record_joint = refine_joint_loss.data.item()

        losses.update(loss.data.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i%10==0 and i!=0):
            print('iteration {} | loss: {}, global root loss: {}, global joint loss: {}, refine root loss: {}, refine joint loss: {}, avg loss: {}'
                  .format(i, loss.data.item(), global_loss_record_root, global_loss_record_joint,
                          refine_loss_record_root, refine_loss_record_joint, losses.avg))

    return losses.avg

def validation(val_loader, model, criterions):
    losses = AverageMeter()
    model.eval()
    for i, (inputs, root_map, kps_map, kps_weight, widths, heights, img_id) in enumerate(val_loader):
        input_var = torch.autograd.Variable(inputs.cuda())
        root_map_var = torch.autograd.Variable(root_map.cuda(async=True))
        kps_map_var = torch.autograd.Variable(kps_map.cuda(async=True))
        kps_weight_var = torch.autograd.Variable(kps_weight.cuda(async=True))

        global_outputs, refine_output = model(input_var)
        root_map_pred = torch.unsqueeze(refine_output[:, 0, :, :], dim=1)
        kps_map_pred = refine_output[:, 1:, :, :]
        refine_loss_root = criterions[0](root_map_pred, root_map_var)
        refine_loss_joint = criterions[1](kps_map_pred.mul(kps_weight_var), kps_map_var)
        loss = refine_loss_root + refine_loss_joint

        if (i % 10 == 0 and i != 0):
            print('iteration {} | loss: {}, refine root loss: {}, refine joint loss: {}, avg loss: {}'
                  .format(i, loss.data.item(), refine_loss_root, refine_loss_joint, losses.avg))

        losses.update(loss.data.item(), inputs.size(0))

    return losses.avg



def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 8 epochs"""
    lr = lr * (0.6 ** (epoch // 2))
    print('LR:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_model(state, checkpoint='checkpoint', filename='checkpoint'):
    filename = filename + '_epoch'+str(state['epoch']) + '.pth.tar'
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch SPM Training')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N',
                        help='batch_size (default: 128)')
    parser.add_argument('--epochs', default=32, type=int, metavar='N',
                        help='number of total epochs to run (default: 32)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='directory to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('-r', '--reset', dest='reset', action='store_true',
                        help='reset the start epoch')
    parser.add_argument('--gpu', '-g', default="0，1", type=str, help='specify which gpu(s) to use')


    main(parser.parse_args())

