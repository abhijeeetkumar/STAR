import shutil
import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torchvision

from dataset import HERDataSet
from model import Model
from train_options import parser
from transforms import GroupCenterCrop
from transforms import GroupScale


SAVE_FREQ = 40
PRINT_FREQ = 20
best_prec1 = 0

def main():
    global args
    global best_prec1
    args = parser.parse_args()

    print('Training arguments:')
    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))

    if args.data_name == 'ucf101':
        num_class = 101
    elif args.data_name == 'hmdb51':
        num_class = 51
    else:
        raise ValueError('Unknown dataset '+ args.data_name)

    model = Model(num_class, args.num_segments, args.representation,
                  base_model=args.arch)
    '''model = STAM(
        dim = 512,
        image_size = 224,     # size of image
        patch_size = 32,      # patch size
        num_frames = args.num_segments,       # number of image frames, selected out of video
        space_depth = 12,     # depth of vision transformer
        space_heads = 8,      # heads of vision transformer
        space_mlp_dim = 2048, # feedforward hidden dimension of vision transformer
        time_depth = 6,       # depth of time transformer (in paper, it was shallower, 6)
        time_heads = 8,       # heads of time transformer
        time_mlp_dim = 2048,  # feedforward hidden dimension of time transformer
        num_classes = num_class,    # number of output classes
        space_dim_head = 64,  # space transformer head dimension
        time_dim_head = 64,   # time transformer head dimension
        dropout = 0.2,         # dropout
        representation = args.representation,
        emb_dropout = 0.2      # embedding dropout
    )'''


    print(model)

    train_loader = torch.utils.data.DataLoader(
        HERDataSet(
            args.data_root,
            args.data_name,
            video_list=args.train_list,
            num_segments=args.num_segments,
            representation=args.representation,
            transform=model.get_augmentation(),
            is_train=True,
            accumulate=(not args.no_accumulation),
            ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        HERDataSet(
            args.data_root,
            args.data_name,
            video_list=args.test_list,
            num_segments=args.num_segments,
            representation=args.representation,
            transform=torchvision.transforms.Compose([
                GroupScale(int(model.scale_size)),
                GroupCenterCrop(model.crop_size),
                ]),
            is_train=False,
            accumulate=(not args.no_accumulation),
            ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    '''
    # Add weights from checkpoint model if specified
    if args.checkpoint_model:
       state_dict = torch.load(args.checkpoint_model)
       from collections import OrderedDict
       new_state_dict = OrderedDict()
       for k, v in state_dict.items():
           name = k[7:] # remove module.
           new_state_dict[name] = v
       model.load_state_dict(new_state_dict)
    ''' 
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    cudnn.benchmark = True

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        decay_mult = 0.0 if 'bias' in key else 1.0

        if ('module.base_model.conv1' in key
                or 'module.base_model.bn1' in key
                or 'data_bn' in key) and args.representation in ['mv', 'residual']:
            lr_mult = 0.1
        elif '.fc.' in key:
            lr_mult = 1.0
        else:
            lr_mult = 0.01

        params += [{'params': value, 'lr': args.lr, 'lr_mult': lr_mult, 'decay_mult': decay_mult}]

   # Add weights from checkpoint model if specified
    if args.checkpoint_model:
        checkpoint = torch.load(args.checkpoint_model)
        print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

        base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
        model.module.load_state_dict(base_dict)

    optimizer = torch.optim.Adam(
        params,
        weight_decay=args.weight_decay,
        eps=0.001)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for epoch in range(args.epochs):
        cur_lr = adjust_learning_rate(optimizer, epoch, args.lr_steps, args.lr_decay)

        train(train_loader, model, criterion, optimizer, epoch, cur_lr)

        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion)

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if is_best or epoch % SAVE_FREQ == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                    },
                    is_best,
                    filename='checkpoint.pth.tar')

def train(train_loader, model, criterion, optimizer, epoch, cur_lr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):

        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # Reset LSTM hidden state
        #uncomment this #model.module.lstm.reset_hidden_state()

        output = model(input_var)
        output = output.view((-1, args.num_segments) + output.size()[1:])
        output = torch.mean(output, dim=1)

        loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
 
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.7f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                       epoch, i, len(train_loader),
                       batch_time=batch_time,
                       data_time=data_time,
                       loss=losses,
                       top1=top1,
                       top5=top5, 
                       lr=cur_lr)))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
             input_var = torch.autograd.Variable(input)
             target_var = torch.autograd.Variable(target)

             # Reset LSTM hidden state
             #uncomment this#model.module.lstm.reset_hidden_state()

             output = model(input_var)
             output = output.view((-1, args.num_segments) + output.size()[1:])
             output = torch.mean(output, dim=1)
             loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            print(('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                       i, len(val_loader),
                       batch_time=batch_time,
                       loss=losses,
                       top1=top1,
                       top5=top5)))

    print(('Testing Results: Prec@1 {top1.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1, loss=losses)))

    return top1.avg

def save_checkpoint(state, is_best, filename):
    filename = '_'.join((args.model_prefix, args.representation.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.model_prefix, args.representation.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)

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


def adjust_learning_rate(optimizer, epoch, lr_steps, lr_decay):
    decay = lr_decay ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    wd = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = wd * param_group['decay_mult']
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
