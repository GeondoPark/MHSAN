import argparse
import os
import time
import shutil
import pickle
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from dataset import CocoCaptionsRV
from evaluation import eval_recall
from loss import HardNegativeContrastiveLoss, DiversityLoss
from model import joint_embedding
from utils import AverageMeter, save_checkpoint, collate_fn_padded, log_epoch, adjust_learning_rate, print_requries_grad
from tensorboardX import SummaryWriter

device = torch.device("cuda")

def main():

    parser = argparse.ArgumentParser(description='COCO_DATASET_TRAIN')
    parser.add_argument("-hop", '--attn_hop', default=1, type=int,
                        help='Number of Attention Hop')
    parser.add_argument("-hidden", '--attn_hidden', default=350, type=int,
                        help='Number of Attention-unit')
    parser.add_argument("-name", '--model-name', default="model",
                        help='Name of the model')
    parser.add_argument("-pf", dest="print_frequency", type=int, default=50,
                        help="Number of element processed between print")
    parser.add_argument("-bs", "--batch_size", type=int, default=160,
                        help="The size of the batches",)
    parser.add_argument("-lr", "--learning_rate", dest="lr", type=float, default=0.005,
                        help="Initialization of the learning rate")  # 0.001
    parser.add_argument("-lr_decay_list", "--lr_decay_list", type=int, default=[2, 5, 10, 40],
                        help="List of epoch where the learning rate is decreased"
                        "(multiplied by first arg of lrd)", nargs='+')
    parser.add_argument("-lr_decay", "--lr_decay", nargs='+', default=[2, 5, 10, 40],
                        help="List of epoch where the learning rate is decreased"
                        "(multiplied by first arg of lrd)")
    parser.add_argument("-p-coeff", "--penalty_coeff", type=float, default=0.1,
                        help="Penalty Coefficient")  # 0.001
    parser.add_argument("-epoch", dest="max_epoch", type=int, default=100,
                        help="Max epoch")
    parser.add_argument('-sru', dest="sru", type=int, default=4)
    parser.add_argument("-de", dest="dim_emb", type=int, default=2400
                        help="Dimension of the joint embedding")
    parser.add_argument("-grad_clip", dest="grad_clip", type=float, default=0.0,
                        help="Max of gradient clipping")
    # parser.add_argument('--resume', default='', type=str, metavar='PATH',
    #                    help='path to latest checkpoint (default: none)')

    args = parser.parse_args()

    if not(os.path.isdir(os.path.join('./weights', args.model_name))):
        os.makedirs(os.path.join('./weights', args.model_name))
    shutil.copyfile('model.py', './weights/{name}/{name}_model.py'.format(name=args.model_name))
    shutil.copyfile('train.py', './weights/{name}/{name}_train.py'.format(name=args.model_name))
    with open('./weights/{}/args.txt'.format(args.model_name), 'w') as f:
        f.write(str(args))

    logger = SummaryWriter(os.path.join("./logs/", args.model_name))

    print("Initializing embedding ...", end=" ")
    join_emb = joint_embedding(args)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_valid = transforms.Compose([
        transforms.Resize((350, 350)),
        transforms.ToTensor(),
        normalize,
    ])

    print("Loading Data ...", end=" ")

    coco_data_train = CocoCaptionsRV(sset="trainrv", transform=transform_train)
    coco_data_val = CocoCaptionsRV(sset="val", transform=transform_valid)

    train_loader = DataLoader(coco_data_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn_padded, pin_memory=True)
    val_loader = DataLoader(coco_data_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn_padded, pin_memory=True)

    criterion = HardNegativeContrastiveLoss()

    if args.penalty_coeff != 0:
        diversity_loss = DiversityLoss(args)
    else:
        diversity_loss = None

    join_emb.to(device)

    # Text pipeline frozen at the begining
    print(" \nTrain Image Fully connected layer\n")
    for param in join_emb.cap_emb.module.parameters():
        param.requires_grad = False
    for param in join_emb.img_emb.module.parameters():
        param.requires_grad = False

    print_requries_grad(join_emb)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, join_emb.parameters()), lr=args.lr)

    best_val = 0
    for epoch in range(0, args.max_epoch):

        val_best = False

        adjust_learning_rate(args, optimizer, epoch)
        train_loss, batch_train = train(train_loader, join_emb, criterion, optimizer, epoch,
                                        print_freq=args.print_frequency,
                                        diversity_loss=diversity_loss,
                                        grad_clip=args.grad_clip)

        val_loss, batch_val, recall_val = validate(val_loader, join_emb, criterion, print_freq=args.print_frequency)

        if(sum(recall_val[0]) + sum(recall_val[1]) > best_val):
            best_val = sum(recall_val[0]) + sum(recall_val[1])
            val_best = True

        state = {
            'epoch': epoch,
            'state_dict': join_emb.state_dict(),
            'best_val': best_val,
            'args_dict': args,
            'optimizer': optimizer.state_dict(),
        }

        log_epoch(logger, epoch, train_loss, val_loss, optimizer.param_groups[0]['lr'], recall_val)
        save_checkpoint(state, val_best, args.model_name, epoch)

        # Optimizing the text pipeline after one epoch
        if epoch == 4:
            print(" \nAt epoch {} Open parameters of Captions\n".format(epoch + 1))
            for param in join_emb.cap_emb.module.parameters():
                param.requires_grad = True

            print_requries_grad(join_emb)

            optimizer.add_param_group({'params': filter(lambda p: p.requires_grad,
                                                        join_emb.cap_emb.module.parameters()), 'lr': optimizer.param_groups[0]['lr']})

        # Starting the finetuning of the whole model
        if epoch == 19:
            print(" \nAt epoch {} Open parameters of Image\n".format(epoch + 1))
            finetune = True

            for param in join_emb.parameters():
                param.requires_grad = True
            # Keep the first layer of resnet frozen
            for i in range(0, 6):
                for param in join_emb.img_emb.module.base_layer[i].parameters():
                    param.requires_grad = False

            print_requries_grad(join_emb)

            optimizer.add_param_group({'params': filter(lambda p: p.requires_grad,
                                                        join_emb.img_emb.module.parameters()), 'lr': optimizer.param_groups[0]['lr']})

    print('Finished Training')


def train(train_loader, model, criterion, optimizer, epoch, print_freq=1000, diversity_loss=None, grad_clip = 0):

    batch_time = AverageMeter()
    losses = AverageMeter()
    penalty_loss_img = AverageMeter()
    penalty_loss_txt = AverageMeter()

    model.train()

    for i, (imgs, caps, lengths) in enumerate(train_loader):
        end = time.time()
        input_imgs, input_caps = imgs.to(device, non_blocking=True), caps.to(device, non_blocking=True)
        output_imgs, output_caps, img_attn, txt_attn = model(input_imgs, input_caps, lengths)
        loss = criterion(output_imgs, output_caps)
        losses.update(loss.item(), imgs.size(0))

        if diversity_loss:
            if epoch > 4:
                txt_diversity = diversity_loss.cal_loss(txt_attn)
                loss += txt_diversity
                penalty_loss_txt.update(txt_diversity.item(), imgs.size(0))

            if epoch > 19:
                img_diversity = diversity_loss.cal_loss(img_attn)
                loss += img_diversity
                penalty_loss_img.update(img_diversity.item(), imgs.size(0))

        optimizer.zero_grad()
        if grad_clip != 0:
            clip_grad_norm_(parameters=list(filter(lambda p:p.requires_grad, model.parameters())), max_norm=grad_clip)
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        
        if i % print_freq == 0 or i == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'lr : {lr}\t'
                  'TxtLoss {t_div.avg:.4f}\t'
                  'ImgLoss {i_div.avg:.4f}\t'
                  'Loss {loss.avg:.4f}\t'
                  .format(epoch, i, len(train_loader),
                          lr=optimizer.param_groups[0]['lr'],
                          t_div=penalty_loss_txt, i_div=penalty_loss_img,
                          batch_time=batch_time, loss=losses))

    return losses.avg, batch_time.avg


def validate(val_loader, model, criterion, print_freq=100):

    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    imgs_enc = list()
    caps_enc = list()

    for i, (imgs, caps, lengths) in enumerate(val_loader):

        input_imgs, input_caps = imgs.to(device, non_blocking=True), caps.to(device, non_blocking=True)

        end = time.time()
        with torch.no_grad():
            output_imgs, output_caps, attn, attn_im = model(input_imgs, input_caps, lengths)
            loss = criterion(output_imgs, output_caps)

        imgs_enc.append(output_imgs.cpu().data.numpy())
        caps_enc.append(output_caps.cpu().data.numpy())
        losses.update(loss.item(), imgs.size(0))

        batch_time.update(time.time() - end)

        if i % print_freq == 0 or i == (len(val_loader) - 1):
            print('Data: [{0}/{1}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  .format(i, len(val_loader),
                          batch_time=batch_time, loss=losses))

    recall = eval_recall(imgs_enc, caps_enc)
    print(recall)
    return losses.avg, batch_time.avg, recall


if __name__ == '__main__':
    main()
