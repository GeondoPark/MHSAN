import argparse
import os
import time
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as uu
import pdb
from dataset import CocoCaptionsRV
from evaluation import eval_recall
from utils import AverageMeter, save_checkpoint, collate_fn_padded, log_epoch
from torch.utils.data import DataLoader
from utils import encode_sentence, _load_dictionary


print_freq = 1000

prepro_val = transforms.Compose([
    transforms.Resize((350, 350)),
    transforms.ToTensor(),
    ])


coco_data_val = CocoCaptionsRV(sset="val", transform=prepro_val)

val_loader = DataLoader(coco_data_val, batch_size=160, shuffle=False,
                    num_workers=4, collate_fn=collate_fn_padded, pin_memory=True)
#dico = _load_dictionary(word_dict_path)
print("start")
for i, (imgs, caps, lengths) in enumerate(val_loader):
    if i % print_freq == 0 or i == (len(val_loader) - 1):
    
        #a = torch.cat([imgs[0].unsqueeze(0),imgs[11].unsqueeze(0),imgs[21].unsqueeze(0),imgs[34].unsqueeze(0)], dim=0)
        #a = uu.make_grid(a)
        #uu.save_image(a,"examples/image{}.jpg".format(i))

        print(caps[0])
        print(caps[11])
        print(caps[21])
        print(caps[34])
        print("end")