import os
import nltk
import pickle
import shutil
import torch
import numpy as np
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence
import pdb

def get_data_path(data):
    if data== 'coco':
        path = {
            "COCO_ROOT": "./data/coco/",
            "COCO_RESTVAL_SPLIT": "./data/coco/dataset.json",
            "WORD_DICT": "./data",
            }
    else:
        path = {
            "flickr_ROOT" : "./data/flickr",
            "flickr_json" : "./data/flickr/dataset_flickr30k.json",
            "WORD_DICT": "./data",
            }
    return path

class AverageMeter(object):

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


def _load_dictionary(dir_st):
    path_dico = os.path.join(dir_st, 'dictionary.txt')
    if not os.path.exists(path_dico):
        print("Invalid path no dictionary found")
    with open(path_dico, 'r') as handle:
        dico_list = handle.readlines()
    dico = {word.strip(): idx for idx, word in enumerate(dico_list)}
    return dico

def preprocess(text):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sent_detector.tokenize(text)
    result = list()
    for s in sents:
        s.replace(".","")
        tokens = word_tokenize(s)
        result.append(tokens)

    return result

def flatten(l):
    return [item for sublist in l for item in sublist]


def encode_sentences(sents, embed, dico):
    sents_list = list()
    for sent in sents:
        sent_tok = preprocess(sent)[0]
        sent_in = Variable(torch.FloatTensor(1, len(sent_tok), 620))
        for i, w in enumerate(sent_tok):
            try:
                sent_in.data[0, i] = torch.from_numpy(embed[dico[w]])
            except KeyError:
                sent_in.data[0, i] = torch.from_numpy(embed[dico["UNK"]])

        sents_list.append(sent_in)
    return sents_list
    #return sent_tok

def encode_sentence(sent, embed, dico, tokenize=True):
    
    if tokenize:
        sent_tok = preprocess(sent)[0]
    else:
            sent_tok = sent

    sent_in = torch.FloatTensor(len(sent_tok), 620)
    for i, w in enumerate(sent_tok):
        try:
            sent_in[i, :620] = torch.from_numpy(embed[dico[w]])
        except KeyError:
            sent_in[i, :620] = torch.from_numpy(embed[dico["UNK"]])

    return sent_in

def save_checkpoint(state, val_best, model_name, epoch):
    file_name = os.path.join('./weights/',model_name, model_name+'_checkpoint.pth.tar')
    torch.save(state, file_name)
    if val_best:
        shutil.copyfile(file_name, os.path.join('./weights/',model_name, model_name+'_best_model.pth.tar'))

def log_epoch(logger, epoch, train_loss, val_loss, lr, recall_val):
    logger.add_scalar('Loss/Train', train_loss, epoch)
    logger.add_scalar('Loss/Val', val_loss, epoch)
    logger.add_scalar('Learning/Rate', lr, epoch)
    logger.add_scalar('Learning/Overfitting', val_loss / train_loss, epoch)
    logger.add_scalar('Recall/Val/CapRet/R@1', recall_val[0][0], epoch)
    logger.add_scalar('Recall/Val/CapRet/R@5', recall_val[0][1], epoch)
    logger.add_scalar('Recall/Val/CapRet/R@10', recall_val[0][2], epoch)
    logger.add_scalar('Recall/Val/CapRet/MedR', recall_val[2], epoch)
    logger.add_scalar('Recall/Val/ImgRet/R@1', recall_val[1][0], epoch)
    logger.add_scalar('Recall/Val/ImgRet/R@5', recall_val[1][1], epoch)
    logger.add_scalar('Recall/Val/ImgRet/R@10', recall_val[1][2], epoch)
    logger.add_scalar('Recall/Val/ImgRet/MedR', recall_val[3], epoch)

def collate_fn_padded(data):
    images, captions = zip(*data)
    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    targets_raw = captions
    targets = pad_sequence(captions, batch_first=True)
    return images, targets, lengths

def collate_fn_cap_padded(data):
    captions = data
    lengths = [len(cap) for cap in captions]
    targets = pad_sequence(captions, batch_first=True)

    return targets, lengths

def collate_fn_semseg(data):
    images, size, targets = zip(*data)
    images = torch.stack(images, 0)

    return images, size, targets

def collate_fn_img_padded(data):
    images = data
    images = torch.stack(images, 0)

    return images

def load_obj(path):
    with open(os.path.normpath(path + '.pkl'), 'rb') as f:
        return pickle.load(f)

def save_obj(obj, path):
    with open(os.path.normpath(path + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def set_random_seed(random_seed):
    if random_seed is None:
        random_seed = 4242
    import random
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)

def load_model(file_name):
    if os.path.isfile(file_name):
        print("=> loading checkpoint '{}'".format(file_name))
        checkpoint = torch.load(file_name)
        start_epoch = checkpoint['epoch']
        best_rsum = checkpoint['best_rec']
        join_emb.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
              .format(file_name, start_epoch, best_rsum))
        return start_epoch, best_rsum

    else:
        print("=> no checkpoint found at '{}'".format(file_name))

def print_requries_grad(network):
    for name, weight in network.named_parameters():
        if weight.requires_grad == True:
            print(name, weight.requires_grad)

def adjust_learning_rate(args, optimizer, epoch):
    if epoch in args.lr_decay_list:
        for idx, decay in enumerate(args.lr_decay_list):
            if epoch == args.lr_decay_list[idx]:
                args.learning_rate = optimizer.param_groups[0]['lr'] * args.lr_decay[idx]

        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate

        print('\n----Decayed learning rate by a factor {} to {}----\n'.format(args.lr_decay, args.learning_rate))
