import json
import os
import re
import numpy as np
import torch
import torch.utils.data as data
import pdb
from utils import get_data_path
from utils import encode_sentence, _load_dictionary
from PIL import Image

class CocoCaptionsRV(data.Dataset):

    def __init__(self, sset="train", transform=None):
        path = get_data_path('coco')
        self.root = os.path.join(path["COCO_ROOT"], "images/")
        self.transform = transform
        # dataset.json come from Karpathy neural talk repository and contain the restval split of coco
        with open(path["COCO_RESTVAL_SPLIT"], 'r') as f:
            datas = json.load(f)

        if sset == "train":
            self.content = [x for x in datas["images"] if x["split"] == "train"]
        elif sset == "trainrv":
            self.content = [x for x in datas["images"] if x["split"] == "train" or x["split"] == "restval"]
        elif sset == "val":
            self.content = [x for x in datas["images"] if x["split"] == "val"]
        else:
            self.content = [x for x in datas["images"] if x["split"] == "test"]

        self.content = [(os.path.join(y["filepath"], y["filename"]), [x["raw"] for x in y["sentences"]]) for y in self.content]
        self.word_dict_path = path["WORD_DICT"]
        path_params = os.path.join(self.word_dict_path, 'utable.npy')
        self.params = np.load(path_params, encoding='latin1', allow_pickle=True)
        self.dico = _load_dictionary(self.word_dict_path)

    def __getitem__(self, index, raw=False):
        idx = index / 5
        idx_cap = index % 5
        path = self.content[int(idx)][0]
        target_raw = self.content[int(idx)][1][idx_cap]
        if raw:
            return path, target_raw

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        target = encode_sentence(target_raw, self.params, self.dico)

        return img, target

    def __len__(self):
        return len(self.content) * 5


class FlickrDataset(data.Dataset):

    def __init__(self, split="train", transform=None):
        
        path = get_data_path('flickr')

        imgdir = os.path.join(root, 'flickr30k-images')
        self.roots = imgdir
        self.transform = transform
        self.dataset = json.load(open(json_file_path, 'r'))['images']
        self.ids = []
        path_params = os.path.join(word_dict_path, 'utable.npy')
        self.params = np.load(path_params, encoding='latin1')
        self.dico = _load_dictionary(word_dict_path)
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]

    def __getitem__(self, index):

        root_img = self.roots
        cap_id = self.ids[index]
        img_id = cap_id[0]
        caption = self.dataset[img_id]['sentences'][cap_id[1]]['raw']
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root_img, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        target = encode_sentence(caption, self.params, self.dico)

        return image, target

    def __len__(self):
        return len(self.ids)