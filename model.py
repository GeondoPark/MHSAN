import torch
import torch.nn as nn
from sru import SRU
import numpy as np
import torchvision.models as models

class SruEmb(nn.Module):
    def __init__(self, args , dim_in):
        super(SruEmb, self).__init__()

        self.dim_out_rnn = args.dimemb
        self.dim_out = args.dimemb
        self.rnn = SRU(dim_in, self.dim_out_rnn, num_layers=args.sru,
                       dropout=0.25, rnn_dropout=0.25,
                       use_tanh=True)

        #self.rnn = nn.LSTM(dim_in, self.dim_out, num_layers=nb_layer,
        #                    dropout = dropout, bidirectional=True)
        self.attn_hop = args.attn_hop
        self.attn_hidden = args.attn_hidden
        self.ws1 = nn.Linear(self.dim_out, self.attn_hidden, bias=False)
        self.ws2 = nn.Linear(self.attn_hidden, self.attn_hop, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        #Number 1 is p = 0.25
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.dim_out*self.attn_hop, self.dim_out, bias=True)

    def _select_last(self, x, lengths):
        batch_size = x.size(0)
        mask = x.data.new().resize_as_(x.data).fill_(0)
        for i in range(batch_size):
            mask[i][lengths[i] - 1].fill_(1)
        x = x.mul(mask)
        x = x.sum(1, keepdim=True).view(batch_size, self.dim_out)
        return x

    def _process_lengths(self, input):
        lengths = []
        max_length = input.size(1)
        non_word_num = input.data.eq(0)[:,:,0].sum(1).tolist()
        for i in range(input.size(0)):
            lengths.append(max_length-non_word_num[i])
        return lengths

    def attn_mask(self, attn_value, lengths):
        batch_size = attn_value.size(0)
        hop_number = attn_value.size(1)
        max_len = attn_value.size(2)
        mask = torch.zeros(batch_size, max_len, dtype=torch.uint8)
        for i, length in enumerate(lengths):
            mask[i, length:] = 1
        mask = mask.unsqueeze(1).expand(-1, hop_number, -1).cuda()

        return mask
    
    def forward(self, input, lg):

        lengths = self._process_lengths(input)
        x = input.permute(1, 0, 2)
        x, hn = self.rnn(x)
        x = x.permute(1, 0, 2).contiguous()
        size = x.size()
        compressed_embeddings = x.view(-1, size[2])
        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))                #[bsz*len, attention-unit
        alphas = self.ws2(hbar).view(size[0], size[1], -1)                          #[bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()                         #[bsz, hop, len]
        mask = self.attn_mask(alphas, lengths)
        attention_value = alphas.masked_fill(mask, -np.inf)
        attention_value = self.softmax(attention_value)                             #[batch, hop, len]
        attention_value = attention_value.view(size[0], self.attn_hop, size[1])
        output = torch.bmm(attention_value, x)                                      #[batch, len hidden_dim*2]
        output = output.view(output.size(0), -1)                                    #[batch, len*hidden_dim*2]

        output = self.fc(self.drop(output))

        return output, attention_value


class img_embedding(nn.Module):

    def __init__(self, args, pretrained=True, D=2048):
        super(img_embedding, self).__init__()

        resnet = models.resnet152(pretrained=pretrained)
        self.dimemb= args.dimemb
        self.attn_hidden = args.attn_hidden
        self.attn_hop = args.attn_hop
        self.base_layer = nn.Sequential(*list(resnet.children())[:-2])
        self.spaConv = nn.Conv2d(D, self.dimemb, 1)
        self.fc1 = nn.Linear(2400, self.attn_hidden, bias=False)
        self.fc2 = nn.Linear(self.attn_hidden, self.attn_hop, bias=False)
        self.drop = nn.Dropout(p=0.5)
        self.relu =nn.ReLU()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        emb = self.base_layer(x) 
        emb = self.spaConv(emb)                                                     #[bsz, 2400, 8, 8]
        size = emb.size()                                                           #[bsz, 2400, 8, 8]
        unfolded_embedding = emb.view(size[0], 2400, -1)                            #[bsz, 2400, 64]
        emb = unfolded_embedding.transpose(1,2).contiguous()                        #[bsz, 64, 2400]
        unfolded_embedding = emb.view(-1,2400)                                      #[bsz*64, 2400]
        hbar_img = self.relu(self.fc1(self.drop(unfolded_embedding)))               #[bsz*64, attn-unit]
        attn_img = self.fc2(hbar_img).view(size[0], size[2]*size[3], self.attn_hop) #[bsz, 64, hop]
        attn_img = torch.transpose(attn_img, 1, 2).contiguous()                     #[bsz, hop, 64]
        attn_img = self.softmax(attn_img)                                           #[bsz, hop, 64]
        out_img = torch.bmm(attn_img, emb)                                          #[bsz, hop, 2400]
        output = out_img.view(out_img.size(0), -1)                                  #[bsz, hop*2400]

        return output, attn_img

    def get_activation_map(self, x):
        x = self.base_layer[0](x)
        act_map = self.base_layer[1](x)
        act = self.base_layer[2](act_map)
        return act, act_map

class joint_embedding(nn.Module):

    def __init__(self, args):
        super(joint_embedding, self).__init__()

        self.img_emb = nn.DataParallel(img_embedding(args))
        self.cap_emb = nn.DataParallel(SruEmb(args, 620))
        self.fc = nn.DataParallel(nn.Linear(args.dimemb*args.attn_hop, args.dimemb, bias=True))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, imgs, caps, lengths):
        if imgs is not None:
            x_imgs, img_attn = self.img_emb(imgs)
            x_imgs = self.dropout(x_imgs)
            x_imgs = self.fc(x_imgs)

            x_imgs = x_imgs / torch.norm(x_imgs, 2, dim=1, keepdim=True).expand_as(x_imgs)
        else:
            x_imgs = None

        if caps is not None:
            x_caps, txt_attn = self.cap_emb(caps, lengths)
            x_caps = x_caps / torch.norm(x_caps, 2, dim=1, keepdim=True).expand_as(x_caps)
        else:
            x_caps = None
        return x_imgs, x_caps, img_attn, txt_attn
