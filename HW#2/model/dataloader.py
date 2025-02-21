import torch
from torch import nn
import numpy as np
from pathlib import Path
from transformers import GPT2TokenizerFast
from model.timer import time_it
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from transformers import GPT2TokenizerFast

@time_it
def read_corpus(filename : Path|str,tokenizer : GPT2TokenizerFast):
    seq = []
    with open(filename,'rt',encoding='utf-8',) as f:
        for line in f:
            line = line.replace('\n','')
            tokens = tokenizer(line)
            for t in tokens['input_ids']:
                seq.append(t)
    return(seq)

@time_it
def get_number_of_model_params(model: nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def nopeak_mask(d_model:int, batch_size:int, cuda=True):
    """remove ability for current wrod int embedding sequence to see ahead in the sequence"""
    nopeak = np.tril(np.ones((batch_size,d_model,d_model))).astype('bool')
    return Variable(torch.from_numpy(nopeak)).cuda() if cuda else Variable(nopeak)

def create_mask(tut:torch.Tensor, pad_id:int, make_masked:bool):
    """tut = tensor under test"""
    d_model = tut.size(dim=1)
    tut_mask = (pad_id != tut).unsqueeze(-2).cuda()

    if make_masked:
        mask = nopeak_mask(d_model, batch_size=tut.size(dim=0))
        # print('mask size: ', mask.size())
        # print('tut_mask size: ', tut_mask.size())
        #unsqueezes the second to last
        # batchsize, 1, d_model, d_model
        # idk whgy this fixes eveyrthing but it does
        tut_mask = tut_mask & mask
    return tut_mask


# this is incredibly dumb
class CustomTransformerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset:list[int], seq_len:int, tokenizer:GPT2TokenizerFast=None,cuda = True,verbose=False, **kwargs):
        self.cuda = cuda
        self.dataset = dataset
        self.seq_len = seq_len
        self.verbose = verbose
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.dataset) - self.seq_len

    def __getitem__(self, idx):
        raw = self.dataset[idx:idx+self.seq_len+1]
        source = torch.tensor(raw, dtype=torch.int64)
        tgt = torch.tensor(raw, dtype=torch.int64)
        if self.cuda:
            source = source.cuda()
            tgt = tgt.cuda()
        if self.verbose:
            print('raw ids: ', raw)
            if self.tokenizer:
                print('translated back: ', self.tokenizer.decode(raw))
            print('masked: ', )
        #have two different tensors for now
        return source, tgt