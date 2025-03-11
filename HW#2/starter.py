import argparse
import os
import sys
import shutil
import random
import numpy as np
import time
import copy
import math
import pickle
from tqdm import tqdm
# from torchtext.legacy import data
# from model.masks import nopeak_mask
from torcheval.metrics.text import Perplexity
from model.timer import time_it
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from transformers import GPT2TokenizerFast
from model.model import Transformer
from model.cosinerestarts import CosineWithRestarts
from model.dataloader import read_corpus, get_number_of_model_params, CustomTransformerDataset, create_mask


@time_it
def get_model(opt : argparse.Namespace, src_vocab_size:int, trg_vocab_size:int) -> nn.Module:
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(src_vocab_size, trg_vocab_size, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    model.to(opt.device)
    if opt.loadname is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(opt.loadname))
    else:
        #flash the model parameters
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    return model



@time_it
def train_model(model :nn.Module, dataloader: DataLoader, optimizer: torch.optim.Adam,loss_func: F.cross_entropy, epoch:int, batchsize:int, vocab_size:int, device:str,savepath:Path=None, verbose:bool=True):
    
    print("training model...")
    model.train()
    
    # write code to:
    #  1. create a nopeak mask
    #  2. feed training data to the model in batches
    #  3. send the indices of training tokens to the GPU
    #  4. linearize the predictions and compute the loss against ground truth
    #     (you can use F.cross_entropy or write your own code)
    #  5. calculate and apply the gradients with loss.backward() and optimizer.step()
    #  6. report intermediate trainining perplexity
    #  7. generate a test perplexity once per training epoch by calling test_model()
    #  8. save model weights to file specified in opt.savename
    #  SEE trainer.py for examples of each of the above
    with tqdm(total=len(dataloader), desc='Training Progress') as progress:
        for i, input in enumerate(dataloader):
            tgt : torch.Tensor = input.to(device)
            input_tokens = tgt[:, :-1]
            mask = create_mask(input_tokens, 0, make_masked=True)
            preds : torch.Tensor = model(input_tokens=input_tokens, mask=mask).permute(0,2,1)
            # print('time to do inference: ', e-s)
            # preds = preds.view(-1, preds.size(-1))
            # true_output = tgt[:, 1:].contiguous().view(-1)
            true_output = tgt[:, 1:]


            #unoptimized python code
            # batchsize, seq_len = true_output.size(0), true_output.size(1)

            # # print('true output size: ', true_output.size())

            # one_hot_true_output = torch.zeros((batchsize,seq_len,vocab_size)).cuda()

            # for i in range(batchsize):
            #     for j in range(seq_len):
            #         one_hot_true_output[i,j,true_output[j]] = 1

            # print('one_hot_true_output: ', one_hot_true_output)
            # print('one_hot_true_output size: ', one_hot_true_output.size())



            #pytorch functions

            # one_hot_true_output = F.one_hot(true_output, num_classes=vocab_size)
            # print('one_hot_true_output: ', one_hot_true_output)
            # print('one_hot_true_output size: ', one_hot_true_output.size())


            # print('preds: ', preds)
            # print('preds size: ', preds.size())
            # print('true_output: ', true_output)
            # print('true_output size: ', true_output.size())
            # print("true_ourpur: ", true_output)
            # print('true_ourpur: ', true_output.size())
            probs  = F.softmax(preds, dim=-1)

            # # manual cross_entropy
            # log_out = one_hot_true_output * torch.log(probs)
            # loss = (-log_out).sum()
            # print('loss: ', loss)
            loss = loss_func(probs, true_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            progress.set_description(f'Epoch: {epoch}')
            progress.update(1)
            progress.set_postfix({'loss':loss.item()})
        if savepath:
            torch.save(model.state_dict(), f"{str(savepath)}{time.strftime('%H-%M-%S')}" )
    # nopeak_mask =


@time_it
def test_model(model, dataloader: DataLoader, device:str, verbose=False):
    print("testing model...")
    model.eval()
    if verbose:
        print("model:", model)
        print("dataloader:", dataloader)
    print('inside perplexity')
    model.eval()
    metric = Perplexity(ignore_index=0,device='cuda:0')
    for i, input in enumerate(dataloader):
        tgt : torch.Tensor = input.to(device)
        input_tokens = tgt[:, :-1]
        tgtmask = create_mask(input_tokens, 0, make_masked=True)
        preds : torch.Tensor = model(input_tokens=input_tokens, mask=tgtmask)
        true_output = tgt[:, 1:].contiguous()
        metric.update(preds,true_output)
    if not i%1000:
        print('Batch num: ', i, ' ', f"{i*100/len(dataloader):.1f}", '%')
    model.train()
    return metric.compute().cpu().item()

global tokenizer
tokenizer : GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")

def main():
    
    random.seed(10)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true')
    #stocastic gradicent decent with restarts
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=1)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=int, default=0.00001)
    parser.add_argument('-seqlen', type=int, default=512)
    parser.add_argument('-savename', type=str)
    parser.add_argument('-loadname', type=str)
    parser.add_argument('-tied', type=int, default=1)
    parser.add_argument('-dir_name', type=str,default='model')
    parser.add_argument('-norm', type=float, default=2.0)


    opt = parser.parse_args()
    opt.verbose = False
    
    opt.device = 0 if opt.no_cuda is False else -1
    if opt.device == 0:
        assert torch.cuda.is_available()
        opt.device = torch.device("cuda:0")
    

    #make a log file??? 
    time_name = time.strftime("%y%m%d_%H%M%S")
    opt.time_name = time_name
    dir_name = "saved/%s" % (opt.dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    source_name = sys.argv[0]
    dir_name = dir_name + "//"
    opt.dir_name = dir_name
    #copy the python file at benging of run time???
    shutil.copy(source_name,dir_name + source_name)
    opt.log_file = dir_name + "log_file.txt"
    opt.train = read_corpus(Path('.') / 'data'/ 'wiki2.train.txt',tokenizer)
    opt.valid = read_corpus(Path('.') /'data'/ 'wiki2.valid.txt',tokenizer)
    opt.test = read_corpus(Path('.') /'data'/ 'wiki2.test.txt',tokenizer)
    print('first 11 tokens: ', tokenizer.decode(opt.train[0:10]))
    print('first 11 token ids: ',opt.train[0:10] )
    
    obs = len(opt.train)
    opt.vocab_size = 50257
    indices = torch.from_numpy(np.arange(50527))
    model = get_model(opt,opt.vocab_size,opt.vocab_size)

    print('number of params: ', get_number_of_model_params(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.savename is not None:
        try:
            os.mkdir(opt.savename)
        except:
            nothing = 1

    train_dataset = DataLoader(CustomTransformerDataset(opt.train, opt.seqlen, tokenizer=tokenizer, verbose=False),batch_size=opt.batchsize)
    print(len(train_dataset))
    test_dataset = DataLoader(CustomTransformerDataset(opt.test, opt.seqlen, tokenizer=tokenizer, verbose=False),batch_size=opt.batchsize)
    # valid_dataset = CustomTransformerDataset(opt.valid, opt.seqlen, tokenizer=tokenizer, verbose=False)

    
    opt.src_pad = 0
    opt.trg_pad = 0
    perplexity = []
    for epoch in range(opt.epochs):
        train_model(model,train_dataset, loss_func= F.cross_entropy, epoch=1, batchsize=opt.batchsize, optimizer=optimizer, vocab_size=opt.vocab_size, device=opt.device, savepath=(Path(os.path.abspath(__file__)).parent / 'saved' / 'model' / f'{opt.savename}'))
        perplexity.append(test_model(model,test_dataset, device=opt.device))
        print('current perplexity: ', perplexity)

if __name__ == "__main__":
    main()