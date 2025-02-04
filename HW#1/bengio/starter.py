import argparse
import random

import torch
import torch.nn as nn
from embeddings import read_corpus, encode
from model import bengio

wi


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-threshold', type=int, default=3)
    parser.add_argument('-window', type=int, default=512)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-batchsize', type=int, default=1)
    parser.add_argument('-lr', type=float, default=0.00001)
    parser.add_argument('-savename', type=str)
    parser.add_argument('-loadname', type=str)
    return parser



def main():
    
    random.seed(10)

    opt = build_parser()
    opt.verbose = False

    [opt.vocab,opt.words,opt.train] = read_corpus('wiki2.train.txt',[],{},[],opt.threshold)
    print('vocab: %d train: %d' % (len(opt.vocab),len(opt.train)))
    [opt.vocab,opt.words,opt.test] = read_corpus('wiki2.test.txt',opt.vocab,opt.words,[],-1)
    print('vocab: %d test: %d' % (len(opt.vocab),len(opt.test)))
    [opt.vocab,opt.words,opt.valid] = read_corpus('wiki2.valid.txt',opt.vocab,opt.words,[],-1)
    print('vocab: %d test: %d' % (len(opt.vocab),len(opt.valid)))

    print('Train: %7d' % (len(opt.train)))
    print('Test:  %7d' % (len(opt.test)))
    print('Valid: %7d' % (len(opt.valid)))
    print('Vocab: %7d' % (len(opt.vocab)))
    print(' ')

    opt.examples = []
    with open('examples.txt','rt') as f:
        for line in f:
            line = line.replace('\n','')
            encoded = encode(line,opt.words)
            text = ''
            for i in range(len(encoded)):
                text = text + opt.vocab[encoded[i]] + ' '
            opt.examples.append(encoded)
            
            print('origianl: %s' % line)
            print('encoded:  %s' % text)
            print(' ')
            
    model = bengio(dim=opt.d_model, 
                   window=opt.window, 
                   batchsize = opt.batchsize, 
                   vocab_size=len(opt.vocab), 
                   activation=torch.tanh)
    if opt.no_cuda == False:
        model = model.cuda()
    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)    
    
    train(model,opt)
    test_model(model,opt,-1)
    
if __name__ == "__main__":
    main()     
