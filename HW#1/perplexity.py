import argparse
import random

import torch
from torcheval.metrics.text import Perplexity
from torch.utils.data import DataLoader, dataset
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as pyplot
import torch.nn as nn
from pathlib import Path
import os
from bengio.embeddings import read_corpus, encode, encode_example_sentences
from bengio.model import bengio
from bengio.dataloader import CustomBengioDataset
from util.timer import time_it, time_it_batch
from util.graphing import *


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-threshold', type=int, help='number of times word has to show up in dataset in order to get its own value', default=3)
    parser.add_argument('-window', type=int, help='number of words before cur word to consider when training', default=5)
    parser.add_argument('-no_cuda', help='disable cudad', action='store_true')
    parser.add_argument('-epochs', help='number of runs through the dataset', type=int, default=20)
    parser.add_argument('-d_model', help='number of dimensions per word', type=int, default=512)
    parser.add_argument('-batchsize', help='number of parallel sentences to train on', type=int, default=1)
    parser.add_argument('-lr', help='lr for adam', type=float, default=0.00001)
    parser.add_argument('-savename', help='name of file to save the model to', default='bengio.state',type=str)
    parser.add_argument('-loadname', help='', type=str)
    return parser

@time_it
def perplexity_model(model:torch.nn.Module, dataloader: DataLoader, verbose: bool = False, **kwargs):
    if verbose:
        print("model:", model)
        print("dataloader:", dataloader)
    print('inside perplexity')
    model.eval()
    metric = Perplexity(ignore_index=0,device='cuda:0')
    for i, (prev_words , next_word) in enumerate(dataloader):
        preds :torch.Tensor = model(prev_words)
        metric.update(preds.unsqueeze(1),next_word.view(-1,1))
        if not i%1000:
            print('Batch num: ', i, ' ', f"{i*100/len(dataloader):.1f}", '%')

    return metric.compute().cpu().item()


def main():
    random.seed(10)

    print('\tBeginning parser')

    parser = build_parser()
    opt = parser.parse_args()
    opt.verbose = True

    print('\tEnd parser')



    print('\tLoad files')
    wiki2Train = Path(os.path.abspath(__file__)).parent / "data" / "wiki2.train.txt"
    wiki2Test = Path(os.path.abspath(__file__)).parent / "data" / "wiki2.test.txt"
    wiki2Valid = Path(os.path.abspath(__file__)).parent /  "data" / "wiki2.valid.txt"
    [opt.vocab,opt.words,opt.train,opt.rev_words] = read_corpus(wiki2Train,[],{},[],{},opt.threshold)
    print('vocab: %d train: %d' % (len(opt.vocab),len(opt.train)))
    # view_example_sentences(wiki2Train, opt.vocab, opt.words, 3)

    [opt.vocab,opt.words,opt.test, opt.rev_words] = read_corpus(wiki2Test,opt.vocab,opt.words,[],opt.rev_words,-1)
    print('vocab: %d test: %d' % (len(opt.vocab),len(opt.test)))
    # view_example_sentences(wiki2Test, opt.vocab, opt.words, 3)

    [opt.vocab,opt.words,opt.valid, opt.rev_words] = read_corpus(wiki2Valid,opt.vocab,opt.words,[],opt.rev_words,-1)
    print('vocab: %d  valid: %d' % (len(opt.vocab),len(opt.valid)))
    # view_example_sentences(wiki2Valid, opt.vocab, opt.words, 3)

    print('Train: %7d' % (len(opt.train)))
    print('Test:  %7d' % (len(opt.test)))
    print('Valid: %7d' % (len(opt.valid)))
    print('Vocab: %7d' % (len(opt.vocab)))
    print(' ')


    # examples = view_example_sentences(examples_path, opt.vocab, opt.words, 1)
    print('\tMaking Model')
    model = bengio(dim=opt.d_model,
                   window=opt.window,
                   batchsize = opt.batchsize,
                   vocab_size=len(opt.vocab))
    if not opt.no_cuda:
        model = model.cuda()

    print('\tModel Made')
    print('\tMaking Dataset')
    bengio_test = DataLoader(CustomBengioDataset(**{"corpus":opt.test,**vars(opt), "verbose":False,"cuda":True}), opt.batchsize, True)

    print('\tDone Dataset')
    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)

    print('\tCalling train')
    perplexity_all_epochs = []
    names = [f for f in os.listdir((Path(os.path.abspath(__file__)).parent / 'test_weights' / 'v5')) if f.endswith(".state")]
    for i, name in enumerate(names):
        print('name of state dir: ', name)
        model.load_state_dict(torch.load((Path(os.path.abspath(__file__)).parent / 'test_weights' / 'v5'/ name),weights_only=True))
        perplexity_epoch = perplexity_model(model=model,dataloader=bengio_test)
        perplexity_all_epochs.append(perplexity_epoch)

    fig, ax = pyplot.subplots(1,1,figsize=(10,10))
    graph_loss_pb(ax,perplexity_all_epochs,x_label="Epoch", y_label="Perplexity", title=f"hyperparams: lr={opt.lr} epochs={opt.epochs} d={opt.d_model} window={opt.window}")
    pyplot.savefig(Path(os.path.abspath(__file__)).parent / f"perplexity_graph_{time.strftime("%Y-%m-%d_%H-%M-%S")}.png")

if __name__ == "__main__":
    main()
