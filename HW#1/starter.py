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
from bengio.embeddings import read_corpus, encode, encode_example_sentences, decode
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
    parser.add_argument('-lr', help='lr for adam', type=float, default=0.00005)
    parser.add_argument('-savename', help='name of file to save the model to', default='bengio.state',type=str)
    parser.add_argument('-loadname', help='', type=str)
    return parser


@time_it_batch
def train_batch(model : torch.nn.Module,prev_words:torch.Tensor, next_word:torch.Tensor,optimizer: torch.optim.Optimizer, loss:torch.nn.CrossEntropyLoss, verbose:bool=True):
    optimizer.zero_grad()
    model.zero_grad()
    preds = model(prev_words)
    loss_val = loss(preds, next_word)
    loss_val.backward()
    optimizer.step()
    if verbose:
        print('Loss in batch: ', loss_val)
    return loss_val.item()

@time_it
def train(model:torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss:torch.nn.CrossEntropyLoss, savename: str, verbose: bool = False, **kwargs):

    if verbose:
        print("model:", model)
        print("dataloader:", dataloader)
        print("optimizer:", optimizer)
        print("loss function:", loss)
        print("savename:", savename)
        print("verbose:", verbose)
        # if kwargs:
        #     print("kwargs: ", kwargs)
    print('inside train')
    model.train()
    loss_epoch = []
    wps_epoch = []
    percentage_epoch = []
    for i, (prev_words , next_word) in enumerate(dataloader):
        # prev_words: torch.Tensor = prev_words
        # print(prev_words.device)
        loss_batch, time_taken = train_batch(model=model,
                        prev_words=prev_words,
                        next_word=next_word,
                        optimizer=optimizer,
                        loss=loss,
                        verbose=False
                        )
        loss_epoch.append(loss_batch)
        wps_epoch.append(kwargs["window"]*kwargs["batchsize"]/time_taken)
        percentage_epoch.append(i*100/len(dataloader))
        if not i%500:
            print('Batch num: ', i, ' ', f"{percentage_epoch[-1]:.1f}", '%')
            print('words per second: ', f"{wps_epoch[-1]:.1f}")
            # print('Loss across the epoch: ', loss_epoch)

    if savename:
        torch.save(model.state_dict(), Path(os.path.abspath(__file__)).parent / f"{time.strftime("%H-%M-%S")}{savename}")
    return loss_epoch, wps_epoch

@time_it
def test_model(model:torch.nn.Module, dataloader: DataLoader, verbose: bool = False, **kwargs):
    if verbose:
        print("model:", model)
        print("dataloader:", dataloader)
    print('inside test')
    model.eval()
    acc_epoch = []
    for i, (prev_words , next_word) in enumerate(dataloader):
        preds :torch.Tensor = model(prev_words)
        acc_batch = (preds.argmax(dim=1) == next_word).sum().item()
        acc_epoch.append(acc_batch)

    return acc_epoch

@time_it
def perplexity_model(model:torch.nn.Module, dataloader: DataLoader, verbose: bool = False, **kwargs):
    if verbose:
        print("model:", model)
        print("dataloader:", dataloader)
    print('inside perplexity')
    metric = Perplexity(ignore_index=0,device='cuda:0')
    for i, (prev_words , next_word) in enumerate(dataloader):
        preds :torch.Tensor = model(prev_words)
        metric.update(preds.unsqueeze(1),next_word.view(-1,1))
        if not i%1000:
            print('Batch num: ', i, ' ', f"{i*100/len(dataloader):.1f}", '%')

    return metric.compute().cpu().item()


@time_it
def examples_perplexity(model:torch.nn.Module, example_sentence: list[int], window:int, unkid: int, verbose: bool = False, **kwargs):
    print('calulating the perplexity of the examples')

    metric = Perplexity(device='cuda:0')
    guessed_sentence = []
    for i in range(len(example_sentence)):
        sliding_window = [example_sentence[j] if j > -1 else unkid for j in range(i-5,i)]
        words = torch.tensor(sliding_window, dtype=torch.long).unsqueeze(dim=0).to(device="cuda:0")
        next_word = torch.tensor([example_sentence[i]], dtype=torch.long).to(device="cuda:0")
        # print('words shape: ',  words.shape)
        # print('next words shape: ', next_word.shape)
        preds : torch.Tensor = model(words)
        # print('preds shape: ', preds.shape)
        metric.update(preds.view(1,1,-1),next_word.view(-1,1))
        # print('done metric')
        guessed_sentence.append(preds.argmax(dim=1).item())

    return metric.compute().item(), guessed_sentence



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
    examples_path = Path(os.path.abspath(__file__)).parent / "data" / "examples.txt"
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


    examples = encode_example_sentences(examples_path, opt.vocab, opt.words, 1)
    print('\tMaking Model')
    model = bengio(dim=opt.d_model,
                   window=opt.window,
                   batchsize = opt.batchsize,
                   vocab_size=len(opt.vocab))
    
    if opt.loadname:
        model.load_state_dict(torch.load((Path(os.path.abspath(__file__)).parent / opt.loadname),weights_only=True))
    if not opt.no_cuda:
        model = model.cuda()

    print('\tModel Made')
    print('\tMaking Dataset')
    bengio_train = DataLoader(CustomBengioDataset(**{"corpus":opt.train,**vars(opt), "verbose":False,"cuda":True}), opt.batchsize, True)
    bengio_test = DataLoader(CustomBengioDataset(**{"corpus":opt.test,**vars(opt), "verbose":False,"cuda":True}), opt.batchsize, True)
    bengio_valid = DataLoader(CustomBengioDataset(**{"corpus":opt.valid,**vars(opt), "verbose":False,"cuda":True}), opt.batchsize, True)

    print('\tDone Dataset')
    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)

    print('\tCalling train')
    loss_all_epochs = []
    test_all_epochs = []
    wps_all_epochs = []
    for i in range(opt.epochs):
        loss_epoch, wps_epoch = train(**{
                "model":model,
                "dataloader":bengio_train,
                "optimizer":opt.optimizer,
                "loss": torch.nn.CrossEntropyLoss(),
                **vars(opt)})
        loss_all_epochs.append(loss_epoch)
        wps_all_epochs.append(wps_epoch)
        acc_epoch = test_model(model,dataloader=bengio_test)
        test_all_epochs.append(acc_epoch)
        loss_each_epoch = [sum(loss_all_epochs[i])/len(loss_all_epochs[i]) for i in range(len(loss_all_epochs))]
        acc_each_epoch = [sum(test_all_epochs[i])/len(test_all_epochs[i]) for i in range(len(test_all_epochs))]
        wps_each_epoch = [sum(wps_all_epochs[i])/len(wps_all_epochs[i]) for i in range(len(wps_all_epochs))]
        print('loss all epochs: ', loss_each_epoch)
        print('acc all epochs: ', acc_each_epoch)
        print('wps all epochs: ', wps_each_epoch)



    fig, ax = pyplot.subplots(1,3,figsize=(30,10))
    graph_loss_each_epoch_pb(ax[0],loss_all_epochs,x_label="Batch In epoch", y_label="Loss", title=f"hyperparams: lr={opt.lr} epochs={opt.epochs} d={opt.d_model} window={opt.window}")
    graph_loss_each_epoch_pb(ax[1],test_all_epochs,x_label="Batch In epoch", y_label="Acc", title=f"hyperparams: lr={opt.lr} epochs={opt.epochs} d={opt.d_model} window={opt.window}")
    graph_loss_each_epoch_pb(ax[2],wps_all_epochs,x_label="Batch In epoch", y_label="WPS", title=f"hyperparams: lr={opt.lr} epochs={opt.epochs} d={opt.d_model} window={opt.window}")
    time.time
    pyplot.savefig(Path(os.path.abspath(__file__)).parent / f"perf_graph_{time.strftime("%Y-%m-%d_%H-%M-%S")}.png")
    time.sleep(2)
    # pyplot.show()


    loss_each_epoch = [sum(loss_all_epochs[i])/len(loss_all_epochs[i]) for i in range(len(loss_all_epochs))]
    acc_each_epoch = [sum(test_all_epochs[i])/len(test_all_epochs[i]) for i in range(len(test_all_epochs))]
    wps_each_epoch = [sum(wps_all_epochs[i])/len(wps_all_epochs[i]) for i in range(len(wps_all_epochs))]
    fig, ax = pyplot.subplots(1,3,figsize=(30,10))
    graph_loss_pb(ax[0],loss_each_epoch,x_label="Epoch", y_label="Loss", title=f"hyperparams: lr={opt.lr} epochs={opt.epochs} d={opt.d_model} window={opt.window}")
    graph_loss_pb(ax[1],acc_each_epoch,x_label="Epoch", y_label="Acc", title=f"hyperparams: lr={opt.lr} epochs={opt.epochs} d={opt.d_model} window={opt.window}")
    graph_loss_pb(ax[2],wps_each_epoch,x_label="Epoch", y_label="WPS", title=f"hyperparams: lr={opt.lr} epochs={opt.epochs} d={opt.d_model} window={opt.window}")
    pyplot.savefig(Path(os.path.abspath(__file__)).parent / f"perf_graph_{time.strftime("%Y-%m-%d_%H-%M-%S")}.png")
    pyplot.show()

    print('Example setnence perplexity')
    for example_sentence in examples:
        ex_per, sent_int = examples_perplexity(model,example_sentence,5,opt.words['<unk>'][0])
        print('sent_int: ', sent_int)
        reconst_sent = decode(sent_int, opt.rev_words)
        print('Input sentence:         ', decode(example_sentence, opt.rev_words))
        print('Reconstructed sentence: ', reconst_sent)
        print(f'Perplexity: {ex_per:0.1f}')
if __name__ == "__main__":
    main()
