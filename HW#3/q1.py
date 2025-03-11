from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import torch.optim as optim

from pathlib import Path
import  torch.nn.functional as F
import os
from util.timer import time_it
from pprint import pprint
import torch
import math
import time
import sys
import json
import numpy as np
from tqdm import tqdm


def load_ds_from_file(file_name:str, verbose:bool) -> list[list[str|int]]:
    samples =[]
    answers = ['A', 'B' , 'C', 'D']
    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)
        base = '[CLS] ' + result['fact1'] + ' [SEP] ' + result['question']['stem'] + ' [SEP] '
        ans = answers.index(result['answerKey'])
        obs = []
        for j in range(4):
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text,label])
        samples.append(obs)
        
        if verbose:
            print(obs)
            print(' ')
            
            print(result['question']['stem'])
            print(' ',result['question']['choices'][0]['label'],result['question']['choices'][0]['text'])
            print(' ',result['question']['choices'][1]['label'],result['question']['choices'][1]['text'])
            print(' ',result['question']['choices'][2]['label'],result['question']['choices'][2]['text'])
            print(' ',result['question']['choices'][3]['label'],result['question']['choices'][3]['text'])
            print('  Fact: ',result['fact1'])
            print('  Answer: ',result['answerKey'])
            print('  ')
    return samples

@time_it
def train_model(model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    dataset: list[list[str, str]],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    linear : torch.Tensor,
    verbose: bool = False,
    **kwargs):

    dataset_size = len(dataset)
    model.train()
    with tqdm(total=dataset_size, desc='Training Progress') as progress:
        for i in range(dataset_size):
            sample = dataset[i]
            texts = [x[0] for x in sample]
            labels = [x[1] for x in sample]
            encoded_inputs = tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                return_tensors="pt").to(device="cuda:0")
            labels =  torch.Tensor(labels).to(dtype=torch.long).cuda()
            preds = model(**encoded_inputs)
            pred_tensor : torch.Tensor = preds.last_hidden_state
            logits = (pred_tensor[:,0,:] @ linear).view(1,-1)
            probability = F.softmax(logits, dim=-1)
            loss = (-labels * torch.log(probability)).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress.set_description(f'Epoch: {epoch}')
            progress.update(1)
            progress.set_postfix({'loss':loss.item()})
            if verbose:
                print('tokens: ', encoded_inputs)
                print('labels: ', labels)
                # print('outputs: ', outputs)
                # print('outputs size: ', outputs.size())
                print('labels: ',      labels)
                print('labels size: ', labels.size())


@time_it
def test_model(model:torch.nn.Module, tokenizer: AutoTokenizer, dataset: list[list[list[int], int]], linear_layer: torch.Tensor, verbose: bool = False, **kwargs):
    if verbose:
        print("model:", model)
        print("tokenizer: ", tokenizer)
        # print("dataset: ", dataset)
    print('inside test')
    model.eval()
    acc_epoch = []
    with torch.no_grad():
        for i, (options) in enumerate(dataset):
            texts = [x[0] for x in options]
            labels = [i for i,x in enumerate(options) if x[1] == 1]
            encoded_inputs = tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                return_tensors="pt").to(device="cuda:0")
            labels =  torch.Tensor(labels).to(dtype=torch.long).cuda()
            preds :torch.Tensor = (model(**encoded_inputs).last_hidden_state)
            preds = (preds[:,0,:] @ linear_layer).view(-1)
            probs = F.softmax(preds, dim=0)
            acc_batch = (probs.argmax(dim=-1) == labels).sum().item()
            acc_epoch.append(acc_batch)
    acc_epoch = sum(acc_epoch) / len(acc_epoch)
    print("acc_epoch: ", acc_epoch)
    return acc_epoch

class CudaNotAvailException(Exception):
    def __init__(self,message):
        super().__init__(message)

def main():
    if not torch.cuda.is_available():
        raise CudaNotAvailException("cuda is not available")
    torch.manual_seed(0)
    answers = ['A','B','C','D']

    train = []
    false = False

    file_name = 'train_complete.jsonl'
    train  = load_ds_from_file(file_name, False)

    file_name = 'dev_complete.jsonl'
    valid = load_ds_from_file(file_name, False)


    file_name = 'test_complete.jsonl'
    test = load_ds_from_file(file_name, false)

    tokenizer : AutoTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model : BertModel = BertModel.from_pretrained("bert-base-uncased").cuda()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    linear = torch.rand(768,1).cuda()
    linear.requires_grad = True
#    Add code to fine-tune and test your MCQA classifier.
    dataset_size = len(train[::8])
    epochs = 20
    epoch_acc = []
    savename = "q1"
    verbose = False
    print("zero shot acc: ", test_model(model=model,tokenizer=tokenizer, dataset=test, linear_layer=linear, verbose=False))
    for epoch in range(epochs):
        model.train()
        train_model(model=model,tokenizer=tokenizer, dataset=train, optimizer=optimizer, epoch=epoch, linear=linear)
        model.eval()
        epoch_acc.append(test_model(model=model,tokenizer=tokenizer, dataset=test, linear_layer=linear, verbose=False))

        if savename:
            torch.save(model.state_dict(), Path(os.path.abspath(__file__)).parent / f"{time.strftime("%H-%M-%S")}{savename}")
    print("epoch acc: ", epoch_acc)


if __name__ == "__main__":
    main()
