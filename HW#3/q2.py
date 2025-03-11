from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import torch.optim as optim

from pathlib import Path
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
        base = '[START] ' + result['fact1'] + ' [SEP] ' + result['question']['stem'] + ' [SEP] '
        ans = result['answerKey']
        for j in range(4):
            base += f" {answers[j]} {result['question']['choices'][j]['text']} "
        samples.append([base,ans])
        
        if verbose:
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
def train_model(model:GPT2LMHeadModel, tokenizer: GPT2Tokenizer, dataset: list[list[str,int]], optimizer: torch.optim.Optimizer, epoch: int, savename: str, verbose: bool = False, **kwargs):

    if verbose:
        print("model:", model)
        print("optimizer:", optimizer)
        print("savename:", savename)
        print("verbose:", verbose)
    print('inside train')
    model.train()
    loss_epoch = []
    with tqdm(total=len(dataset), desc='Training Progress') as progress:
        for i, sample in enumerate(dataset):
            text, label = sample[0], sample[1]
            encoded = tokenizer(f"{text} [ANS] {label}", return_tensors="pt").to(device="cuda:0")
            # print('encoded train: ', f"{text} [ANS] {label}")
            # print('encoded_input train: ', encoded)
            output = model(**encoded, labels=encoded["input_ids"], pad_token_id=tokenizer.pad_token_id)

            loss = output.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch.append(loss.item())

            if i%100==0:
                progress.set_description(f'Training Epoch: {epoch}')
                progress.update(100)
                progress.set_postfix({'loss':loss.item()})
            if verbose and i%200==0:
                print('sample: ', sample)
                print('text + label: ', text + label)

    loss_epoch = sum(loss_epoch) / len(loss_epoch)
    if savename:
        torch.save(model.state_dict(), Path(os.path.abspath(__file__)).parent / f"{time.strftime("%H-%M-%S")}{savename}")
    return loss_epoch

@time_it
def test_model(model:GPT2LMHeadModel, tokenizer: GPT2Tokenizer, dataset: list[list[str,int]], epoch: int, verbose: bool = False, **kwargs):
     model.eval()
     cor = 0
     letter_count = 0
     labels_encoded = tokenizer(f"A B C D")['input_ids']
     label_lut = dict(zip(['A', 'B', 'C', 'D'], labels_encoded))
    #  print('labels lut: ', label_lut)
     with tqdm(total=len(dataset), desc='Testing Progress') as progress:
        for i, sample in enumerate(dataset):
            # print('sample: ', sample)
            text, label = sample[0], sample[1]
            # print('text + label: ', text + label)
            encoded_input = tokenizer(f"{text} [ANS]", return_tensors='pt').to(device="cuda:0")
            # print('encoded test: ', f"{text} [ANS]")
            # print('encoded_input: test ', encoded_input)
            output : torch.Tensor = model.generate(**encoded_input, max_new_tokens=5, pad_token_id=tokenizer.pad_token_id)
            # print('output: ', output)
            guess_id = output[0]

            # remove whitespace
            guess = tokenizer.decode(guess_id).strip()

            # if one of the five words generated is a letter, use it
            for word in guess_id[-4:]:
                if word in labels_encoded:
                    letter_count += 1
                    if word == label_lut[label]:
                        cor += 1
            if not i%100 and i > 10:
                progress.set_description(f'Testing Epoch: {epoch}')
                progress.update(100)
                progress.set_postfix({'acc':cor/i})
     acc = cor/len(dataset)
     print('acc: ', acc)
     print('letter: ', letter_count/len(dataset))
     return acc


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

    model : GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
    tokenizer : GPT2Tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    print('eos token: ', tokenizer.eos_token)
    print('eos token')
    print('220 token decoded: ', tokenizer.decode([220]))
    
    tokenizer.pad_token = tokenizer.eos_token
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    loss_epoch = []
    acc_epoch = []
    epochs = 10
    for i in range(1,epochs+1):
        acc_epoch.append(test_model(model=model,tokenizer=tokenizer,dataset=valid, epoch=0,verbose=True))
        loss_epoch.append(train_model(model=model,tokenizer=tokenizer,dataset=train, optimizer=optimizer,epoch=0,savename='q2',verbose=False))

    print('acc_epoch: ', acc_epoch)
    print('loss_epoch: ', loss_epoch)

if __name__ == "__main__":
    main()
