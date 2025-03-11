from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import torch.optim as optim

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
    linear = torch.rand(768,2).cuda()
#    Add code to fine-tune and test your MCQA classifier.
    dataset_size = len(train)
    epochs = 1

    verbose = False
    with tqdm(total=dataset_size, desc='Training Progress') as progress:
        for epoch in range(epochs):
            for i in range(dataset_size):
                sample = train[i]
                texts = [x[0] for x in sample]
                # labels = [i+1 for i,x in enumerate(sample) if x[1] == 1]
                labels = [x[1] for x in sample]
                encoded_inputs = tokenizer(
                    list(texts),
                    padding=True,
                    truncation=True,
                    return_tensors="pt").to(device="cuda:0")
                labels =  torch.Tensor(labels).to(dtype=torch.long).cuda()
                labels.max()
                preds = model(**encoded_inputs)
                pred_tensor : torch.Tensor = preds.last_hidden_state
                outputs = pred_tensor[:,0,:] @ linear
                loss = loss_fn(outputs,labels)
                loss.backward()
                optimizer.step()

                progress.set_description(f'Epoch: {epoch}')
                progress.update(1)
                progress.set_postfix({'loss':loss.item()})
                if verbose:
                    print('tokens: ', encoded_inputs)
                    print('labels: ', labels)
                    print('outputs: ', outputs)
                    print('outputs size: ', outputs.size())
                    print('labels: ',      labels)
                    print('labels size: ', labels.size())

    # for sample in test:
    #     sample = train[i]
    #     if verbose:
    #         print('sample: ', sample)
    #         print("sample text: ", sample[0])
    #         print('tokens: ', tokens)
    #         print('preds: ',preds.last_hidden_state[:,0,:])
    #         print('preds view: ', preds.last_hidden_state[:,0,:].size())
    #         print('linear view: ', linear.size())
    #         print('outputs size: ', outputs.size())
    #         print("outputs: ", outputs)
    #         print('label: ', label)
    #         print('label view: ', label.size())
    #     tokens =  tokenizer(sample[0][0], return_tensors="pt")
    #     preds = model(**tokens)
    #     pred_tensor : torch.Tensor = preds.last_hidden_state
    #     outputs = pred_tensor[:,0,:] @ linear
    #     label = torch.Tensor([sample[0][1]]).to(dtype=torch.int64)

if __name__ == "__main__":
    main()
