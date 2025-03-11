import torch
import torch.nn as nn
import torch.nn.functional as F

class bengio(torch.nn.Module):
    def __init__(self, dim=50, window=3, batchsize = 1, vocab_size=33279, activation=nn.Tanh()):
        super().__init__()
        self.activation = activation
        self.loss = torch.nn.CrossEntropyLoss()

        # from pardo's class
        # layers = [
        #     nn.Embedding(vocab_size,dim),
        #     nn.Flatten(start_dim=1),
        #     nn.Linear(window*dim,window),
        #     self.activation,
        #     nn.Linear(window,vocab_size)]

        # layers = [
        #     nn.Embedding(vocab_size,dim),
        #     nn.Flatten(start_dim=1),
        #     nn.Linear(window*dim,vocab_size),
        #     self.activation,
        #     nn.Linear(vocab_size,vocab_size),
        #     nn.Softmax(dim = 1)]

        layers = [
            nn.Embedding(vocab_size,dim),
            nn.Flatten(start_dim=1),
            nn.Linear(window*dim,window*dim*2),
            self.activation,
            nn.Linear(window*dim*2,vocab_size)]

        self.net = nn.Sequential(*layers)

        #from torch docs
        for idx, m in enumerate(self.net.modules()):
            print(idx, '->', m)



    def forward(self, x:torch.Tensor):
                # perform a forward pass (inference) on a batch of concatenated word embeddings
        # hint: it may be more efficiwnt to pass a matrix of indices for the context, and
        # perform a look-up and concatenation of the word embeddings on the GPU.
        
        # # debug
        # for layer in self.net:
        #     print('shape of input')
        #     print(x.shape)
        #     # print(x)
        #     x = layer(x)
        # return x
        return self.net(x)