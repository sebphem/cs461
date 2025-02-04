class bengio(torch.nn.Module):
    def __init__(self, dim=50, window=3, batchsize = 1, vocab_size=33279, activation=torch.tanh):
        super().__init__()
        
        # specify weights, activation functions and any 'helper' function needed for the neural net

    def forward(self, x):
        # perform a forward pass (inference) on a batch of concatenated word embeddings
        # hint: it may be more efficiwnt to pass a matrix of indices for the context, and
        # perform a look-up and concatenation of the word embeddings on the GPU.
        return x

def train(model,opt):
    # implement code to split you corpus into batches, use a sliding window to construct contexts over
    # your batches (sub-corpora), you can manually replicate the functionality of datafeeder() to present 
    # training examples to you model, you can manually calculate the probability assigned to the target 
    # token using torch matrix operations (note: a mask to isolate the target woord in the numerator may help),
    # calculate the negative average ln(prob) over the batch and perform gradient descent.  you may want to loop
    # over the number of epochs internal to this function or externally.  it is helpful to report training
    # perplexity, percent complete and training speed as words-per-second.  it is also prudent to save
    # you model after every epoch.
    #
    # inputs to your neural network can be either word embeddings or word look-up indices

    if opt.savename:
        torch.save(model.state_dict(), opt.savename + '/model_weights')
    return

def test_model(model, opt, epoch):
    # functionality for this function is similar to train() except that you construct examples for the
    # test or validation corpus; and you do not appy gradient descent.
    return