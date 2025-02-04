import torch

#data loader
class CustomBengioDataset(torch.utils.data.Dataset):
    def __init__(self, corpus: list[int], rev_words:dict[str:int], window: int, verbose=False, **kwargs):
        self.window = window
        self.corpus = corpus
        self.rev_words = rev_words
        self.verbose = verbose

    def __len__(self):
        return len(self.corpus) - (self.window+1)

    def __getitem__(self, idx):
        raw = self.corpus[idx:idx+self.window+1]
        word_id_list_tensor = torch.tensor(raw[:-1], dtype=torch.long)
        next_word_id_tensor = torch.tensor(raw[-1], dtype=torch.long)
        if self.verbose:
            print('raw ids: ', raw)
            print('translated back: ', ' '.join([self.rev_words[r] for r in raw]))
            print('word_id_list_tensor: ', word_id_list_tensor)
        return word_id_list_tensor, next_word_id_tensor