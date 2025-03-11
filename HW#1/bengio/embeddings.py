
from pathlib import Path
from util.timer import time_it

def read_corpus(file_name:Path|str,vocab: list[str],words:dict[str:[int,int]],
                corpus:list[int],rev_words:dict[int:str],threshold:int):
    """

    Args:
        file_name (Path | str): _description_
        vocab (list[str]): list of all unique words in the dataset
        words (): lut for str word to word id and freq
        corpus (list[int]): list of all unique word IDs
        threshold (int): number of times to appear before not getting unk'd

    Returns:
        [vocab,words,corpus]: _description_
    """


    # corpus = [wid, wid, wid]
    # vocab = [str, str, str]
    # words[str] = [word ID, frequency]


    # total number of unique words in the dataset
    wID = len(vocab)
    if threshold > -1:
        with open(file_name,'r', encoding="utf-8") as f:
            for line in f:
                line = line.replace('\n','')
                tokens = line.split(' ')
                for t in tokens:
                    try:
                        elem = words[t]
                    except:
                        elem = [wID,0]
                        vocab.append(t)
                        wID = wID + 1
                    elem[1] = elem[1] + 1
                    words[t] = elem

        # pass through and remove any words that are to uncommon
        temp = words
        words = {}
        vocab = []
        wID = 0
        words['<unk>'] = [wID,100]
        vocab.append('<unk>')
        for word, (_,freq) in temp.items():
            if freq >= threshold:
                vocab.append(word)
                wID = wID + 1
                words[word] = [wID,freq]
    
    with open(file_name,'rt', encoding="utf-8") as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(' ')
            for t in tokens:
                try:
                    wID = words[t][0]
                except:
                    wID = words['<unk>'][0]
                corpus.append(wID)
                rev_words[wID] = t

    return [vocab,words,corpus,rev_words]

def encode_example_sentences(f_path:Path, vocab:list[str], words:dict[str:[int,int]], n_ex:int=-1):
    examples = []
    with open(f_path,'rt', encoding='utf-8') as f:
        for i,line in enumerate(f):
            line = line.replace('\n','')
            encoded = encode(line,words)
            text = ' '.join([f"{vocab[encoded[i]]}" for i in range(len(encoded))])

            examples.append(encoded)
            # print('original: %s' % line)
            # print('encoded:  %s' % text)
            # print(' ')
    return examples


def encode(text:str, words:dict[str:[int,int]]):
    encoded = []
    tokens = text.split(' ')
    for i in range(len(tokens)):
        try:
            wID = words[tokens[i]][0]
        except:
            wID = words['<unk>'][0]
        encoded.append(wID)
    return encoded

def decode(text:list[int], rev_words:dict[int:str]):
    decoded = ' '.join([rev_words[text[i]] for i in range(len(text))])
    return decoded