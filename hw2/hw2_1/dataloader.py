import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import glob
from tqdm import tqdm
#import spacy
import string
import time
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
#from scipy.special import expit
import random
import torch
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_caption(label_json_path):
    f = open(label_json_path)
    data = json.load(f)
    filename=label_json_path[:-14]+'_newcaption.txt'
    if os.path.exists(filename):
        #print(filename+" file already exists")
        return filename
    else:
        with open(filename, 'a') as fb:
            fb.write('videoID'+';'+'Caption'+'\n')
            for i in range(len(data)):
                for j in range(len(data[i]['caption'])):
                    fb.write(data[i]['id']+';'+data[i]['caption'][j]+'\n')
        return filename
    
def process(captions):
    rem_punct = str.maketrans('', '', string.punctuation)
    for i in range(len(captions)):
      line = captions[i]
      line = line.split()

      # converting to lowercase
      line = [word.lower() for word in line]

      # remove punctuation from each caption and hanging letters
      line = [word.translate(rem_punct) for word in line]

      # remove numeric values
      line = [word for word in line if word.isalpha()]

      captions[i] = ' '.join(line)
    return captions



def numerize(caption, wtoi):
    return [wtoi[word] if word in wtoi else wtoi['<UNK>'] for word in caption.split()]


class Vocabulary:
    def __init__(self, captions, freq_threshold=3):
        self.captions = process(captions)
        self.captions = captions
        self.itow = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.wtoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
    
    def __len__(self):
        return len(self.itow)
        
    def build_vocab(self,):       
        #captions = captions.tolist()
        vocab = {}
        idx=4
        for sentence in self.captions:
            for word in sentence.split():
                if word not in vocab:
                    vocab[word] = 1

                else:
                    vocab[word]+=1

                if vocab[word] == self.freq_threshold:
                    self.itow[idx] = word
                    self.wtoi[word] = idx
                    idx+=1

        #return self.vocab, self.itow, self.wtoi
        



class VideoDataset(Dataset):
    def __init__(self, feat_dir, label_json_path):
        #self.caption_file = create_caption(label_json_path)
        #self.df = pd.read_csv(self.caption_file, sep=';')
        self.df = pd.json_normalize(json.load(open(label_json_path)), meta=['id'], record_path=['caption'])
        self.df.columns = ['caption', 'id']
        self.feat_dir = feat_dir
        self.captions = self.df['caption']
        #self.captions = process(self.captions)
        self.img_feat = self.df['id']
        self.vocab = Vocabulary(self.captions)
        self.vocab.build_vocab()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        caption = self.captions[index]
        #print(caption)
        feat = self.img_feat[index]
        
        numericalized_caption = [self.vocab.wtoi["<SOS>"]]
        numericalized_caption += numerize(caption,self.vocab.wtoi)
        numericalized_caption.append(self.vocab.wtoi["<EOS>"])
        
        return feat, torch.Tensor(np.load(self.feat_dir+feat+'.npy')), torch.tensor(numericalized_caption)


class TestDataset(Dataset):
    def __init__(self, feat_dir):
        self.feat_dir = feat_dir
        self.features = []
        files = os.listdir(feat_dir)
        for file in files:
            self.features.append(file)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        idx =  self.features[index][:-4]
        feat = self.features[index]

        return idx, torch.Tensor(np.load(self.feat_dir+feat))

    
class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self, batch):
        ids = [item[0] for item in batch]
        feats = [item[1].unsqueeze(0) for item in batch]
        feats = torch.cat(feats, dim=0)
        targets = [item[2] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return ids, feats, torch.transpose(targets,0,1)
    