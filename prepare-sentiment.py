#!/usr/bin/env python3

from torch.utils.data import Dataset
from collections import Counter
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer
import pandas as pd

def convert_sst_label(s):
    return s.split(" ")[-1] 

dynasent_r1 = load_dataset("dynabench/dynasent", 'dynabench.dynasent.r1.all', trust_remote_code=True)
dynasent_r2 = load_dataset("dynabench/dynasent", 'dynabench.dynasent.r2.all', trust_remote_code=True)
sst = load_dataset("SetFit/sst5", trust_remote_code=True)

labels = ['negative', 'neutral', 'positive']
class2index = dict(zip(sorted(labels), range(len(labels))))
index2class = dict(zip(range(len(labels)), sorted(labels)))

def merge_sentence_datasets(split, limit=None):
    sst_labels = [convert_sst_label(s) for s in sst[split]['label_text']]
    input_sentences = [s for sl in [dynasent_r1[split]['sentence'], sst[split]['text'], dynasent_r2[split]['sentence']] for s in sl]
    input_labels = [l for sl in [dynasent_r1[split]['gold_label'], sst_labels, dynasent_r2[split]['gold_label']] for l in sl]
    classes = [class2index[l] for l in input_labels]
    if limit is not None:
        input_sentences = input_sentences[:limit]
        input_labels = input_labels[:limit]
        classes = classes[:limit]
    return input_sentences, input_labels, torch.tensor(classes)


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, labels=None):
        self.tokenizer = tokenizer
        self.toke
        self.labels = labels if labels is not None else [0] * len(texts)
        print(self.texts)
        print(self.labels)
    
    def encode_text(self, text):
        return [self.vocab[token] for token in self.tokenizer(text)]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

(inputs, sentiment_labels, labels) = merge_sentence_datasets("train", limit=1000)

df = pd.DataFrame({'text': inputs, 'sentiment': sentiment_labels, 'label': labels})
df.to_csv('sentiment_dataset.csv', index=False)

