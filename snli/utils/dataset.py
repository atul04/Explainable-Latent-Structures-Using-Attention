# @Author: Atul Sahay <atul>
# @Date:   2020-06-09T13:57:57+05:30
# @Email:  atulsahay01@gmail.com
# @Last modified by:   atul
# @Last modified time: 2020-06-09T14:17:24+05:30



import jsonlines
import torch
from nltk import word_tokenize
from torch.utils.data import Dataset
import json
import h5py
import pickle
import numpy as np


class SNLIDataset(Dataset):

    def __init__(self, data_path, word_vocab, label_vocab, max_length, lower):
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lower = lower
        self._max_length = max_length
        self._data = []
        with jsonlines.open(data_path, 'r') as reader:
            for obj in reader:
                converted = self._convert_obj(obj)
                if converted:
                    self._data.append(converted)

    def _convert_obj(self, obj):
        pre_sentence = obj['sentence1']
        hyp_sentence = obj['sentence2']
        if self.lower:
            pre_sentence = pre_sentence.lower()
            hyp_sentence = hyp_sentence.lower()
        pre_words = word_tokenize(pre_sentence)
        hyp_words = word_tokenize(hyp_sentence)
        pre = [self.word_vocab.word_to_id(w) for w in pre_words]
        pre = pre[:self._max_length]
        hyp = [self.word_vocab.word_to_id(w) for w in hyp_words]
        hyp = hyp[:self._max_length]
        pre_length = len(pre)
        hyp_length = len(hyp)
        label = obj['gold_label']
        #print(label)
        if(label not in ['neutral','entailment','contradiction']):
            #print(label)
            return None
        label = self.label_vocab.word_to_id(label)
        return pre, hyp, pre_length, hyp_length, label

    def _pad_sentence(self, data):
        max_length = max(len(d) for d in data)
        padded = [d + [self.word_vocab.pad_id] * (max_length - len(d))
                  for d in data]
        return padded

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def collate(self, batch):
        (pre_batch, hyp_batch,
         pre_length_batch, hyp_length_batch, label_batch) = list(zip(*batch))
        pre_batch = self._pad_sentence(pre_batch)
        hyp_batch = self._pad_sentence(hyp_batch)
        pre = torch.LongTensor(pre_batch)
        hyp = torch.LongTensor(hyp_batch)
        pre_length = torch.LongTensor(pre_length_batch)
        hyp_length = torch.LongTensor(hyp_length_batch)
        label = torch.LongTensor(label_batch)
        return {'pre': pre, 'hyp': hyp,
                'pre_length': pre_length, 'hyp_length': hyp_length,
                'label': label}

class SelQADataset(Dataset):
    #label_size = 2

    def __init__(self, data_path, word_vocab, label_vocab, max_length, lower):
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lower = lower
        self._max_length = max_length
        self._data = []
        print(data_path)
        print(lower)
        data = self.load_data(data_path,lower)
        for row in data:
            converted = self._convert_obj(row)
            if converted:
                self._data.append(converted)

    def _convert_obj(self, obj):
        pre_sentence = " ".join(obj['question'])
        hyp_sentence = " ".join(obj['sentence'])
        if self.lower:
            pre_sentence = pre_sentence.lower()
            hyp_sentence = hyp_sentence.lower()
        pre_words = word_tokenize(pre_sentence)
        hyp_words = word_tokenize(hyp_sentence)
        pre = [self.word_vocab.word_to_id(w) for w in pre_words]
        pre = pre[:self._max_length]
        hyp = [self.word_vocab.word_to_id(w) for w in hyp_words]
        hyp = hyp[:self._max_length]
        pre_length = len(pre)
        hyp_length = len(hyp)
        label = obj['label']
        #print(label)
        #if(label not in ['0','entailment','contradiction']):
            #print(label)
        #    return None
        label = self.label_vocab.word_to_id(label)
        return pre, hyp, pre_length, hyp_length, label


    def _pad_sentence(self, data):
        max_length = max(len(d) for d in data)
        padded = [d + [self.word_vocab.pad_id] * (max_length - len(d))
                  for d in data]
        return padded

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def collate(self, batch):
        (pre_batch, hyp_batch,
         pre_length_batch, hyp_length_batch, label_batch) = list(zip(*batch))
        pre_batch = self._pad_sentence(pre_batch)
        hyp_batch = self._pad_sentence(hyp_batch)
        pre = torch.LongTensor(pre_batch)
        hyp = torch.LongTensor(hyp_batch)
        pre_length = torch.LongTensor(pre_length_batch)
        hyp_length = torch.LongTensor(hyp_length_batch)
        label = torch.LongTensor(label_batch)
        return {'pre': pre, 'hyp': hyp,
                'pre_length': pre_length, 'hyp_length': hyp_length,
                'label': label}

    def load_data(self,file_path, lower):
        data = []
        with open(file_path) as f:
            file = json.load(f)
            for e in file:
                new_e = {}
                if not isinstance(e['question'],str):
                    continue
                new_e["question"] = [(word.lower() if lower else word)  for word in e['question'].split()]
                for i,c in enumerate(e['candidates'],start = 0):
                    if i in e['answers']:
                        new_e["label"] = 1
                    else:
                        new_e["label"] = 0
                    new_e["sentence"] = [(word.lower() if lower else word)  for word in c.split()]
                    data.append(new_e)
                    # print(new_e)
        return data

