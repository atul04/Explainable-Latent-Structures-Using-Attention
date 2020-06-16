import argparse

import jsonlines
from nltk import word_tokenize
import pandas as pd
import json
import h5py
import pickle

def collect_words(path, lower):
    word_set = set()
    with jsonlines.open(path, 'r') as reader:
        for obj in reader:
            for key in ['sentence1', 'sentence2']:
                sentence = obj[key]
                if lower:
                    sentence = sentence.lower()
                words = word_tokenize(sentence)
                word_set.update(words)
    return word_set
def collect_words_from_list(data):
	word_set = set()
	for row in data:
		sentence1 = " ".join(row['question'])
		words = word_tokenize(sentence1)
		word_set.update(words)
		sentence2 = " ".join(row['sentence'])
		words = word_tokenize(sentence2)
		word_set.update(words)
	return word_set
	
def load_data(file_path, lower):
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

def save_vocab(word_set, path):
    with open(path, 'w', encoding='utf-8') as f:
        for word in word_set:
            f.write(f'{word}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-paths', required=True)
    parser.add_argument('--lower', default=False, action='store_true')
    parser.add_argument('--out', required=True)
    parser.add_argument('--multitask',default=False,action='store_true')
    parser.add_argument('--aux-data-paths', required=False)
    args = parser.parse_args()

    data_paths = args.data_paths.split(':')
    data_paths = [p for p in data_paths if p.strip()]
    if args.multitask:
    	aux_paths = args.aux_data_paths.split(':')
    	aux_paths = [p for p in aux_paths if p.strip()]
    word_set = set()
    for data_path in data_paths:
    	print(data_path)
    	word_set = word_set | collect_words(path=data_path, lower=args.lower)
    if args.multitask:
    	for data_path in aux_paths:
    		print(data_path)
    		data = load_data(data_path, args.lower)
    		word_set = word_set | collect_words_from_list(data)
    save_vocab(word_set=word_set, path=args.out)


if __name__ == '__main__':
    main()
