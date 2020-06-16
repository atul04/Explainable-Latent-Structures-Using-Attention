# @Author: Atul Sahay <atul>
# @Date:   2020-06-09T13:57:57+05:30
# @Email:  atulsahay01@gmail.com
# @Last modified by:   atul
# @Last modified time: 2020-06-09T14:05:18+05:30



import argparse
import pickle

from snli.utils.dataset import SNLIDataset, SelQADataset
from utils.vocab import Vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--vocab-size', required=True, type=int)
    parser.add_argument('--max-length', required=True, type=int)
    parser.add_argument('--lower', default=False, action='store_true')
    parser.add_argument('--out', required=True)
    parser.add_argument('--multi-task',default=False,action='store_true')
    args = parser.parse_args()



    word_vocab = Vocab.from_file(path=args.vocab, add_pad=True, add_unk=True,
    max_size=args.vocab_size)
    if args.multi_task:
        print("Creating Auxillary Dataset")
        label_dict = {0:0,1:1}
        label_vocab = Vocab(vocab_dict=label_dict, add_pad=False, add_unk=False)
        data_reader = SelQADataset(
        data_path=args.data,word_vocab=word_vocab,label_vocab=label_vocab,
        max_length=args.max_length,lower=args.lower)
        with open(args.out, 'wb') as f:
            pickle.dump(data_reader, f)
    else:
        label_dict = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        label_vocab = Vocab(vocab_dict=label_dict, add_pad=False, add_unk=False)
        data_reader = SNLIDataset(
        data_path=args.data, word_vocab=word_vocab, label_vocab=label_vocab,
        max_length=args.max_length, lower=args.lower)
        with open(args.out, 'wb') as f:
            pickle.dump(data_reader, f)


if __name__ == '__main__':
    main()

