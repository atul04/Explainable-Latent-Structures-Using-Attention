# @Author: Atul Sahay <atul>
# @Date:   2020-06-09T13:57:57+05:30
# @Email:  atulsahay01@gmail.com
# @Last modified by:   atul
# @Last modified time: 2020-06-09T17:00:34+05:30



import argparse
import logging
import os
import pickle

import math
from tensorboardX import SummaryWriter

import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from snli.model import SNLIModel
from snli.utils.dataset import SNLIDataset, SelQADataset
from utils.glove import load_glove

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


def train(args):
    with open(args.train_data, 'rb') as f:
        train_dataset: SNLIDataset = pickle.load(f)
    with open(args.valid_data, 'rb') as f:
        valid_dataset: SNLIDataset = pickle.load(f)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2,
                              collate_fn=train_dataset.collate,
                              pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=2,
                              collate_fn=valid_dataset.collate,
                              pin_memory=True)
    word_vocab = train_dataset.word_vocab
    label_vocab = train_dataset.label_vocab
    selqa_label_vocab = None
    #print(len(label_vocab))
    #print(len(word_vocab))
    #exit()
    if args.multitask:
        with open(args.aux_train_data, 'rb') as f:
            selqa_train_dataset: SelQADataset = pickle.load(f)
        with open(args.aux_valid_data, 'rb') as f:
            selqa_valid_dataset: SelQADataset = pickle.load(f)

        selqa_train_loader = DataLoader(dataset=selqa_train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2,
                              collate_fn=selqa_train_dataset.collate,
                              pin_memory=True)
        selqa_valid_loader = DataLoader(dataset=selqa_valid_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=2,
                              collate_fn=selqa_valid_dataset.collate,
                              pin_memory=True)
        selqa_label_vocab = selqa_train_dataset.label_vocab

    model = SNLIModel(prim_num_classes=len(label_vocab), num_words=len(word_vocab),
                      word_dim=args.word_dim, hidden_dim=args.hidden_dim,
                      clf_hidden_dim=args.clf_hidden_dim,
                      clf_num_layers=args.clf_num_layers,
                      use_leaf_rnn=args.leaf_rnn,
                      use_batchnorm=args.batchnorm,
                      intra_attention=args.intra_attention,
                      dropout_prob=args.dropout,
                      bidirectional=args.bidirectional,aux_num_classes=len(selqa_label_vocab))
    for name, param in model.named_parameters():
    	if param.requires_grad:
        	print(name, param.data.shape, param.data)
    if args.glove:
        logging.info('Loading GloVe pretrained vectors...')
        glove_weight = load_glove(
            path=args.glove, vocab=word_vocab,
            init_weight=model.word_embedding.weight.data.numpy())
        glove_weight[word_vocab.pad_id] = 0
        model.word_embedding.weight.data.set_(torch.FloatTensor(glove_weight))
    if args.fix_word_embedding:
        logging.info('Will not update word embeddings')
        model.word_embedding.weight.requires_grad = False
    model.to(args.device)
    logging.info(f'Using device {args.device}')
    if args.optimizer == 'adam':
        optimizer_class = optim.Adam
    elif args.optimizer == 'adagrad':
        optimizer_class = optim.Adagrad
    elif args.optimizer == 'adadelta':
        optimizer_class = optim.Adadelta
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_class(params=params, weight_decay=args.l2reg)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    criterion = nn.CrossEntropyLoss()

    train_summary_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, 'log', 'train'))
    valid_summary_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, 'log', 'valid'))

    def run_iter(batch, is_training,Gate):
        model.train(is_training)
        pre = batch['pre'].to(args.device)
        hyp = batch['hyp'].to(args.device)
        pre_length = batch['pre_length'].to(args.device)
        hyp_length = batch['hyp_length'].to(args.device)
        #print('pre_length',pre_length)
        #print('hyp_length',hyp_length)
        label = batch['label'].to(args.device)
        logits = model(pre=pre, pre_length=pre_length,
                       hyp=hyp, hyp_length=hyp_length,Gate=Gate)
        label_pred = logits.max(1)[1]
        accuracy = torch.eq(label, label_pred).float().mean()
        loss = criterion(input=logits, target=label)
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(parameters=params, max_norm=5)
            optimizer.step()
        return loss, accuracy

    def add_scalar_summary(summary_writer, name, value, step):
        if torch.is_tensor(value):
            value = value.item()
        summary_writer.add_scalar(tag=name, scalar_value=value,
                                  global_step=step)

    num_train_batches = len(train_loader)
    validate_every = num_train_batches // 10
    best_vaild_accuacy = 0
    iter_count = 0
    switch = 0
    selqa_batch = 0
    for epoch_num in range(args.max_epoch):
        logging.info(f'Epoch {epoch_num}: start')
        dataloader_iterator = iter(selqa_train_loader)
        for batch_iter, train_batch in enumerate(train_loader):
            if(switch%8==0):
                #print("Selqa Training")
                selqa_batch+=1
                try:
                    aux_train_batch = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(selqa_train_loader)
                    aux_train_batch = next(dataloader_iterator)

                if iter_count % args.anneal_temperature_every == 0:
                    rate = args.anneal_temperature_rate
                    new_temperature = max([0.5, math.exp(-rate * iter_count)])
                    model.encoder.gumbel_temperature = new_temperature
                    logging.info(f'Iter #{iter_count}: '
                                 f'Set Gumbel temperature to {new_temperature:.4f}')
                train_loss, train_accuracy = run_iter(
                    batch=aux_train_batch, is_training=True,Gate = 1)
                iter_count += 1
                add_scalar_summary(
                    summary_writer=train_summary_writer,
                    name='loss', value=train_loss, step=iter_count)
                add_scalar_summary(
                    summary_writer=train_summary_writer,
                    name='accuracy', value=train_accuracy, step=iter_count)

                if selqa_batch%50==0:
                    selqa_batch=0
                    torch.set_grad_enabled(False)
                    valid_loss_sum = valid_accuracy_sum = 0
                    num_valid_batches = len(selqa_valid_loader)
                    for valid_batch in selqa_valid_loader:
                        valid_loss, valid_accuracy = run_iter(
                            batch=valid_batch, is_training=False,Gate=1)
                        valid_loss_sum += valid_loss.item()
                        valid_accuracy_sum += valid_accuracy.item()
                    torch.set_grad_enabled(True)
                    valid_loss = valid_loss_sum / num_valid_batches
                    valid_accuracy = valid_accuracy_sum / num_valid_batches
                    print("Selqa : valid loss {}, valid acc {}".format(valid_loss,valid_accuracy))
                    #scheduler.step(valid_accuracy)
                    #progress = epoch_num + batch_iter/num_train_batches
                    # logging.info(f'Epoch {progress:.2f}: '
                    #              f'valid loss = {valid_loss:.4f}, '
                    #              f'valid accuracy = {valid_accuracy:.4f}')
                    # if valid_accuracy > best_vaild_accuacy:
                    #     best_vaild_accuacy = valid_accuracy
                    #     model_filename = (f'model-{progress:.2f}'
                    #                       f'-{valid_loss:.4f}'
                    #                       f'-{valid_accuracy:.4f}.pkl')
                    #     model_path = os.path.join(args.save_dir, model_filename)
                    #     torch.save(model.state_dict(), model_path)
                    #     print(f'Saved the new best model to {model_path}')
                switch=0


            if iter_count % args.anneal_temperature_every == 0:
                rate = args.anneal_temperature_rate
                new_temperature = max([0.5, math.exp(-rate * iter_count)])
                model.encoder.gumbel_temperature = new_temperature
                logging.info(f'Iter #{iter_count}: '
                             f'Set Gumbel temperature to {new_temperature:.4f}')
            train_loss, train_accuracy = run_iter(
                batch=train_batch, is_training=True,Gate=0)
            iter_count += 1
            add_scalar_summary(
                summary_writer=train_summary_writer,
                name='loss', value=train_loss, step=iter_count)
            add_scalar_summary(
                summary_writer=train_summary_writer,
                name='accuracy', value=train_accuracy, step=iter_count)

            if (batch_iter + 1) % validate_every == 0:
                torch.set_grad_enabled(False)
                valid_loss_sum = valid_accuracy_sum = 0
                num_valid_batches = len(valid_loader)
                for valid_batch in valid_loader:
                    valid_loss, valid_accuracy = run_iter(
                        batch=valid_batch, is_training=False,Gate=0)
                    valid_loss_sum += valid_loss.item()
                    valid_accuracy_sum += valid_accuracy.item()
                torch.set_grad_enabled(True)
                valid_loss = valid_loss_sum / num_valid_batches
                valid_accuracy = valid_accuracy_sum / num_valid_batches
                print("valid loss {}, valid acc {}".format(valid_loss,valid_accuracy))
                scheduler.step(valid_accuracy)
                add_scalar_summary(
                    summary_writer=valid_summary_writer,
                    name='loss', value=valid_loss, step=iter_count)
                add_scalar_summary(
                    summary_writer=valid_summary_writer,
                    name='accuracy', value=valid_accuracy, step=iter_count)
                progress = epoch_num + batch_iter/num_train_batches
                logging.info(f'Epoch {progress:.2f}: '
                             f'valid loss = {valid_loss:.4f}, '
                             f'valid accuracy = {valid_accuracy:.4f}')
                if valid_accuracy > best_vaild_accuacy:
                    best_vaild_accuacy = valid_accuracy
                    model_filename = (f'model-{progress:.2f}'
                                      f'-{valid_loss:.4f}'
                                      f'-{valid_accuracy:.4f}.pkl')
                    model_path = os.path.join(args.save_dir, model_filename)
                    torch.save(model.state_dict(), model_path)
                    print(f'Saved the new best model to {model_path}')
            switch+=1
def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--train-data', required=True)
    parser.add_argument('--valid-data', required=True)
    parser.add_argument('--word-dim', required=True, type=int)
    parser.add_argument('--hidden-dim', required=True, type=int)
    parser.add_argument('--clf-hidden-dim', required=True, type=int)
    parser.add_argument('--clf-num-layers', required=True, type=int)
    parser.add_argument('--leaf-rnn', default=False, action='store_true')
    parser.add_argument('--bidirectional', default=False, action='store_true')
    parser.add_argument('--intra-attention', default=False, action='store_true')
    parser.add_argument('--batchnorm', default=False, action='store_true')
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--anneal-temperature-every', default=1e10, type=int)
    parser.add_argument('--anneal-temperature-rate', default=0, type=float)
    parser.add_argument('--glove', default=None)
    parser.add_argument('--fix-word-embedding', default=False,
                        action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', required=True, type=int)
    parser.add_argument('--max-epoch', required=True, type=int)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--l2reg', default=0, type=float)
    parser.add_argument('--multitask',default=False,action='store_true')
    parser.add_argument('--aux-train-data', required=False)
    parser.add_argument('--aux-valid-data', required=False)
    args = parser.parse_args()
    print("attention",args.intra_attention)
    print("Multitask",args.multitask)
    train(args)


if __name__ == '__main__':
    main()


'''
CUDA_LAUNCH_BLOCKING=1 python3 train.py --train-data snli/Data/train.pkl --valid-data snli/Data/val.pkl --glove snli/glove.6B.300d.txt --save-dir snli/SaveModel/IntraAttention_non_leaf/ --word-dim 300 --hidden-dim 300 --clf-hidden-dim 1024 --clf-num-layers 1 --batch-size 64 --max-epoch 20 --device cuda --intra-attention


CUDA_LAUNCH_BLOCKING=1 python3 train_multitask.py --train-data snli/Data/Multitask/snlitrain.pkl --valid-data snli/Data/Multitask/snlitest.pkl --glove snli/glove.840B.300d.txt --save-dir snli/SaveModel/Multitask_Snli_Selqa_Gumble_Intra_att_Linear_embd_non_leaf_log_leaf_rnn_batch_norm_test_840B_0.3_bidirectional_32/ --word-dim 300 --hidden-dim 300 --clf-hidden-dim 1024 --clf-num-layers 1 --batch-size 32 --max-epoch 30 --device cuda:0 --intra-attention --batchnorm --dropout 0.13 --leaf-rnn --bidirectional
'''

