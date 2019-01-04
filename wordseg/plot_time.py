import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import sqrt
import time
import argparse
import json
import pickle
import os
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from train import BatchLSTMCRFTagger
from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='./data/train.txt', help='training data')
    parser.add_argument('--test', default='./data/test.txt', help='testing data')
    parser.add_argument('--vocab', default='./data/vocab.txt', help='vocabulary')
    parser.add_argument('--pretrained', help='pretrained embedding')
    parser.add_argument('--export', default='./export/', help='export directory')
    parser.add_argument('--seed', type=float, default=0, help='random seed')
    parser.add_argument('--cpu', action='store_true', help='use cpu only')
    parser.add_argument('--output', default='./export/batch_time.pdf', help='output path')

    parser.add_argument('--emb_dim', type=int, default=128, help='dimensionality of the charecter embedding')
    parser.add_argument('--hidden_dim', type=int, default=100, help='number of hidden units')
    parser.add_argument('--num_layers', type=int, default=3, help='number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--max_length', type=int, default=64, help='maximum sequence length (shorter padded, longer truncated)')
    parser.add_argument('-bi', '--bidirectional', type=bool, default=True, help='use bidirectional lstm')
    parser.add_argument('--window_size', type=int, default=3, help='window size')
    parser.add_argument('--num_attention', type=int, default=0, help='number of self-attention heads')
    parser.add_argument('--regularization', type=float, default=1e-6, help='l2-regularization strength')

    parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs to run')
    parser.add_argument('-bs', '--batch_size', type=int, nargs='+', default=[100, 200, 300, 400, 500, 600, 700], help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='learning rate decay factor')
    parser.add_argument('--lr_min', type=float, default=0.01, help='minimum learning rate')
    parser.add_argument('-opt', '--optimizer', default='Adam', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping by norm')

    parser.add_argument('--val_disable', action='store_true', help='no validation')
    parser.add_argument('--val_fraction', type=float, default=0.1, help='ndev / ntrain')
    parser.add_argument('--val_step', type=int, default=1000, help='perform validation every n iterations')

    args = parser.parse_args()

    print('Configuration:')
    print('\n'.join('\t{:15} {}'.format(k + ':', str(v)) for k, v in sorted(dict(vars(args)).items())))
    export_config(args, os.path.join(args.export, 'config.json'))

    # set random seed
    torch.manual_seed(args.seed)

    # load training/testing data
    vocab = load_vocab(args.vocab)
    train_x, train_y = load_train(args.train, vocab, args.max_length)
    test_x = load_test(args.test, vocab)
    args.target_size = len(TAGS)
    args.vocab_size = len(vocab)

    # training
    print()
    print('Testing:')
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.device = device

    lr = args.learning_rate
    res = []
    for bs in args.batch_size:
        model = BatchLSTMCRFTagger(args).to(device)
        model.train()

        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.regularization)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.regularization)

        ans = []
        for _ in range(10):
            t0 = time.time()
            model.zero_grad()
            xs = torch.tensor(train_x[:bs].T).to(device)
            ys = torch.tensor(train_y[:bs].T).to(device)
            loss = model.loss(xs, ys)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            ans.append(time.time() - t0)

        res.append(sum(ans) / len(ans))
        print('\tBatch Size: {}  Time: {}'.format(bs, sum(ans) / len(ans)))

    fig, ax = plt.subplots()
    ax.plot(args.batch_size, res, color='b', marker='D')
    ax.set_xlim(min(args.batch_size), max(args.batch_size))
    ax.set_ylim(0, max(res) + 0.01)
    ax.set_xticks(args.batch_size)
    ax.set_xlabel('Batch Size', fontsize=15)
    ax.set_ylabel('Time per Batch (seconds)', fontsize=15)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)

    fig.savefig(args.output, format='pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()
