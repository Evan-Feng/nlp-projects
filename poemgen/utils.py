from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import os
import json
from collections import defaultdict


# SOS_TOKEN = 0
# EOS_TOKEN = 1


class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_vocab(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as fin:
        vocab = [row.strip() for row in fin]
    return vocab


def load_train(path, vocab, shuffle=True, encoding='utf-8', random_state=0):
    np.random.seed(random_state)

    with open(path, 'r', encoding=encoding) as fin:
        lines = [row.strip() for row in fin]

    char2idx = defaultdict(int, {c: i for i, c in enumerate(vocab)})

    lines = [[char2idx[c] for c in line.split() if c not in set(' ,.')] for line in lines]
    train_x = np.array(lines).reshape((-1, 4, 7))
    train_y = train_x[:, 1:].reshape(-1, 21)
    train_x = train_x[:, 0]

    if shuffle:
        perm = np.random.permutation(train_x.shape[0])
        train_x, train_y = train_x[perm], train_y[perm]

    return train_x, train_y


# def load_dev(path, char2idx, encoding='utf-8'):
#     with open(path, 'r', encoding=encoding) as fin:
#         lines = [row.strip() for row in fin]
#     lines = [[char2idx[c] for c in line.split() if c not in set(' ,.')] for line in lines]
#     dev_x = np.array(lines).reshape((-1, 4, 7))
#     dev_y = dev_x[:, 1:].reshape(-1, 21)
#     dev_x = dev_x[:, 0]
#     return dev_x, dev_y


def load_test(path, vocab, encoding='utf-8'):
    char2idx = defaultdict(int, {c: i for i, c in enumerate(vocab)})

    with open(path, 'r', encoding=encoding) as fin:
        lines = [row.strip() for row in fin]

    lines = [[char2idx[c] for c in line.split() if c not in set(' ,.')] for line in lines]
    test_x = np.array(lines)
    return test_x


def export_config(config, path):
    param_dict = dict(vars(config))
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)
    with open(path, 'w') as fout:
        json.dump(param_dict, fout)


def export(sents, path, encoding='utf-8'):
    with open(path, 'w', encoding=encoding) as fout:
        for line in sents:
            fout.write(line + '\n')


def eval_bleu(gold_x, pred_x, val_metric):
    if val_metric == 'bleu-1':
        weights = [1.]
    elif val_metric == 'bleu-2':
        weights = [1. / 2.] * 2
    elif val_metric == 'bleu-3':
        weights = [1. / 3.] * 3
    elif val_metric == 'bleu-4':
        weights = [1. / 4.] * 4

    elif val_metric == '2-bleu-2':
        weights = [1. / 2.] * 2
        gold_x = gold_x[:, :7]
        pred_x = pred_x[:, :7]

    bleu = 0.
    for gx, px in zip(gold_x, pred_x):
        bleu += sentence_bleu([gx.tolist()], px.tolist(), weights)
    bleu /= len(gold_x)
    return bleu
