import numpy as np
import json
import os
from collections import defaultdict

TAGS = ['P', 'B', 'M', 'E', 'S']


def load_vocab(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as fin:
        vocab = [row.strip() for row in fin]
    return vocab


def load_train(path, vocab, max_length=None, shuffle=True, encoding='utf-8', seed=0):
    np.random.seed(seed)

    with open(path, 'r', encoding=encoding) as fin:
        lines = [row.strip() for row in fin if len(row.strip()) > 0]

    char2idx = defaultdict(int, {c: i for i, c in enumerate(vocab)})
    # char2idx = defaultdict(int, {c: i + 1 for i, c in enumerate(vocab)})  # index 0 reserved for padding
    tag2idx = defaultdict(int, {t: i for i, t in enumerate(TAGS)})  # index 0 reserved for padding

    train_x, train_y = [], []
    for i, line in enumerate(lines):
        chars, tags = [], []
        for seg in line.split():
            if len(seg) == 1:
                chars.append(seg)
                tags.append('S')
            else:
                chars += list(seg)
                tags += list('B' + 'M' * (len(seg) - 2) + 'E')
        train_x.append(chars)
        train_y.append(tags)

    # converting to indices and (optional) padding
    if max_length is None:
        for i, (xs, ys) in enumerate(zip(train_x, train_y)):
            train_x[i] = [char2idx[c] for c in xs[:max_length]]
            train_y[i] = [tag2idx[t] for t in ys[:max_length]]
        if shuffle:
            perm = np.random.permutation(len(train_x))
            train_x = [train_x[i] for i in perm]
            train_y = [train_y[i] for i in perm]
    else:
        for i, (xs, ys) in enumerate(zip(train_x, train_y)):
            train_x[i] = [char2idx[c] for c in xs[:max_length]]
            train_y[i] = [tag2idx[t] for t in ys[:max_length]]
            train_x[i] += [0] * (max_length - len(train_x[i]))
            train_y[i] += [0] * (max_length - len(train_y[i]))
        train_x = np.array(train_x, dtype=np.int64)  # shape (num_sample, max_length)
        train_y = np.array(train_y, dtype=np.int64)  # shape (num_sample, max_length)
        if shuffle:
            perm = np.random.permutation(train_x.shape[0])
            train_x, train_y = train_x[perm], train_y[perm]

    print('\t{:d} sentences loaded for training'.format(len(train_x)))
    return train_x, train_y


def load_test(path, vocab, max_length=None, encoding='utf-8'):
    char2idx = defaultdict(int, {c: i for i, c in enumerate(vocab)})

    with open(path, 'r', encoding=encoding) as fin:
        lines = [row.strip() for row in fin]

    if max_length is None:
        test_x = []
        for line in lines:
            test_x.append([char2idx[c] for c in line])
    else:
        test_x = np.zeros((len(lines), max_length), dtype=np.int64)
        for i, line in enumerate(lines):
            test_x[i][:len(line)] = [char2idx[c] for c in line[:max_length]]

    print('\t{:d} sentences loaded for testing'.format(len(test_x)))
    return test_x


def segment(x, y, vocab):
    # idx2char = {i: c for c, i in char2idx.items()}
    # idx2tag = {i: t for t, i in tag2idx.items()}
    sents = []
    for xs, ys in zip(x, y):
        sent = ''.join([vocab[idx] for idx in xs if idx != 0])
        tag = ''.join([TAGS[idx] for idx in ys[:len(sent)]])
        words = []
        p = 0
        while p < len(tag):
            if tag[p] == 'B':
                q = tag.find('E', p) % len(tag) + 1
                words.append(sent[p:q])
                p = q
            else:
                words.append(sent[p])
                p += 1
        sents.append(' '.join(words))
    return sents


def export(sents, path, encoding='utf-8'):
    with open(path, 'w', encoding=encoding) as fout:
        for line in sents:
            fout.write(line + '\n')


def export_config(config, path):
    param_dict = dict(vars(config))
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)
    with open(path, 'w') as fout:
        json.dump(param_dict, fout)


def eval_sent(gold_sent, pred_sent):
    def s2w(sent):
        res, p = set(), 0
        for w in sent.split():
            res.add((p, len(w)))
            p += len(w)
        return res

    return len(s2w(gold_sent) & s2w(pred_sent)), len(gold_sent.split()), len(pred_sent.split())


def eval_fscore(gold_sents, pred_sents):
    common_count = 0
    gold_count = 0
    pred_count = 0
    for gs, ps in zip(gold_sents, pred_sents):
        c, g, p = eval_sent(gs, ps)
        common_count += c
        gold_count += g
        pred_count += p
    p = common_count / pred_count
    r = common_count / gold_count
    f = (2 * p * r) / (p + r + 1e-6)
    return f
