import numpy as np
from collections import defaultdict

TAGS = ['B', 'M', 'E', 'S']


def load_train(path, max_length, shuffle=True, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as fin:
        lines = [row.strip() for row in fin]

    vocab = set(' '.join(lines)) - {' '}
    char2idx = defaultdict(int, {c: i + 1 for i, c in enumerate(vocab)})  # index 0 reserved for padding
    tag2idx = defaultdict(int, {t: i + 1 for i, t in enumerate(TAGS)})  # index 0 reserved for padding
    char2idx['<pad>'], tag2idx['P'] = 0, 0

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

    # converting to indices and padding
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

    print('\t{:d} sentences loaded for training'.format(train_x.shape[0]))
    return train_x, train_y, char2idx, tag2idx


def load_test(path, max_length, char2idx, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as fin:
        lines = [row.strip() for row in fin]
    test_x = np.zeros((len(lines), max_length), dtype=np.int64)
    for i, line in enumerate(lines):
        test_x[i][:len(line)] = [char2idx[c] for c in line[:max_length]]
    print('\t{:d} sentences loaded for testing'.format(test_x.shape[0]))
    return test_x


def segment(x, y, char2idx, tag2idx):
    idx2char = {i: c for c, i in char2idx.items()}
    idx2tag = {i: t for t, i in tag2idx.items()}
    sents = []
    for xs, ys in zip(x, y):
        sent = ''.join([idx2char[idx] for idx in xs if idx != 0])
        tag = ''.join([idx2tag[idx] for idx in ys[:len(sent)]])
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
