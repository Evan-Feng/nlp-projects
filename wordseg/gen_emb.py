import argparse
import numpy as np
import pickle
from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb', default='./data/wiki.zh.vec', help='pretrained word embedding')
    parser.add_argument('--vocab', default='./data/vocab.txt', help='vocabulary')
    parser.add_argument('--output', default='./data/zh-300d.bin', help='output file')
    parser.add_argument('--dim', type=int, default=300, help='dimension of the vectors')
    parser.add_argument('--encoding', default='utf-8', help='encoding format')
    args = parser.parse_args()

    vocab = load_vocab(args.vocab, args.encoding)
    w2idx = {w: i for i, w in enumerate(vocab)}

    emb = np.zeros((len(vocab), args.dim), dtype=np.float64)
    count = 0
    with open(args.emb, 'r', encoding=args.encoding) as fin:
        for line in fin:
            word, vec = line.split(' ', 1)
            if word in w2idx:
                emb[w2idx[word]] = np.fromstring(vec, sep=' ', dtype=np.float64)
                count += 1

    print('OOV rate = {:.4f}'.format(1 - count / len(vocab)))

    with open(args.output, 'wb') as fout:
        pickle.dump(emb, fout)


if __name__ == '__main__':
    main()
