import numpy as np
import argparse
from collections import Counter
from utils import *


NUM_SENTS = 3
VAL_TYPES = ['full', 'avg', '1', '2', '3']


def evaluate(y, pred, metrics):
    """
    y: ndarray of shape (batch_size, max_length)
    pred: ndarray of shape (batch_size, max_length)
    metrics: list

    returns: list
    """
    res = np.zeros((5, len(metrics)), dtype=np.float64)
    res[0] = [eval_bleu(y, pred, m) for m in metrics]
    res[2] = [eval_bleu(y[:, :7], pred[:, :7], m) for m in metrics]
    res[3] = [eval_bleu(y[:, 7:14], pred[:, 7:14], m) for m in metrics]
    res[4] = [eval_bleu(y[:, 14:21], pred[:, 14:21], m) for m in metrics]
    res[1] = res[2:].mean(0)
    return res.reshape(-1).tolist()


def sample(vocab, freq):
    T = freq['total']
    x = np.random.randint(T)
    for i, c in enumerate(vocab):
        x -= freq[i]
        if x < 0:
            return i


def generate(X, n, vocab, freq):
    return np.array([sample(vocab, freq) for _ in range(X.shape[0] * n)]).reshape(-1, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='./data/train.txt', help='training data')
    parser.add_argument('--dev', default='./data/val.txt', help='validation data')
    parser.add_argument('--test', default='./data/test.txt', help='testing data')
    parser.add_argument('--vocab', default='./data/vocab.txt', help='vocabulary')
    parser.add_argument('--export', default='./export/', help='export directory')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--val_metric', default=['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4'], nargs='+', choices=['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4'], help='validation metric')

    args = parser.parse_args()

    np.random.seed(args.seed)

    vocab = load_vocab(args.vocab)
    train_x, train_y = load_train(args.train, vocab, random_state=args.seed)
    dev_x, dev_y = load_train(args.dev, vocab, shuffle=False)
    test_x = load_test(args.test, vocab)

    char2idx = {c: i for i, c in enumerate(vocab)}
    n = train_y.shape[-1]
    freq = Counter(train_x.reshape(-1)) + Counter(train_y.reshape(-1))
    for i, c in freq.most_common(10):
        print(vocab[i], c)
    freq['total'] = sum(freq.values())

    print('Configuration:')
    print('\n'.join('\t{:20} {}'.format(k + ':', str(v)) for k, v in sorted(dict(vars(args)).items())))
    export_config(args, os.path.join(args.export, 'config.json'))

    print()
    print('Evaluation:')
    train_pred = generate(train_x[:1000], n, vocab, freq)
    dev_pred = generate(dev_x, n, vocab, freq)
    test_pred = generate(test_x, n, vocab, freq)

    num_val = len(args.val_metric)
    header = []
    for s in ('val', 'train'):
        for t in VAL_TYPES:
            for metric in args.val_metric:
                header.append('_'.join([s, t, metric]))
    with open(os.path.join(args.export, 'log.csv'), 'w') as fout:
        fout.write(','.join(header) + '\n')

    dev_bleu = evaluate(dev_y, dev_pred, args.val_metric)
    train_bleu = evaluate(train_y[:1000], train_pred, args.val_metric)

    with open(os.path.join(args.export, 'log.csv'), 'a') as fout:
        fout.write((','.join(['{:f}'] * num_val * len(VAL_TYPES) * 2) + '\n').format(*dev_bleu, *train_bleu))

    print()
    print(('\t[Train] ' + ' {}: {:.4f} ' * num_val).format(*[t for l in zip(args.val_metric, train_bleu[:num_val]) for t in l]))
    print(('\t[Val]   ' + ' {}: {:.4f} ' * num_val).format(*[t for l in zip(args.val_metric, dev_bleu[:num_val]) for t in l]))

    test_sents = [[vocab[i] for i in row] for row in test_pred]
    test_sents = [' '.join(row[:7] + ['.'] + row[7:14] + [','] + row[14:] + ['.']) for row in test_sents]
    export(test_sents, os.path.join(args.export, 'generated_poem.txt'))
    print('\texperiment exported to directory {}'.format(args.export))
    print()

if __name__ == '__main__':
    main()
