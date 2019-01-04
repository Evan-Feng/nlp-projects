import torch
import argparse
import os
import json
import torch
from train import *
from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--c', default='export/model.ckpt', help='checkpoint file')
    parser.add_argument('--vocab', default='data/vocab.txt', help='vocabulary')
    parser.add_argument('--test', default='data/test.txt', help='test set')
    parser.add_argument('--savepath', default='export/prediction.txt', help='export file')
    args = parser.parse_args()

    vocab = load_vocab(args.vocab)

    ckpt_dic = torch.load(args.c)
    test_x = load_test(args.test, vocab)

    model = ckpt_dic['model_state_dict']
    if next(model.parameters()).is_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model.eval()
    test_y = []
    with torch.no_grad():
        for i, xs in enumerate(test_x):
            xs = np.array(xs).reshape((-1, 1))
            _, path = model(torch.tensor(xs).to(device))
            path = path.cpu().numpy().reshape((-1,))
            test_y.append(path)
    test_sents = segment(test_x, test_y, vocab)
    export(test_sents, args.savepath)
    print('\tsegmented sentences exported to {}'.format(args.savepath))
    print()


if __name__ == '__main__':
    main()
