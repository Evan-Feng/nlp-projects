import torch
import argparse
import os
import json
from train import PoemGenerater
from rnn_gen import BiRNNGenerator
from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--c', default='export/model.ckpt', help='checkpoint file')
    # parser.add_argument('--model', choices=['lstm', 'train'], default='train', help='model')
    parser.add_argument('--test', default='data/test.txt', help='test set')
    parser.add_argument('--savepath', default='export/generated_poem.txt', help='export file')
    args = parser.parse_args()

    # model_path = args.checkpoint
    # config_path = os.path.join(os.path.dirname(args.checkpoint), 'config.json')

    # with open(config_path, 'r') as fin:
    #     config = AttrDict(json.load(fin))

    ckpt_dic = torch.load(args.c)
    # device = torch.device(config.device)
    vocab = ckpt_dic['vocab']
    test_x = load_test(args.test, vocab)

    # if args.model == 'train':
    #     model = PoemGenerater(config).to(device)
    # else:
    #     model = BiLSTMGenerator(config).to(device)
    # model.load_state_dict(ckpt_dic['model_state_dict'])
    model = ckpt_dic['model_state_dict']
    if next(model.parameters()).is_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model.eval()
    with torch.no_grad():
        test_pred = model(torch.tensor(test_x.T).to(device))
        test_pred = test_pred.cpu().numpy().argmax(-1).T
    test_sents = [[vocab[i] for i in row] for row in test_pred]
    test_sents = [' '.join(row[:7] + ['.'] + row[7:14] + [','] + row[14:] + ['.']) for row in test_sents]
    export(test_sents, args.savepath)
    print('generated poems exported to {}'.format(args.savepath))
    print()


if __name__ == '__main__':
    main()
