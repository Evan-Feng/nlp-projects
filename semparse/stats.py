########################################################################
#   stats.py - display dataset statistics                              #
#   author: fengyanlin@pku.edu.cn                                      #
########################################################################

import re
import argparse
from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='data/EMNLP.train', help='training data')
    parser.add_argument('--dev', default='data/EMNLP.dev', help='development data')
    parser.add_argument('--test', default='data/EMNLP.test', help='test data')
    args = parser.parse_args()

    train = load_data(args.train)
    dev = load_data(args.dev)
    test = load_data(args.test)

    train_sgl = [t for t in train if t['question_type'] == 'single-relation']
    train_cvt = [t for t in train if t['question_type'] == 'cvt']

    relations = [re.match(r'\( lambda \?x \( (.*?) .*? \?x \) \)',
                          t['logical_form']).group(1) for t in train
                 if t['question_type'] == 'single-relation']

    print('train size     = {}'.format(len(train)))
    print('train_sgl size = {}'.format(len(train_sgl)))
    print('train_cvt size = {}'.format(len(train_cvt)))
    print('num_relations  = {}'.format(len(set(relations))))
    print()
    print('train example:')
    print(train[0])
    print('test example:')
    print(test[0])


if __name__ == '__main__':
    main()
