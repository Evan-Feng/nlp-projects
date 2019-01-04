import argparse
from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infiles', nargs='+', help='predicted segmentations')
    parser.add_argument('--gold', default='data/test.answer.txt', help='gold segmentation')
    parser.add_argument('--encoding', default='utf-8', help='file encoding')
    args = parser.parse_args()

    with open(args.gold, 'r', encoding=args.encoding) as fin:
        gold = [sent.strip() for sent in fin]

    for file in args.infiles:
        with open(file, 'r', encoding=args.encoding) as fin:
            test = [sent.strip() for sent in fin]

        fscore = eval_fscore(gold, test)
        print('file: {}   fscore: {:.4f}'.format(file, fscore))


if __name__ == '__main__':
    main()
