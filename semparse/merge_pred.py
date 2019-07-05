########################################################################
#   merge_pred.py - merge sgl.pred and cvt.pred                        #
#   author: fengyanlin@pku.edu.cn                                      #
########################################################################

import argparse
from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sgl', default='export/sgl.pred', help='predictions from the sgl model')
    parser.add_argument('--cvt', default='export/cvt.pred')
    parser.add_argument('--test', default='data/EMNLP.test', help='test file')
    parser.add_argument('-o', '--output', default='export/predictions.txt', help='output file')
    args = parser.parse_args()

    pred_sgl = load_data(args.sgl)
    pred_cvt = load_data(args.cvt)
    test = load_data(args.test)

    assert len(pred_sgl) + len(pred_cvt) == len(test)

    res = []
    for x in test:
        if x['question_type'] == 'single-relation':
            res.append(pred_sgl.pop(0))
        elif x['question_type'] == 'cvt':
            res.append(pred_cvt.pop(0))
        else:
            raise ValueError('Invalid question type encountered')

    write_data(res, args.output)
    print('Merged predictions saved to export/predictions.txt')


if __name__ == '__main__':
    main()
