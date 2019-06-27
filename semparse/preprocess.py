########################################################################
#   preprocess.py - preprocess and binarize data                       #
#   author: fengyanlin@pku.edu.cn                                      #
########################################################################

import torch
import re
import argparse
import copy
from vocab import Vocab
from utils import *


BOS_TOK = '<BOS>'
EOS_TOK = '<EOS>'
UNK_TOK = '<UNK>'
PAD_TOK = '<PAD>'
ENT_TOK = '<ENT>'
ENT_TOK_SPACED = ' <ENT> '
NTE_TOK = '<NTE>'  # non-terminal token
EXTRA_TOKENS = [BOS_TOK, EOS_TOK, UNK_TOK, PAD_TOK, ENT_TOK]  # no need to include ENT_TOK since it's already in the questions


def unique(array):
    res = []
    seen = set()
    for x in array:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


def mask_out_entities(data_list, is_test=False):
    data_list = copy.deepcopy(data_list)
    ent_indexes = [] if not is_test else None
    for t in data_list:
        ents = [e[0] for e in t['parameters']]

        if not is_test:
            ei = [(t['logical_form'].find(' ' + e + ' '), i) for i, e in enumerate(ents)]
            ei = sorted(ei)
            ei = [ej[1] for ej in ei]
            ent_indexes.append(ei)

        for e in ents:
            t['question'] = t['question'].replace(' ' + e.replace('_', ' ') + ' ', ENT_TOK_SPACED)
            if not is_test:
                t['logical_form'] = t['logical_form'].replace(' ' + e + ' ', ENT_TOK_SPACED)
    return data_list, ent_indexes


def to_indexes(sents, vocab):
    res = []
    for sent in sents:
        curr = []
        for w in sent.split(' '):
            if w in vocab:
                curr.append(vocab.stoi[w])
            else:
                curr.append(vocab.stoi[UNK_TOK])
        res.append(curr)
    return res


def convert(data_list, vocabs=None, qvocab_cutoff=None, is_test=False):
    data_list, ent_indexes = mask_out_entities(data_list, is_test)
    quest = [t['question'] for t in data_list]
    if not is_test:
        rels = [[x[0] for x in re.findall(r'\s((mso|r-mso).*?)\s', t['logical_form'])] for t in data_list]
        logic = [t['logical_form'] for t in data_list]
    ents = [[e[0] for e in t['parameters']] for t in data_list]
    if vocabs is not None:
        qvocab, lvocab, rvocab = vocabs
    else:
        qvocab = Vocab(quest).frequency_cutoff(qvocab_cutoff).add_words(EXTRA_TOKENS)
        lvocab = Vocab(logic).add_words(EXTRA_TOKENS)
        rvocab = Vocab([' '.join(r) for r in rels]).add_words(EXTRA_TOKENS)
    quest = to_indexes(quest, qvocab)
    if not is_test:
        logic = to_indexes(logic, lvocab)
        rels = to_indexes([' '.join(r) for r in rels], rvocab)
    dic = {
        'q': quest,
        'l': None if is_test else logic,
        'r': None if is_test else rels,
        'ei': ent_indexes,
        'e': ents,
        'qv': qvocab,
        'lv': lvocab,
        'rv': rvocab,
    }
    return dic


def split_and_convert(data_list, vocabs=None, qvocab_cutoff=None, is_test=False):
    ds_sgl = [t for t in data_list if t['question_type'] == 'single-relation']
    ds_cvt = [t for t in data_list if t['question_type'] == 'cvt']
    ds_mix = data_list
    if vocabs is not None:
        return [convert(ds_sgl, vocabs[0], is_test=is_test),
                convert(ds_cvt, vocabs[1], is_test=is_test),
                convert(ds_mix, vocabs[2], is_test=is_test)]
    else:
        return [convert(ds_sgl, qvocab_cutoff=qvocab_cutoff, is_test=is_test),
                convert(ds_cvt, qvocab_cutoff=qvocab_cutoff, is_test=is_test),
                convert(ds_mix, qvocab_cutoff=qvocab_cutoff, is_test=is_test)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='data/EMNLP.train', help='training data')
    parser.add_argument('--dev', default='data/EMNLP.dev', help='development data')
    parser.add_argument('--test', default='data/EMNLP.test', help='test data')
    parser.add_argument('--emb', default='data/wiki.en.vec', help='pretrained word embeddings')
    parser.add_argument('--output_dir', default='data/', help='output directory')
    parser.add_argument('--vocab_cutoff', type=int, default=5, help='frequency based vocab cutoff')
    parser.add_argument('--verbose', type=bool_flag, default=True, help='test data')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    print('Loading data...')
    train = load_data(args.train)
    dev = load_data(args.dev)
    test = load_data(args.test)

    print('Saving text data...')
    for ds_type in ['sgl', 'cvt', 'mix']:
        for part in ['train', 'dev', 'test']:
            data = locals()[part]
            if ds_type == 'sgl':
                data = [t for t in data if t['question_type'] == 'single-relation']
            elif ds_type == 'cvt':
                data = [t for t in data if t['question_type'] == 'cvt']
            dest = os.path.join(args.output_dir, f'{ds_type}.{part}')
            write_data(data, dest)

    print('Binarizing data...')
    sgl_train, cvt_train, mix_train = split_and_convert(train, qvocab_cutoff=args.vocab_cutoff)
    vocabs = [[ds['qv'], ds['lv'], ds['rv']] for ds in [sgl_train, cvt_train, mix_train]]
    sgl_dev, cvt_dev, mix_dev = split_and_convert(dev, vocabs)
    sgl_test, cvt_test, mix_test = split_and_convert(test, vocabs, is_test=True)

    print('Saving binarized data...')
    for ds_type in ['sgl', 'cvt', 'mix']:
        for part in ['train', 'dev', 'test']:
            dest = os.path.join(args.output_dir, f'{ds_type}_{part}.pth')
            torch.save(locals()[f'{ds_type}_{part}'], dest)

    print('Loading word vectors...')
    oov_rates = []
    for ds_type in ['sgl', 'cvt', 'mix']:
        v = locals()[f'{ds_type}_train']['qv']
        x, cnt = load_vectors_with_vocab(args.emb, v)
        dest = os.path.join(args.output_dir, f'{ds_type}_emb.pth')
        torch.save(x, dest)
        oov_rates.append(1 - cnt / len(v))

    if args.verbose:
        print()
        print('Statistics:')
        print('\tsgl_train size = {}'.format(len(sgl_train['q'])))
        print('\tsgl_dev size   = {}'.format(len(sgl_dev['q'])))
        print('\tsgl_test size  = {}'.format(len(sgl_test['q'])))
        print('\tcvt_train size = {}'.format(len(cvt_train['q'])))
        print('\tcvt_dev size   = {}'.format(len(cvt_dev['q'])))
        print('\tcvt_test size  = {}'.format(len(cvt_test['q'])))
        print('\tmix_train size = {}'.format(len(mix_train['q'])))
        print('\tmix_dev size   = {}'.format(len(mix_dev['q'])))
        print('\tmix_test size  = {}'.format(len(mix_test['q'])))
        print()
        print('\tsgl_oov_rate   = {:2f}'.format(oov_rates[0]))
        print('\tcvt_oov_rate   = {:2f}'.format(oov_rates[1]))
        print('\tmix_oov_rate   = {:2f}'.format(oov_rates[2]))
        print()


if __name__ == '__main__':
    main()
