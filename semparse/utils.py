########################################################################
#   utils.py - utility functions for data loading and model training   #
#   author: fengyanlin@pku.edu.cn                                      #
########################################################################

import torch
import numpy as np
import json
import os
import argparse


def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def export_config(config, path):
    param_dict = dict(vars(config))
    check_path(path)
    with open(path, 'w') as fout:
        json.dump(param_dict, fout, indent=4)


def data2dict(data):
    """
    data: list[str]
    returns: dict
    """
    dic = {
        'question': data[0].split('\t')[1],
        'logical_form': None if '\t' not in data[1] else data[1].split('\t')[1],
        'parameters': data[2].split('\t')[1],
        'question_type': data[3].split('\t')[1],
    }
    entities = dic['parameters'].split(' ||| ')
    entities = [e.split(' ') for e in entities]
    entities = [[e[0], e[1], list(map(int, e[2][1:-1].split(',')))] for e in entities]
    dic['parameters'] = entities
    return dic


def load_data(filepath):
    """
    filepath: str
    returns: list[dict]
    """
    data_list = []
    data = []
    with open(filepath, 'r') as fin:
        for line in fin:
            if line.strip().startswith('='):
                data_list.append(data2dict(data))
                data = []
            else:
                data.append(line.strip())
    return data_list


def write_data(data_list, filepath):
    with open(filepath, 'w') as fout:
        for i, t in enumerate(data_list):
            parameters = ' ||| '.join(['{} {} [{},{}]'.format(e[0], e[1], e[2][0], e[2][1]) for e in t['parameters']])
            fout.write('<question id={}>\t{}\n'.format(i + 1, t['question']))
            fout.write('<logical form id={}>\t{}\n'.format(i + 1, t['logical_form']))
            fout.write('<parameters id={}>\t{}\n'.format(i + 1, parameters))
            fout.write('<question type id={}>\t{}\n'.format(i + 1, t['question_type']))
            fout.write('==================================================\n')


def to_device(data, cuda):
    if isinstance(data, (list, tuple)):
        return [to_device(t, cuda) for t in data]
    else:
        if data is not None:
            return data.cuda() if cuda else data.cpu()
        else:
            return None


def pad_sequences(seqs, pad_value):
    lengths = torch.tensor([len(s) for s in seqs])
    paded = np.full((len(seqs), lengths.max()), pad_value)
    for i in range(len(seqs)):
        paded[i, :lengths[i]] = seqs[i]
    paded = torch.tensor(paded)
    return paded, lengths
