########################################################################
#   sinlge_relation.py - training/evaluating single-relation model     #
#   author: fengyanlin@pku.edu.cn                                      #
########################################################################

import matplotlib as mpl
mpl.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import argparse
import os
import re
import time
import json
from model import CNNClassifier, PCNNClassifier
from data import DataLoader
from preprocess import UNK_TOK, PAD_TOK, ENT_TOK, EOS_TOK, ENT_TOK_SPACED
from utils import *


def print_line():
    print('-' * 80)


def evaluate(model, batch):
    acc = 0
    size = 0
    for qx, lx, rx, len_q, len_l, _ in batch:
        pred = model(qx)
        _, pred = pred.max(-1)
        acc += (pred == rx).sum().item()
        size += qx.size(0)
    return acc / size


def predict(model, qx, pad_id, batch_size):
    """
    model: nn.Module
    pad_id: int
    qx: list[list[int]]

    returns: torch.Tensor
    """
    is_cuda = next(model.parameters()).is_cuda
    pred = []
    for i in range(0, len(qx), batch_size):
        j = min(len(qx), i + batch_size)
        input_x, _ = pad_sequences(qx[i:j], pad_id)
        input_x = to_device(input_x, is_cuda)
        p = model(input_x)
        _, p = p.max(-1)
        pred.append(p)
    pred = torch.cat(pred, 0)
    return pred


def predict_with_batch(model, batch):
    """
    model: nn.Module
    batch: DataLoader

    returns: torch.Tensor
    """

    pred = []
    for qx, lx, rx, len_q, len_l, _ in batch:
        p = model(qx)
        _, p = p.max(-1)
        pred.append(p)
    pred = torch.cat(pred, 0)
    return pred


def plot_confusion_matrix(model, batch, filepath, normalize=True):
    pred = predict_with_batch(model, batch).cpu().numpy()
    y = torch.cat([b[2] for b in batch], 0).cpu().numpy()
    cm = confusion_matrix(y, pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    fig.savefig(filepath, format='pdf')


def model_save(model, opt, path):
    torch.save([model, opt], path)


def model_load(path):
    model, opt = torch.load(path)
    return model, opt


def load_config(config_dir, args):
    with open(os.path.join(config_dir, 'config.json'), 'r') as fin:
        dic = json.load(fin)

    for k in dict(vars(args)):
        if k not in ('resume', 'mode', 'cuda', 'test', 'output'):
            setattr(args, k, dic[k])
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='data/sgl_train.pth', help='training data')
    parser.add_argument('--dev', default='data/sgl_dev.pth', help='development data')
    parser.add_argument('--test', default='data/EMNLP.test', help='test data')
    parser.add_argument('--emb', default='data/sgl_emb.pth', help='pretrained word embeddings')
    parser.add_argument('--emb_mode', choices=['init', 'freeze', 'random'], default='init', help='use pretarined word embeddings')
    parser.add_argument('--output', default='data/sgl.pred', help='output file path')
    parser.add_argument('--sample_train', type=int, default=0, help='downsample training set to n examples (zero to disable)')
    parser.add_argument('--resume', help='path of model to resume')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train', help='train or evaluate')
    parser.add_argument('--model', choices=['cnn'], default='cnn', help='encoder model')

    # architecture
    parser.add_argument('--emb_size', type=int, default=300, help='size of word embeddings')
    parser.add_argument('--hidden_size', type=int, default=600, help='number of hidden units per layer of the language model')
    parser.add_argument('--nkernels', type=int, default=100, help='number of cnn kernels')
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[2, 3, 4], help='number of hidden units per layer of the language model')
    parser.add_argument('--nlayers', type=int, default=2, help='number of layers')

    # regularization
    # parser.add_argument('--dropoutc', type=float, default=0.6, help='dropout applied to classifier')
    # parser.add_argument('--dropouto', type=float, default=0.4, help='dropout applied to rnn outputs')
    parser.add_argument('--dropouth', type=float, default=0.3, help='dropout for rnn layers')
    # parser.add_argument('--dropouti', type=float, default=0.4, help='dropout for input embedding layers')
    parser.add_argument('--dropoute', type=float, default=0.1, help='dropout to remove words from embedding layer')
    # parser.add_argument('--dropoutw', type=float, default=0.5, help='weight dropout applied to the RNN hidden to hidden matrix')
    # parser.add_argument('--dropoutd', type=float, default=0.1, help='dropout applied to language discriminator')
    parser.add_argument('--wdecay', type=float, default=1e-6, help='weight decay applied to all weights')

    # optimization
    parser.add_argument('--max_steps', type=int, default=200000, help='upper step limit')
    parser.add_argument('--max_steps_before_stop', type=int, default=5000, help='stop if dev_acc does not increase for N steps')
    parser.add_argument('-bs', '--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--optimizer', type=str,  default='adam', choices=['adam', 'sgd'], help='optimizer to use (sgd, adam)')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam optimizer')
    parser.add_argument('-lr', '--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--grad_clip', type=float, default=0.25, help='gradient clipping')

    # device / logging settings
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--cuda', type=bool_flag, nargs='?', const=True, default=True, help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=50, metavar='N', help='report interval')
    parser.add_argument('--val_interval', type=int, default=200, metavar='N', help='validation interval')
    parser.add_argument('--plot', type=bool_flag, nargs='?', const=True, default=False, help='plot confusion matrix')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--export', type=str,  default='export/sgl/', help='dir to save the model')

    args = parser.parse_args()
    if args.debug:
        parser.set_defaults(log_interval=20, val_interval=40)
    args = parser.parse_args()
    if args.mode == 'eval':
        args = load_config(args.export, args)
    elif args.resume:
        args = load_config(os.path.dirname(args.resume), args)

    if args.mode == 'train':
        train(args)
    else:
        eval(args)


def train(args):
    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    print('Configuration:')
    print('\n'.join('\t{:15} {}'.format(k + ':', str(v)) for k, v in sorted(dict(vars(args)).items())))
    print()

    model_path = os.path.join(args.export, 'model.pt')
    config_path = os.path.join(args.export, 'config.json')
    cm_path = os.path.join(args.export, 'confusion_matrix.pdf')
    export_config(args, config_path)
    check_path(model_path)

    ###############################################################################
    # Load data
    ###############################################################################

    train_batch = DataLoader(args.train, args.batch_size, args.cuda)
    train_batch_eval = DataLoader(args.train, args.batch_size, args.cuda)
    dev_batch = DataLoader(args.dev, args.batch_size, args.cuda)

    ###############################################################################
    # Build the model
    ###############################################################################
    if args.resume:
        model, opt = model_load(args.resume)

    else:
        if args.model == 'cnn':
            model = CNNClassifier(len(train_batch.ds['qv']), args.emb_size, args.nkernels, args.kernel_sizes,
                                  args.dropoute, 1, args.hidden_size, len(train_batch.ds['rv']), args.dropouth)
        # elif args.model == 'pcnn':
        #     tok_ids = [train_batch.ds['qv'].stoi[ENT_TOK], train_batch.ds['qv'].stoi[EOS_TOK]]
        #     model = PCNNClassifier(len(train_batch.ds['qv']), args.emb_size, args.nkernels, args.kernel_sizes,
        #                           args.dropoute, 1, args.hidden_size, len(train_batch.ds['rv']), args.dropouth, tok_ids)
        if args.emb_mode in ('init', 'freeze'):
            emb_x = torch.load(args.emb)
            model.encoder.emb.weight.data.copy_(torch.from_numpy(emb_x))
            if args.emb_mode == 'freeze':
                freeze_net(model.encoder.emb)
        if args.optimizer == 'sgd':
            opt = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
        if args.optimizer == 'adam':
            opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay, betas=(args.beta1, 0.999))

    crit = nn.CrossEntropyLoss()
    bs = args.batch_size
    if args.cuda:
        model.cuda(), crit.cuda()
    else:
        model.cpu(), crit.cpu()

    print('Parameters:')
    total_params = sum([np.prod(x.size()) for x in model.parameters()])
    print('\ttotal params:   {}'.format(total_params))
    print('\tparam list:     {}'.format(len(list(model.parameters()))))
    for name, x in model.named_parameters():
        print('\t' + name + '\t', tuple(x.size()))
    print()

    ###############################################################################
    # Training code
    ###############################################################################

    print('Traning:')
    print_line()
    best_acc = 0
    total_loss = 0
    last_best_step = 0
    step = 0
    start_time = time.time()
    model.train()
    do_stop = False
    while True:
        for batch in train_batch:
            opt.zero_grad()

            qx, lx, rx, len_q, len_l, _ = batch
            # print(' '.join(train_batch.ds['qv'].itos[w] for w in qx[0]))
            with torch.autograd.set_detect_anomaly(True):
                loss = crit(model(qx), rx)
                loss.backward()
            opt.step()
            total_loss += loss.item()

            if (step + 1) % args.log_interval == 0:
                elapsed = time.time() - start_time
                print('| step {:5d} | lr {:05.5f} | ms/batch {:7.2f} | loss {:7.4f} |'.format(
                    step, opt.param_groups[0]['lr'], elapsed * 1000 / args.log_interval,
                    total_loss / args.log_interval))
                total_loss = 0
                start_time = time.time()

            if (step + 1) % args.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    train_acc = evaluate(model, train_batch_eval)
                    dev_acc = evaluate(model, dev_batch)
                    print_line()
                    print('| train_acc {:4f} | dev_acc {:4f} |'.format(train_acc, dev_acc))
                    if dev_acc > best_acc:
                        last_best_step = step
                        best_acc = dev_acc
                        print(f'saving model to {model_path}')
                        model_save(model, opt, model_path)
                        plot_confusion_matrix(model, dev_batch, cm_path)
                    print_line()
                model.train()
                start_time = time.time()

            step += 1
            if step >= args.max_steps or step - last_best_step >= args.max_steps_before_stop:
                do_stop = True
                break

        if do_stop:
            break
        train_batch.reshuffle()

    print_line()
    print('Training ended with {} steps'.format(step))
    print('Best val acc: {:.4f}'.format(best_acc))


def eval(args):
    model, opt = model_load(os.path.join(args.export, 'model.pt'))   # Load the best saved model.
    model.cuda() if args.cuda else model.cpu()
    model.eval()

    ds = torch.load(args.dev)
    qv = ds['qv']
    rv = ds['rv']
    data_list = load_data(args.test)
    data_list = [t for t in data_list if t['question_type'] == 'single-relation']
    quest = [t['question'] for t in data_list]
    ents = [t['parameters'][0][0] for t in data_list]

    # do_mask = lambda s, e: ('_' + s.replace(' ', '_') + '_').replace('_' + e + '_', '_' + ENT_TOK + '_', 1).replace('_', ' ').strip()

    # quest = [q.replace(' ' + e.replace('_', ' ') + ' ', ENT_TOK_SPACED) for q, e in zip(quest, ents)]
    # quest = [do_mask(q, e) for q, e in zip(quest, ents)]
    qx = [[qv.stoi[w] if w in qv else qv.stoi[UNK_TOK] for w in q.split(' ')] for q in quest]
    with torch.no_grad():
        pred = predict(model, qx, qv.stoi[PAD_TOK], args.batch_size)

    rels = [rv.itos[r] for r in pred]
    lambda_exps = ['( lambda ?x ( {} {} ?x ) )'.format(r, e) for r, e in zip(rels, ents)]
    for t, exp in zip(data_list, lambda_exps):
        t['logical_form'] = exp
    write_data(data_list, args.output)
    print('Predictions saved to {}'.format(args.output))


if __name__ == '__main__':
    main()
