import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import json
import pickle
import os
import logging
from utils import *


class BatchLSTMCRFTagger(nn.Module):

    def __init__(self, config):
        super(BatchLSTMCRFTagger, self).__init__()
        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.bidirectional = config.bidirectional
        self.vocab_size = config.vocab_size
        self.max_length = config.max_length
        self.target_size = config.target_size
        self.num_layers = config.num_layers
        self.dropout_rate = config.dropout
        self.device = config.device
        self.linear_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim

        self.emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim,
                            num_layers=self.num_layers, bidirectional=self.bidirectional, dropout=self.dropout_rate)
        self.hidden2tag = nn.Linear(self.linear_dim, self.target_size)

        self.transitions = nn.Parameter(torch.zeros(self.target_size, self.target_size))
        self.start_trans = nn.Parameter(torch.zeros(self.target_size,))
        self.stop_trans = nn.Parameter(torch.zeros(self.target_size,))

    def forward(self, sentences):
        feats = self._get_feats(sentences)
        scores, paths = self._batch_viterbi_decode(feats)
        return scores, paths

    def _log_sum_exp(self, X):
        """
        compute torch.log(torch.sum(torch.exp(x), 1)) in a stable way
        X: 3D tensor
        """
        xmax, _ = X.max(-1)
        X = X - xmax.unsqueeze(-1)
        return xmax + torch.log(torch.sum(torch.exp(X), -1))

    def _score_batch(self, feats, tags):
        """
        feats: 3D tensor of shape (max_length, batch_size, target_size)
        tags: 2D tensor of shape (max_length, batch_size)

        returns: 1D tensor of shape (batch_size,)
        """
        dim0, dim1 = feats.size()[:-1]
        r0 = torch.tensor(range(dim0)).to(self.device)
        r1 = torch.tensor(range(dim1)).to(self.device)
        r0, r1 = torch.meshgrid(r0, r1)
        return feats[r0, r1, tags].sum(0) + self.transitions[tags[1:], tags[:-1]].sum(0) + \
            self.start_trans[tags[0]] + self.stop_trans[tags[-1]]

    def _get_feats(self, sentences):
        """
        sentences: 2D tensor of shape (max_length, batch_size)
        """
        embed = self.emb(sentences)
        embed = self.dropout(embed)
        lstm_out, _ = self.lstm(embed)
        feats = self.hidden2tag(lstm_out.view(-1, self.linear_dim))
        return feats.view(len(sentences), -1, self.target_size)

    def _get_batch_Z(self, feats):
        """
        feats: tensor of shape (max_length, batch_size, target_size)
        """
        dp = feats[0] + self.start_trans  # shape (batch_size, target_size)
        for feat in feats[1:]:
            R = self.transitions + dp.unsqueeze(1) + feat.unsqueeze(-1)  # shape (batch_size, target_size, target_size)
            dp = self._log_sum_exp(R)
        Z = self._log_sum_exp(dp + self.stop_trans)
        return Z

    def _batch_viterbi_decode(self, feats):
        """
        feats: tensor of shape (max_length, batch_size, target_size)
        """
        bp = []
        dp = self.start_trans + feats[0]  # shape (batch_size, target_size)
        for feat in feats[1:]:
            R = self.transitions + dp.unsqueeze(1) + feat.unsqueeze(-1)
            Rmax, Rargmax = R.max(-1)
            dp = Rmax
            bp.append(Rargmax)
        dp = dp + self.stop_trans
        scores, last_tags = dp.max(-1)  # shape (batch_size,)
        batch_paths = torch.empty(feats.size()[:-1], dtype=torch.int64)
        batch_paths[-1] = last_tags
        r = torch.tensor(range(feats.size()[1])).to(self.device)  # batch indices
        for i in range(len(feats) - 2, -1, -1):
            batch_paths[i] = bp[i][r, batch_paths[i + 1]]
        return scores, batch_paths

    def loss(self, sentence, tags):
        feats = self._get_feats(sentence)
        Z = self._get_batch_Z(feats)
        gold_score = self._score_batch(feats, tags)
        return (Z - gold_score).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='./data/train.txt', help='training data')
    parser.add_argument('--test', default='./data/test.txt', help='testing data')
    parser.add_argument('--vocab', default='./data/vocab.txt', help='vocabulary')
    parser.add_argument('--pretrained', help='pretrained embedding')
    parser.add_argument('--export', default='./export/', help='export directory')
    parser.add_argument('--seed', type=float, default=0, help='random seed')
    parser.add_argument('--cpu', action='store_true', help='use cpu only')

    parser.add_argument('--emb_dim', type=int, default=128, help='dimensionality of the charecter embedding')
    parser.add_argument('--hidden_dim', type=int, default=100, help='number of hidden units')
    parser.add_argument('--num_layers', type=int, default=3, help='number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--max_length', type=int, default=64, help='maximum sequence length (shorter padded, longer truncated)')
    parser.add_argument('-bi', '--bidirectional', type=bool, default=True, help='use bidirectional lstm')

    parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs to run')
    parser.add_argument('-bs', '--batch_size', type=int, default=500, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='learning rate decay factor')
    parser.add_argument('--lr_min', type=float, default=0.01, help='minimum learning rate')
    parser.add_argument('-opt', '--optimizer', default='Adam', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping by norm')

    parser.add_argument('--val_disable', action='store_true', help='no validation')
    parser.add_argument('--val_fraction', type=float, default=0.1, help='ndev / ntrain')
    parser.add_argument('--val_step', type=int, default=1000, help='perform validation every n iterations')

    args = parser.parse_args()

    print('Configuration:')
    print('\n'.join('\t{:15} {}'.format(k + ':', str(v)) for k, v in sorted(dict(vars(args)).items())))
    export_config(args, os.path.join(args.export, 'config.json'))

    # set random seed
    torch.manual_seed(args.seed)

    print()
    print('Training:')

    # load training/testing data
    vocab = load_vocab(args.vocab)
    train_x, train_y = load_train(args.train, vocab, args.max_length)
    test_x = load_test(args.test, vocab)
    args.target_size = len(TAGS)
    args.vocab_size = len(vocab)

    # partition train set into train and dev
    num_val = int(len(train_x) * args.val_fraction)
    dev_x, dev_y = train_x[-num_val:], train_y[-num_val:]
    if not args.val_disable:
        train_x, train_y = train_x[:-num_val], train_y[:-num_val]
    dev_gold = segment(dev_x, dev_y, vocab)

    # training
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.device = device
    model = BatchLSTMCRFTagger(args).to(device)

    if args.pretrained is not None:
        with open(args.pretrained, 'rb') as fin:
            emb = pickle.load(fin)
        assert emb.shape[1] == args.emb_dim
        model.emb.weight.data.copy_(torch.from_numpy(emb))

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print('\tstart training on device {}'.format(device))
    model.train()
    lr = args.learning_rate
    best_dev_f = 0.
    niter = 0
    for epoch in range(args.num_epochs):
        for i in range(0, train_x.shape[0], args.batch_size):
            niter += 1
            model.zero_grad()
            j = min(i + args.batch_size, train_x.shape[0])
            xs = torch.tensor(train_x[i:j].T).to(device)
            ys = torch.tensor(train_y[i:j].T).to(device)
            loss = model.loss(xs, ys)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            print('\r\tepoch: {:<3d}  lr: {:<5g}  loss: {:.4f}'.format(epoch, lr, loss), end='')

            if niter % args.val_step == 0 or j == train_x.shape[0]:
                model.eval()
                with torch.no_grad():
                    dev_pred = np.zeros(dev_y.shape, dtype=int)
                    for i in range(0, dev_x.shape[0], args.batch_size):
                        j = min(dev_x.shape[0], i + args.batch_size)
                        _, paths = model(torch.tensor(dev_x[i:j].T).to(device))
                        dev_pred[i:j] = paths.cpu().numpy().T
                    dev_pred = segment(dev_x, dev_pred, vocab)
                    dev_f = eval_fscore(dev_gold, dev_pred)
                model.train()
                print('\t[Validation] val_fscore: {:.4f}'.format(dev_f))
                if dev_f > best_dev_f:
                    best_dev_f = dev_f
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model,
                        'optimizer_state_dict': optimizer,
                        'loss': loss,
                        'best_dev_f': best_dev_f,
                    }
                    torch.save(checkpoint, os.path.join(args.export, 'model.ckpt'))
                else:
                    # adjust lerning rate if dev_f decreases
                    if args.optimizer == 'SGD':
                        lr = min(args.lr_min, lr * args.lr_decay)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

    # evaluation
    print()
    print('Evaluation:')
    print('\tLoading best model')
    if not args.val_disable:
        checkpoint = torch.load(os.path.join(args.export, 'model.ckpt'))
        model = checkpoint['model_state_dict']
    model.eval()

    print('\tsegmenting test sentences')
    test_y = []
    with torch.no_grad():
        for i, xs in enumerate(test_x):
            xs = np.array(xs).reshape((-1, 1))
            _, path = model(torch.tensor(xs).to(device))
            path = path.cpu().numpy().reshape((-1,))
            test_y.append(path)
    test_sents = segment(test_x, test_y, vocab)
    if args.export != '':
        export(test_sents, os.path.join(args.export, 'prediction.txt'))
        print('\texperiment exported to directory {}'.format(args.export))
    print()

if __name__ == '__main__':
    main()
