import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import json
import os
import logging
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


class BiRNNGenerator(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        self.target_length = config.target_length
        self.dropout_rate = config.dropout
        self.rnn_cell = config.rnn_cell

        self.device = torch.device(config.device)

        self.char_emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.encoder_lstm = getattr(nn, self.rnn_cell)(self.emb_dim, self.hidden_dim, num_layers=self.num_layers,
                                                       dropout=self.dropout_rate, bidirectional=True)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.decoder_lstm = getattr(nn, self.rnn_cell)(self.emb_dim, self.hidden_dim * 2, dropout=self.dropout_rate, num_layers=self.num_layers,)
        self.hidden2char = nn.Linear(self.hidden_dim * 2, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def _reshape_hidden(self, hidden):
        if self.rnn_cell == 'LSTM':
            reshaped = []
            for h in hidden:
                h = h.view(self.num_layers, 2, -1, self.hidden_dim)
                h = torch.cat((h[:, 0, :], h[:, 1, :]), -1)
                reshaped.append(h)
        else:
            h = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
            reshaped = torch.cat((h[:, 0, :], h[:, 1, :]), -1)
        return reshaped

    def forward(self, sentences, targets=None):
        """
        sentences: 3D tensor of shape (max_length, batch_size)
        """
        bs = sentences.size()[1]
        embed = self.char_emb(sentences)  # shape (max_length, batch_size, emb_dim)
        output, hidden = self.encoder_lstm(embed)  # output of shape (max_length, batch_size, hidden_dim*2)
        # hidden = [h.view(self.num_layers, 2, -1, self.hidden_dim).mean(1) for h in hidden]
        hidden = self._reshape_hidden(hidden)

        outputs = []
        prev_chars = sentences[-1]
        for t in range(self.target_length):
            if t >= 1 and targets is not None:
                prev_chars = targets[t - 1]

            char_embs = self.dropout(self.char_emb(prev_chars))
            t_input = char_embs.unsqueeze(0)  # 3D tensor
            t_output, hidden = self.decoder_lstm(t_input, hidden)
            t_output = self.hidden2char(t_output.view(bs, -1))
            _, prev_chars = t_output.max(-1)
            prev_chars = prev_chars.detach()
            outputs.append(t_output)

        outputs = torch.stack(outputs, 0)
        outputs = self.softmax(outputs)
        return outputs


def export_checkpoint(epoch, model, optimizer, loss, best_dev_bleu, vocab, savepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model,
        'optimizer_state_dict': optimizer,
        'loss': loss,
        'best_dev_bleu': best_dev_bleu,
        'vocab': vocab,
    }
    torch.save(checkpoint, savepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='./data/train.txt', help='training data')
    parser.add_argument('--dev', default='./data/val.txt', help='validation data')
    parser.add_argument('--test', default='./data/test.txt', help='testing data')
    parser.add_argument('--vocab', default='./data/vocab.txt', help='vocabulary')
    parser.add_argument('--export', default='./export/', help='export directory')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--cpu', action='store_true', help='use cpu only')

    parser.add_argument('--emb_dim', type=int, default=128, help='dimension of the charecter embedding')
    parser.add_argument('--hidden_dim', type=int, default=100, help='number of hidden units')
    parser.add_argument('--num_layers', type=int, default=3, help='number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('-bi', '--bidirectional', type=bool, default=True, help='use bidirectional lstm')
    parser.add_argument('--rnn_cell', choices=['GRU', 'LSTM'], default='GRU', help='type of rnn cell')

    parser.add_argument('--num_epochs', type=int, default=3000, help='number of epochs to run')
    parser.add_argument('-bs', '--batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='learning rate decay factor')
    parser.add_argument('--lr_min', type=float, default=0.01, help='minimum learning rate')
    parser.add_argument('-opt', '--optimizer', default='Adam', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping by norm')
    parser.add_argument('--teacher_forcing', type=bool, default=True, help='use teacher forcing during training')
    parser.add_argument('--regularization', type=float, default=1e-5, help='l2 regularization strength')

    parser.add_argument('--val_metric', default=['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4'], nargs='+', choices=['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4'], help='validation metric')
    parser.add_argument('--val_step', type=int, default=80, help='perform validation every n iterations')

    args = parser.parse_args()

    # set random seed
    torch.manual_seed(args.seed)

    # load train/dev/test data
    vocab = load_vocab(args.vocab)
    train_x, train_y = load_train(args.train, vocab, random_state=args.seed)
    dev_x, dev_y = load_train(args.dev, vocab, shuffle=False)
    test_x = load_test(args.test, vocab)
    args.vocab_size = len(vocab)
    args.target_length = train_y.shape[-1]

    # training
    args.device = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    device = torch.device(args.device)
    print('Configuration:')
    print('\n'.join('\t{:20} {}'.format(k + ':', str(v)) for k, v in sorted(dict(vars(args)).items())))
    export_config(args, os.path.join(args.export, 'config.json'))
    model = BiRNNGenerator(args).to(device)

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    criterion = nn.NLLLoss()

    # training
    num_val = len(args.val_metric)
    header = ['iteration', 'loss']
    for s in ('val', 'train'):
        for t in VAL_TYPES:
            for metric in args.val_metric:
                header.append('_'.join([s, t, metric]))
    with open(os.path.join(args.export, 'log.csv'), 'w') as fout:
        fout.write(','.join(header) + '\n')

    print()
    print('Training:')
    print('\tstart training on device {}'.format(device))
    model.train()
    lr = args.learning_rate
    best_dev_bleu = -1.
    niter = 0
    for epoch in range(args.num_epochs):
        for i in range(0, train_x.shape[0], args.batch_size):
            niter += 1
            model.zero_grad()
            j = min(i + args.batch_size, train_x.shape[0])
            xs = torch.tensor(train_x[i:j].T).to(device)
            ys = torch.tensor(train_y[i:j].T).to(device)
            if args.teacher_forcing:
                scores = model(xs, ys)
            else:
                scores = model(xs)
            loss = criterion(scores.view(-1, args.vocab_size), ys.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            print('\r\tepoch: {:<3d}  loss: {:.4f}'.format(epoch, loss), end='')

            if niter % args.val_step == 0:
                model.eval()
                with torch.no_grad():
                    dev_pred = model(torch.tensor(dev_x.T).to(device))
                    dev_pred = dev_pred.cpu().numpy().argmax(-1).T
                    train_pred = model(torch.tensor(train_x[:1000].T).to(device))
                    train_pred = train_pred.cpu().numpy().argmax(-1).T
                model.train()
                # dev_bleu = eval_bleu(dev_y, dev_pred, args.val_metric)
                # train_bleu = eval_bleu(train_y[:1000], train_pred, args.val_metric)
                dev_bleu = evaluate(dev_y, dev_pred, args.val_metric)
                train_bleu = evaluate(train_y[:1000], train_pred, args.val_metric)

                with open(os.path.join(args.export, 'log.csv'), 'a') as fout:
                    fout.write(('{:d},{:f}' + ',{:f}' * num_val * len(VAL_TYPES) * 2 + '\n').format(niter, loss, *dev_bleu, *train_bleu))
                    # fout.flush()

                print()
                print(('\t[Train] ' + ' {}: {:.4f} ' * num_val).format(*[t for l in zip(args.val_metric, train_bleu[:num_val]) for t in l]))
                print(('\t[Val]   ' + ' {}: {:.4f} ' * num_val).format(*[t for l in zip(args.val_metric, dev_bleu[:num_val]) for t in l]))

                # print('     [Val] train_bleu: {:.4f}   val_bleu: {:.4f}'.format(train_bleu, dev_bleu))
                export_checkpoint(epoch, model, optimizer, loss, best_dev_bleu, vocab, os.path.join(args.export, 'final_state.ckpt'))
                if dev_bleu[0] >= best_dev_bleu:
                    export_checkpoint(epoch, model, optimizer, loss, best_dev_bleu, vocab, os.path.join(args.export, 'model.ckpt'))
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
    checkpoint = torch.load(os.path.join(args.export, 'model.ckpt'))
    model = checkpoint['model_state_dict']
    model.eval()

    print('\tgenearting poems')
    with torch.no_grad():
        test_pred = model(torch.tensor(test_x.T).to(device))
        test_pred = test_pred.cpu().numpy().argmax(-1).T
    test_sents = [[vocab[i] for i in row] for row in test_pred]
    test_sents = [' '.join(row[:7] + ['.'] + row[7:14] + [','] + row[14:] + ['.']) for row in test_sents]
    if args.export != '':
        export(test_sents, os.path.join(args.export, 'generated_poem.txt'))
        print('\texperiment exported to directory {}'.format(args.export))
    print()

if __name__ == '__main__':
    main()
