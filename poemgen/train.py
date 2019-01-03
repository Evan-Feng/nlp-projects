import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
import json
import os
import logging
import pickle
from utils import *

NUM_SENTS = 3
VAL_TYPES = ['avg', '1', '2', '3']


class PoemGenerater(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.emb_dim = config.emb_dim
        self.pos_dim = config.pos_dim
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        self.dropout_rate = config.dropout
        self.attention = config.attention
        self.rnn_cell = config.rnn_cell

        self.num_sents = config.num_sents
        self.num_chars = config.num_chars

        self.device = torch.device(config.device)

        self.decoder_input_dim = self.emb_dim + self.pos_dim + self.hidden_dim * 2

        self.char_emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.pos_emb = nn.Embedding(self.num_sents * self.num_chars, self.pos_dim)
        self.encoder_lstm = getattr(nn, self.rnn_cell)(self.emb_dim, self.hidden_dim,
                                                       num_layers=self.num_layers, dropout=self.dropout_rate, bidirectional=True)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.sent_decoder = getattr(nn, self.rnn_cell)(self.hidden_dim * 4, self.hidden_dim * 2, dropout=self.dropout_rate, num_layers=self.num_layers)
        self.char_decoder = getattr(nn, self.rnn_cell)(self.decoder_input_dim, self.hidden_dim * 2, dropout=self.dropout_rate, num_layers=self.num_layers)
        self.hidden2char = nn.Linear(self.hidden_dim * 4, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

        if self.attention == 'general':
            self.atten_matrix = nn.Parameter(torch.randn(1, self.hidden_dim * 2, self.hidden_dim * 2))
        elif self.attention == 'concat':
            self.atten_layer1 = nn.Linear(self.hidden_dim * 4, self.hidden_dim)
            self.atten_layer2 = nn.Linear(self.hidden_dim, 1)

        # self.attention_layer = nn.Linear(self.hidden_dim * 3, 1)

    def _dot_attention(self, enc_h, h):
        """
        enc_h: shape (max_lenth, batch_size, hidden_dim*2)
        h: shape (batch_size, hidden_dim*2)
        """
        # enc_h = (enc_h[:, :, self.hidden_dim:] + enc_h[:, :, :self.hidden_dim]) / 2
        weights = torch.transpose(enc_h, 0, 1).bmm(h.unsqueeze(-1))
        weights = torch.transpose(weights, 0, 1)  # shape (max_length, batch_size)
        weights = F.softmax(weights, 0)
        atten = torch.sum(enc_h * weights, 0)  # shape (batch_size, hidden_dim)
        return atten

    def _general_attention(self, enc_h, h):
        """
        enc_h: shape (max_lenth, batch_size, hidden_dim*2)
        h: shape (batch_size, hidden_dim*2)
        """
        # enc_h = (enc_h[:, :, self.hidden_dim:] + enc_h[:, :, :self.hidden_dim]) / 2
        xatten = self.atten_matrix.expand(h.size()[0], self.hidden_dim * 2, self.hidden_dim * 2)
        weights = torch.transpose(enc_h, 0, 1).bmm(xatten.bmm(h.unsqueeze(-1)))
        weights = torch.transpose(weights, 0, 1)  # shape (max_length, batch_size)
        weights = F.softmax(weights, 0)
        atten = torch.sum(enc_h * weights, 0)  # shape (batch_size, hidden_dim)
        return atten

    def _concat_attention(self, enc_h, h):
        """
        enc_h: shape (max_lenth, batch_size, hidden_dim*2)
        h: shape (batch_size, hidden_dim*2)
        """
        h = h.unsqueeze(0).expand(*enc_h.size())
        h = torch.cat((enc_h, h), -1)
        weights = self.atten_layer2(F.relu(self.atten_layer1(h)))
        weights = F.softmax(weights, 0)
        atten = torch.sum(enc_h * weights, 0)  # shape (batch_size, hidden_dim)
        return atten

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

    def _encode(self, sentences):
        char_embs = self.char_emb(sentences)
        output, hidden = self.encoder_lstm(char_embs)
        hidden = self._reshape_hidden(hidden)
        if self.rnn_cell == 'LSTM':
            context = hidden[0][-1]
        else:
            context = hidden[-1]
        return output, hidden, context

    def _decode_sent(self, hidden, context, prev_sent):
        inputs = torch.cat((context, prev_sent), 1).unsqueeze(0)  # 3D
        context, hidden = self.sent_decoder(inputs, hidden)
        context = context.squeeze(0)  # (bs, hdim)
        return hidden, context

    def _decode_char(self, enc_h, hidden, context, prev_char, pos):
        bs = prev_char.size()[0]
        pos = torch.tensor(pos).unsqueeze(0).expand(bs).to(self.device)
        pos_embs = self.pos_emb(pos)
        char_embs = self.dropout(self.char_emb(prev_char))
        inputs = torch.cat((char_embs, pos_embs, context), 1).unsqueeze(0)  # 3D
        context, hidden = self.char_decoder(inputs, hidden)
        context = context.squeeze(0)

        if self.attention == 'dot':
            atten = self._dot_attention(enc_h, context)
        elif self.attention == 'general':
            atten = self._general_attention(enc_h, context)
        elif self.attention == 'concat':
            atten = self._concat_attention(enc_h, context)

        inputs = torch.cat((context, atten), 1)
        output = self.hidden2char(inputs)
        return hidden, context, output

    def forward(self, sentences, targets=None):
        """
        sentences: 2D tensor of shape (max_length, batch_size)
        targets: 2D tensor of shape (target_length, batch_size)
            if None: no teacher forcing
            otherwise: use teacher forcing
        """
        bs = sentences.size()[1]  # batch_size
        enc_out, hidden, context = self._encode(sentences)

        prev_sent = context
        prev_char = sentences[-1]
        outputs = []
        for ts in range(self.num_sents):
            hidden, char_context = self._decode_sent(hidden, context, prev_sent)
            char_hidden = hidden
            for tc in range(self.num_chars):
                pos = ts * self.num_chars + tc
                if pos >= 1 and targets is not None:  # use teacher forcing
                    prev_char = targets[pos - 1]

                char_hidden, prev_sent, output = self._decode_char(enc_out, char_hidden, char_context, prev_char, pos)
                _, prev_char = output.max(-1)
                prev_char = prev_char.detach()
                outputs.append(output)

        outputs = torch.stack(outputs, 0)
        outputs = self.softmax(outputs)
        return outputs


def export_checkpoint(epoch, model, optimizer, loss, best_dev_bleu, vocab, savepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model,  # .state_dict(),
        'optimizer_state_dict': optimizer,  # .state_dict(),
        'loss': loss,
        'best_dev_bleu': best_dev_bleu,
        'vocab': vocab,
    }
    torch.save(checkpoint, savepath)


def evaluate(y, pred, metrics):
    """
    y: ndarray of shape (batch_size, max_length)
    pred: ndarray of shape (batch_size, max_length)
    metrics: list

    returns: list
    """
    res = np.zeros((4, len(metrics)), dtype=np.float64)
    res[1] = [eval_bleu(y[:, :7], pred[:, :7], m) for m in metrics]
    res[2] = [eval_bleu(y[:, 7:14], pred[:, 7:14], m) for m in metrics]
    res[3] = [eval_bleu(y[:, 14:21], pred[:, 14:21], m) for m in metrics]
    res[0] = res[1:].mean(0)
    return res.reshape(-1).tolist()


def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='./data/train.txt', help='training data')
    parser.add_argument('--dev', default='./data/val.txt', help='validation data')
    parser.add_argument('--test', default='./data/test.txt', help='testing data')
    parser.add_argument('--vocab', default='./data/vocab.txt', help='vocabulary')
    parser.add_argument('--pretrained', help='pretrained embedding')
    parser.add_argument('--export', default='./export/', help='export directory')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--cpu', action='store_true', help='use cpu only')

    parser.add_argument('--emb_dim', type=int, default=128, help='dimension of the charecter embedding')
    parser.add_argument('--pos_dim', type=int, default=8, help='dimension of the position embedding')
    parser.add_argument('--hidden_dim', type=int, default=100, help='number of hidden units')
    parser.add_argument('--num_layers', type=int, default=3, help='number of stacked LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('-bi', '--bidirectional', type=bool_flag, nargs='?', const=True, default=True, help='use bidirectional lstm')
    parser.add_argument('--regularization', type=float, default=1e-5, help='l2 regularization strength')
    parser.add_argument('--attention', choices=['dot', 'general', 'concat'], default='general', help='attention machanism')
    parser.add_argument('--label_smoothing', type=float, default=0.2, help='label smoothing (zero to disable)')
    parser.add_argument('--rnn_cell', choices=['GRU', 'LSTM'], default='GRU', help='type of rnn cell')

    parser.add_argument('--num_epochs', type=int, default=10000, help='number of epochs to run')
    parser.add_argument('-bs', '--batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='learning rate decay factor')
    parser.add_argument('--lr_min', type=float, default=0.01, help='minimum learning rate')
    parser.add_argument('-opt', '--optimizer', default='Adam', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping by norm')
    parser.add_argument('--teacher_forcing', type=bool_flag, nargs='?', const=True, default=True, help='use teacher forcing during training')

    parser.add_argument('--val_metric', default=['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4'], nargs='+', choices=['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4'], help='validation metric')
    parser.add_argument('--val_step', type=int, default=80, help='perform validation every n iterations')

    args = parser.parse_args()

    # set random seed
    torch.manual_seed(args.seed)

    #------------------------------------------------------#
    #  load train/dev/test data                            #
    #------------------------------------------------------#
    vocab = load_vocab(args.vocab)
    train_x, train_y = load_train(args.train, vocab, random_state=args.seed)
    dev_x, dev_y = load_train(args.dev, vocab, shuffle=False)
    test_x = load_test(args.test, vocab)

    #------------------------------------------------------#
    #  configuration                                       #
    #------------------------------------------------------#
    args.vocab_size = len(vocab)
    args.num_chars = train_x.shape[-1]
    args.num_sents = train_y.shape[-1] // args.num_chars
    args.device = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"

    device = torch.device(args.device)

    print('Configuration:')
    print('\n'.join('\t{:20} {}'.format(k + ':', str(v)) for k, v in sorted(dict(vars(args)).items())))
    export_config(args, os.path.join(args.export, 'config.json'))

    #------------------------------------------------------#
    #  construct model, optimizer, criterion               #
    #------------------------------------------------------#
    model = PoemGenerater(args).to(device)
    if args.pretrained is not None:
        with open(args.pretrained, 'rb') as fin:
            emb = pickle.load(fin)
        assert emb.shape[1] == args.emb_dim
        model.char_emb.weight.data.copy_(torch.from_numpy(emb))

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.learning_rate, weight_decay=args.regularization)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=args.learning_rate, weight_decay=args.regularization)

    if args.label_smoothing > 0:
        criterion = nn.KLDivLoss(reduction='batchmean')
    else:
        criterion = nn.NLLLoss()

    #------------------------------------------------------#
    #  training                                            #
    #------------------------------------------------------#
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
    t0 = time.time()
    lr = args.learning_rate
    best_dev_bleu = -1.
    niter = 0
    # train_x = torch.tensor(train_x.T).to(device)
    # test_x = torch.tensor(test_x.T).to(device)
    one_hot_add = torch.full((args.vocab_size,), args.label_smoothing / args.vocab_size).to(device)
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

            if args.label_smoothing > 0:
                ys = ys.view(-1)
                one_hot = torch.zeros(ys.size()[0], args.vocab_size).to(device)
                one_hot = one_hot.scatter_(-1, ys.unsqueeze(-1), torch.tensor(1 - args.label_smoothing).to(device))  # shape (length, bs, vsize)
                one_hot = one_hot + one_hot_add
                # loss = (-one_hot * scores.view(-1, args.vocab_size)).sum(-1).mean()
                loss = criterion(scores.view(-1, args.vocab_size), one_hot)
            else:
                loss = criterion(scores.view(-1, args.vocab_size), ys.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            print('\r\tepoch: {:<3d}  loss: {:.4f}'.format(epoch, loss), end='')

            #------------------------------------------------------#
            #  validation                                          #
            #------------------------------------------------------#
            if niter % args.val_step == 0:
                model.eval()
                with torch.no_grad():
                    dev_pred = model(torch.tensor(dev_x.T).to(device))
                    dev_pred = dev_pred.cpu().numpy().argmax(-1).T
                    train_pred = model(torch.tensor(train_x[:1000].T).to(device))
                    train_pred = train_pred.cpu().numpy().argmax(-1).T
                model.train()
                # dev_bleu = [eval_bleu(dev_y, dev_pred, metric) for metric in args.val_metric]
                # train_bleu = [eval_bleu(train_y[:1000], train_pred, metric) for metric in args.val_metric]
                dev_bleu = evaluate(dev_y, dev_pred, args.val_metric)
                train_bleu = evaluate(train_y[:1000], train_pred, args.val_metric)

                with open(os.path.join(args.export, 'log.csv'), 'a') as fout:
                    fout.write(('{:d},{:f}' + ',{:f}' * num_val * len(VAL_TYPES) * 2 + '\n').format(niter, loss, *dev_bleu, *train_bleu))
                    # fout.flush()

                print()
                print(('\t[Train] ' + ' {}: {:.4f} ' * num_val).format(*[t for l in zip(args.val_metric, train_bleu[:num_val]) for t in l]))
                print(('\t[Val]   ' + ' {}: {:.4f} ' * num_val).format(*[t for l in zip(args.val_metric, dev_bleu[:num_val]) for t in l]))

                export_checkpoint(epoch, model, optimizer, loss, best_dev_bleu, vocab, os.path.join(args.export, 'final_state.ckpt'))
                if dev_bleu[0] >= best_dev_bleu:
                    best_dev_bleu = dev_bleu[0]
                    export_checkpoint(epoch, model, optimizer, loss, best_dev_bleu, vocab, os.path.join(args.export, 'model.ckpt'))
                else:
                    # adjust lerning rate if dev_f decreases
                    if args.optimizer == 'SGD':
                        lr = min(args.lr_min, lr * args.lr_decay)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
    print()
    print('\ttraining finished in {:d} seconds'.format(int(time.time() - t0)))
    export_checkpoint(epoch, model, optimizer, loss, best_dev_bleu, vocab, os.path.join(args.export, 'final_state.ckpt'))

    #------------------------------------------------------#
    #  evaluation                                          #
    #------------------------------------------------------#
    print()
    print('Evaluation:')
    print('\tLoading best model')
    checkpoint = torch.load(os.path.join(args.export, 'model.ckpt'))
    # model.load_state_dict(checkpoint['model_state_dict'])
    model = checkpoint['model_state_dict']
    model.eval()

    if args.export != '':
        print('\tgenerating poems')
        with torch.no_grad():
            test_pred = model(torch.tensor(test_x.T).to(device))
            test_pred = test_pred.cpu().numpy().argmax(-1).T
        test_sents = [[vocab[i] for i in row] for row in test_pred]
        test_sents = [' '.join(row[:7] + ['.'] + row[7:14] + [','] + row[14:] + ['.']) for row in test_sents]
        export(test_sents, os.path.join(args.export, 'generated_poem.txt'))
        print('\texperiment exported to directory {}'.format(args.export))
        print()

if __name__ == '__main__':
    main()
