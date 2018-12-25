import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import json
import os
import logging
from utils import *


class PoemGenerater(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.emb_dim = config.emb_dim
        self.pos_dim = config.pos_dim
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        self.dropout_rate = config.dropout

        self.num_sents = config.num_sents
        self.num_chars = config.num_chars

        self.device = config.device

        self.decoder_input_dim = self.emb_dim + self.pos_dim + self.hidden_dim

        self.char_emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.pos_emb = nn.Embedding(self.num_sents * self.num_chars, self.pos_dim)
        self.encoder_lstm = nn.LSTM(self.emb_dim, self.hidden_dim,
                                    num_layers=self.num_layers, dropout=self.dropout_rate, bidirectional=True)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.sent_decoder = nn.LSTM(self.hidden_dim * 2, self.hidden_dim, dropout=self.dropout_rate, num_layers=self.num_layers)
        self.char_decoder = nn.LSTM(self.decoder_input_dim, self.hidden_dim, dropout=self.dropout_rate, num_layers=self.num_layers)
        self.hidden2char = nn.Linear(self.hidden_dim, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.attention_layer = nn.Linear(self.hidden_dim * 3, 1)

    def _dot_attention(self, enc_h, h):
        """
        enc_h: shape (max_lenth, batch_size, hidden_dim*2)
        h: shape (num_layers, batch_size, hidden_dim)
        """
        enc_h = (enc_h[:, :, self.hidden_dim:] + enc_h[:, :, :self.hidden_dim]) / 2
        weights = torch.transpose(enc_h, 0, 1).bmm(h[-1].unsqueeze(-1))
        weights = torch.transpose(weights, 0, 1)  # shape (max_length, batch_size)
        weights = F.softmax(weights, 0)
        atten = torch.sum(enc_h * weights, 0)  # shape (batch_size, hidden_dim*2)
        return atten

    def _concat_attention(self, enc_h, h):
        """
        enc_h: shape (max_lenth, batch_size, hidden_dim*2)
        h: shape (num_layers, batch_size, hidden_dim)
        """
        h = h[-1].expand(list(enc_h.size()[:2]) + [self.hidden_dim])
        concated = torch.cat((enc_h, h), -1).view(-1, self.hidden_dim * 3)
        weights = self.attention_layer(concated).view(enc_h.size()[:2])
        weights = F.softmax(weights, 0)
        atten = torch.sum(enc_h * weights.unsqueeze(-1).expand(enc_h.size()), 0)
        return atten

    def forward(self, sentences):
        """
        sentences: 2D tensor of shape (max_length, batch_size)
        """
        bs = sentences.size()[1]
        embed = self.char_emb(sentences)  # shape (max_length, batch_size, emb_dim)
        enc_hidden, (h_n, c_n) = self.encoder_lstm(embed)  # output of shape (max_length, batch_size, hidden_dim*2)
        h_n = h_n.view(self.num_layers, 2, -1, self.hidden_dim).mean(1)
        c_n = c_n.view(self.num_layers, 2, -1, self.hidden_dim).mean(1)
        hidden = (h_n, c_n)  # h_n of shape (num_layers, batch_size, hidden_dim*2)
        enc_final_h = h_n[-1]  # shape (batch_size, hiddem_dim)

        outputs = []
        prev_sent_h = enc_final_h
        for ts in range(self.num_sents):
            atten = self._concat_attention(enc_hidden, hidden[0])
            ts_in = torch.cat((enc_final_h, prev_sent_h), 1)
            ts_out, hidden = self.sent_decoder(atten.unsqueeze(0), hidden)

            h = hidden
            prev_chars = torch.tensor([SOS_TOKEN] * bs).to(self.device)
            for tc in range(self.num_chars):
                char_embs = self.dropout(self.char_emb(prev_chars))  # (bs, char_dim)
                t = torch.tensor(ts * self.num_chars + tc).to(self.device)
                pos_embs = self.pos_emb(t).unsqueeze(0).expand(bs, self.pos_dim)  # (bs, pos_dim)
                atten = self._dot_attention(enc_hidden, h[0])  # (bs, hidden_dim)
                tc_input = torch.cat((char_embs, pos_embs, atten), -1).unsqueeze(0)
                tc_out, h = self.char_decoder(tc_input, h)
                tc_out = self.hidden2char(tc_out.view(bs, -1))
                outputs.append(tc_out)

        outputs = torch.stack(outputs, 0)
        outputs = self.softmax(outputs)
        return outputs

        # for t in range(self.target_length):
        #     char_embs = self.dropout(self.char_emb(prev_chars))
        #     t = torch.tensor(t).to(self.device)
        #     pos_embs = self.pos_emb(t).unsqueeze(0).expand(bs, self.pos_dim)
        #     t_input = torch.cat((char_embs, pos_embs), -1).unsqueeze(0)  # 3D tensor
        #     t_output, hidden = self.decoder_lstm(t_input, hidden)
        #     t_output = self.hidden2char(t_output.view(bs, -1))
        #     _, prev_chars = t_output.max(-1)
        #     prev_chars = prev_chars.detach()
        #     outputs.append(t_output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='./data/train.txt', help='training data')
    parser.add_argument('--dev', default='./data/val.txt', help='validation data')
    parser.add_argument('--test', default='./data/test.txt', help='testing data')
    parser.add_argument('--export', default='./export/', help='export directory')
    parser.add_argument('--seed', type=float, default=0, help='random seed')
    parser.add_argument('--cpu', action='store_true', help='use cpu only')

    parser.add_argument('--emb_dim', type=int, default=128, help='dimension of the charecter embedding')
    parser.add_argument('--pos_dim', type=int, default=8, help='dimension of the position embedding')
    parser.add_argument('--hidden_dim', type=int, default=100, help='number of hidden units')
    parser.add_argument('--num_layers', type=int, default=3, help='number of stacked LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('-bi', '--bidirectional', type=bool, default=True, help='use bidirectional lstm')

    parser.add_argument('--num_epochs', type=int, default=5000, help='number of epochs to run')
    parser.add_argument('-bs', '--batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='learning rate decay factor')
    parser.add_argument('--lr_min', type=float, default=0.01, help='minimum learning rate')
    parser.add_argument('-opt', '--optimizer', default='Adam', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping by norm')

    parser.add_argument('--val_metric', default='bleu-2', choices=['bleu-2', 'bleu-3', 'bleu-4'], help='validation metric')
    parser.add_argument('--val_step', type=int, default=1000, help='perform validation every n iterations')

    args = parser.parse_args()

    print('Configuration:')
    print('\n'.join('\t{:15} {}'.format(k + ':', str(v)) for k, v in sorted(dict(vars(args)).items())))
    export_config(args, os.path.join(args.export, 'config.json'))

    # set random seed
    torch.manual_seed(args.seed)

    print()
    print('Training:')

    # load train/dev/test data
    train_x, train_y, char2idx = load_train(args.train)
    dev_x, dev_y = load_dev(args.dev, char2idx)
    test_x = load_test(args.test, char2idx)
    args.vocab_size = len(char2idx)
    args.num_chars = train_x.shape[-1]
    args.num_sents = train_y.shape[-1] // args.num_chars

    # training
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.device = device
    model = PoemGenerater(args).to(device)

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    criterion = nn.NLLLoss()

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
            scores = model(xs)
            loss = criterion(scores.view(-1, args.vocab_size), ys.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            print('\r\tepoch: {:<3d}  loss: {:.4f}'.format(epoch, loss), end='')

            if niter % args.val_step == 0 or j == train_x.shape[0]:
                model.eval()
                with torch.no_grad():
                    dev_pred = model(torch.tensor(dev_x.T).to(device))
                    dev_pred = dev_pred.cpu().numpy().argmax(-1).T
                    train_pred = model(torch.tensor(train_x[:1000].T).to(device))
                    train_pred = train_pred.cpu().numpy().argmax(-1).T
                model.train()
                dev_bleu = eval_bleu(dev_y, dev_pred, args.val_metric)
                train_bleu = eval_bleu(train_y[:1000], train_pred, args.val_metric)

                print('    [Val] train_bleu: {:.4f}   val_bleu: {:.4f}'.format(train_bleu, dev_bleu))
                if dev_bleu >= best_dev_bleu:
                    best_dev_bleu = dev_bleu
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'best_dev_bleu': best_dev_bleu,
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
    checkpoint = torch.load(os.path.join(args.export, 'model.ckpt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print('\tgenerating poems')
    with torch.no_grad():
        test_pred = model(torch.tensor(test_x.T).to(device))
        test_pred = test_pred.cpu().numpy().argmax(-1).T
    idx2char = {i: c for c, i in char2idx.items()}
    test_sents = [[idx2char[i] for i in row] for row in test_pred]
    test_sents = [' '.join(row[:7] + ['.'] + row[7:14] + [','] + row[14:] + ['.']) for row in test_sents]
    if args.export != '':
        export(test_sents, os.path.join(args.export, 'generated_poem.txt'))
        print('\texperiment exported to directory {}'.format(args.export))
    print()

if __name__ == '__main__':
    main()
