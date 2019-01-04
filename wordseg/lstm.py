import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
from utils import *


class LSTMTagger(nn.Module):

    def __init__(self, config):
        super(LSTMTagger, self).__init__()
        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.bidirectional = config.bidirectional
        self.vocab_size = config.vocab_size
        self.target_size = config.target_size
        self.max_length = config.max_length

        self.emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, bidirectional=self.bidirectional)
        self.in_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        self.hidden2tag = nn.Linear(self.in_dim, self.target_size)

    def forward(self, x):
        """
        x: tensor of shape (max_length, batch_size)
        """
        embs = self.emb(x)  # of shape (max_length, batch_size, emb_dim)
        lstm_out, self.hidden = self.lstm(embs)
        tag_space = self.hidden2tag(lstm_out.view(-1, self.in_dim))
        tag_scores = F.log_softmax(tag_space, dim=1).view(len(x), -1, self.target_size)
        return tag_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='./data/train.txt', help='training data')
    parser.add_argument('--test', default='./data/test.txt', help='testing data')
    parser.add_argument('--export', default='./export/', help='export directory')
    parser.add_argument('--seed', type=float, default=0, help='random seed')
    parser.add_argument('--cpu', action='store_true', help='use cpu only')

    parser.add_argument('--emb_dim', type=int, default=64, help='dimensionality of the charecter embedding')
    parser.add_argument('--hidden_dim', type=int, default=100, help='number of hidden units')
    parser.add_argument('--max_length', type=int, default=64, help='maximum sequence length (shorter padded, longer truncated)')
    parser.add_argument('-bi', '--bidirectional', type=bool, default=True, help='use bidirectional lstm')

    parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs to run')
    parser.add_argument('-bs', '--batch_size', type=int, default=50, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='learning rate decay factor')
    parser.add_argument('--lr_min', type=float, default=0.01, help='minimum learning rate')
    parser.add_argument('-opt', '--optimizer', default='Adam', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping by norm')

    parser.add_argument('--val_fraction', type=float, default=0.1, help='ndev / ntrain')
    parser.add_argument('--val_step', type=int, default=1000, help='perform validation every n iterations')

    args = parser.parse_args()

    #######################################
    args.num_epochs = int(args.num_epochs * (args.batch_size / 50) + 10)

    print('Configuration:')
    print('\n'.join('\t{:15} {}'.format(k + ':', str(v)) for k, v in sorted(dict(vars(args)).items())))
    export_config(args, os.path.join(args.export, 'config.json'))

    # set random seed
    torch.manual_seed(args.seed)

    print()
    print('Training:')

    # load training/testing data
    train_x, train_y, char2idx, tag2idx = load_train(args.train, args.max_length)
    test_x = load_test(args.test, char2idx)
    args.target_size = len(tag2idx)
    args.vocab_size = len(char2idx)

    # partition train set into train and dev
    num_val = int(len(train_x) * args.val_fraction)
    dev_x, dev_y = train_x[-num_val:], train_y[-num_val:]
    train_x, train_y = train_x[:-num_val], train_y[:-num_val]
    dev_gold = segment(dev_x, dev_y, char2idx, tag2idx)

    # training
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = LSTMTagger(args).to(device)
    loss_function = nn.NLLLoss()

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print('\tstart training on device {}'.format(device))
    model.train()
    best_dev_f = 0.
    niter = 0
    for epoch in range(args.num_epochs):
        for i in range(0, train_x.shape[0], args.batch_size):
            niter += 1
            model.zero_grad()
            j = min(i + args.batch_size, train_x.shape[0])
            xs = torch.tensor(train_x[i:j].T).to(device)
            ys = torch.tensor(train_y[i:j].T).to(device)
            scores = model(xs)
            loss = loss_function(scores.view(-1, args.target_size), ys.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            print('\r\tepoch: {:<5d}  loss: {:.4f}'.format(epoch, loss), end='')

            if niter % args.val_step == 0 or j == train_x.shape[0]:
                model.eval()
                with torch.no_grad():
                    dev_pred = model(torch.tensor(dev_x.T).to(device)).cpu().numpy()
                    dev_pred = dev_pred.argmax(-1).T
                    dev_pred = segment(dev_x, dev_pred, char2idx, tag2idx)
                    dev_f = eval_fscore(dev_gold, dev_pred)
                model.train()
                print('\t[Validation] val_fscore: {:.4f}'.format(dev_f))
                if dev_f > best_dev_f:
                    best_dev_f = dev_f
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'best_dev_f': best_dev_f,
                    }
                    torch.save(checkpoint, os.path.join(args.export, 'model.ckpt'))
                else:
                    # adjust lerning rate if dev_f decreases
                    if args.optimizer == 'SGD':
                        lr = min(args.lr_min, lr * args.lr_decay)
                        adjust_lr(optimizer, lr)

    # evaluation
    print()
    print('Evaluation:')
    print('\tLoading best model')
    checkpoint = torch.load(os.path.join(args.export, 'model.ckpt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print('\tsegmenting test sentences')
    test_y = []
    with torch.no_grad():
        for xs in test_x:
            xs = np.array(xs).reshape((-1, 1))
            ys = model(torch.tensor(xs).to(device)).cpu().numpy()
            ys = ys.argmax(-1).reshape((-1,)).tolist()
            test_y.append(ys)

    test_sents = segment(test_x, test_y, char2idx, tag2idx)
    if args.export != '':
        export(test_sents, os.path.join(args.export, 'prediction.txt'))
        print('\texperiment exported to directory {}'.format(args.export))
    print()


if __name__ == '__main__':
    main()
