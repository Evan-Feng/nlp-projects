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
        tag_scores = F.log_softmax(tag_space, dim=1).view(self.max_length, -1, self.target_size)
        return tag_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='./data/train.txt', help='training data')
    parser.add_argument('--test', default='./data/test.txt', help='testing data')
    parser.add_argument('--export', default='./export/', help='export directory')
    parser.add_argument('--seed', type=float, default=0, help='random seed')
    parser.add_argument('--emb_dim', type=int, default=64, help='dimensionality of the charecter embedding')
    parser.add_argument('--hidden_dim', type=int, default=64, help='number of hidden units')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size')
    parser.add_argument('--max_length', type=int, default=64, help='maximum sequence length (shorter padded, longer truncated)')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs to run')
    parser.add_argument('-bi', '--bidirectional', type=bool, default=True, help='use bidirectional lstm')
    args = parser.parse_args()

    print('==================== Configuration ====================')
    print('\n'.join('\t{:15} {}'.format(k + ':', str(v)) for k, v in sorted(dict(vars(args)).items())))

    # set random seed
    torch.manual_seed(args.seed)

    print()
    print('====================== Training =======================')

    # load training/testing data
    train_x, train_y, char2idx, tag2idx = load_train(args.train, args.max_length)
    test_x = load_test(args.test, args.max_length, char2idx)
    m = train_x.shape[0]
    args.target_size = len(tag2idx)
    args.vocab_size = len(char2idx)

    # training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LSTMTagger(args).to(device)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print('\tstart training on device {}'.format(device))
    for epoch in range(args.num_epochs):
        for i in range(0, m, args.batch_size):
            model.zero_grad()
            j = min(i + args.batch_size, m)
            xs = torch.tensor(train_x[i:j].T).to(device)
            ys = torch.tensor(train_y[i:j].T).to(device)
            scores = model(xs)
            loss = loss_function(scores.view(-1, args.target_size), ys.view(-1))
            loss.backward()
            optimizer.step()
            print('\r\tepoch: {:d}  loss: {:.4f}'.format(epoch, loss), end='')
        print()

    with torch.no_grad():
        test_y = model(torch.tensor(test_x.T).to(device)).cpu().numpy()
        test_y = test_y.argmax(-1).T

    print()
    print('===================== Evaluation ======================')
    print('\tsegmenting test sentences')
    test_sents = segment(test_x, test_y, char2idx, tag2idx)
    if args.export != '':
        export(test_sents, os.path.join(args.export, 'prediction.txt'))
        print('\texperiment exported to directory {}'.format(args.export))
    print()
    print('=======================================================')


if __name__ == '__main__':
    main()
