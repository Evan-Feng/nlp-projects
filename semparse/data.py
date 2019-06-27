########################################################################
#   data.py - helper class for loading DataLoader                      #
#   author: fengyanlin@pku.edu.cn                                      #
########################################################################

import torch
import numpy as np
from preprocess import BOS_TOK, EOS_TOK, PAD_TOK
from utils import *


class DataLoader(object):

    def __init__(self, filepath, batch_size, cuda, is_test=False, shuffle=True):
        self.filepath = filepath
        self.batch_size = batch_size
        self.cuda = cuda
        self.is_test = is_test

        self.ds = torch.load(filepath)
        self.size = len(self.ds['q'])
        if self.is_test or not shuffle:
            self.perm = np.arange(self.size)
        else:
            self.perm = np.random.permutation(self.size)
        self.q_pad_id = self.ds['qv'].stoi[PAD_TOK]
        self.q_eos_id = self.ds['qv'].stoi[EOS_TOK]
        self.l_bos_id = self.ds['lv'].stoi[BOS_TOK]
        self.l_eos_id = self.ds['lv'].stoi[EOS_TOK]
        self.l_pad_id = self.ds['lv'].stoi[PAD_TOK]

    def __len__(self):
        return self.size

    def __iter__(self):
        for i in range(0, self.size, self.batch_size):
            q, l, r, ei = [], [], [], []
            j = min(self.size, i + self.batch_size)
            for idx in self.perm[i:j]:
                q.append(self.ds['q'][idx] + [self.q_eos_id])
                if not self.is_test:
                    l.append([self.l_bos_id] + self.ds['l'][idx] + [self.l_eos_id])
                    r.append(self.ds['r'][idx])
                    ei.append(self.ds['ei'][idx])

            qx, len_q = pad_sequences(q, self.q_pad_id)
            if self.is_test:
                batch = [qx, None, None, len_q, None, None]
            else:
                lx, len_l = pad_sequences(l, self.l_pad_id)
                rx = torch.tensor(sum(r, []))  # r is flattened into a 1-D array
                ex = torch.tensor(sum(ei, []))  # ei is flattened into a 1-D array
                batch = [qx, lx, rx, len_q, len_l, ex]
            yield to_device(batch, self.cuda)

    def reshuffle(self):
        self.perm = np.random.permutation(self.size)
