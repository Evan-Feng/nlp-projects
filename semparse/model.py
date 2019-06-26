import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *


class CNNClassifier(nn.Module):
    init_range = 0.1

    def __init__(self, vocab_size, emb_size, num_fc_layers, hidden_size, output_size,
                 kernel_num, kernel_sizes, embed_p, fc_p):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.emb.weight.data.uniform_(-self.init_range, self.init_range)
        self.emb_dp = EmbeddingDropout(self.emb, embed_p)

        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, emb_size)) for K in kernel_sizes])

        if (num_fc_layers < 0):
            raise ValueError('Invalid layer numbers')
        self.fcnet = nn.Sequential()
        for i in range(num_fc_layers):
            if fc_p > 0:
                self.fcnet.add_module('fc-dropout-{}'.format(i), nn.Dropout(p=fc_p))
            in_dim = len(kernel_sizes) * kernel_num if i == 0 else hidden_size
            out_dim = output_size if i == num_fc_layers - 1 else hidden_size
            self.fcnet.add_module('fc-linear-{}'.format(i), nn.Linear(in_dim, output_size))
            if (i != num_fc_layers - 1):
                self.fcnet.add_module('fc-relu-{}'.format(i), nn.ReLU())

    def forward(self, inputs):
        embeds = self.emb_dp(inputs)
        embeds = embeds.unsqueeze(1)  # batch_size, 1, seq_len, emb_size
        x = [F.relu(conv(embeds)).squeeze(3) for conv in self.convs]  # batch_size, kernel_num, seq_len - C
        x = [F.max_pool1d(xx, xx.size(2)).squeeze(2) for xx in x]
        x = torch.cat(x, 1)
        return self.fcnet(x)
