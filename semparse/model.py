import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *


class CNNEncoder(nn.Module):
    init_range = 0.1

    def __init__(self, vocab_size, emb_size, kernel_num, kernel_sizes, embed_p):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.embed_p = embed_p

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.emb.weight.data.uniform_(-self.init_range, self.init_range)
        self.emb_dp = EmbeddingDropout(self.emb, embed_p)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, emb_size)) for K in kernel_sizes])

    def forward(self, inputs):
        embeds = self.emb_dp(inputs)
        embeds = embeds.unsqueeze(1)  # batch_size, 1, seq_len, emb_size
        outputs = [F.relu(conv(embeds)).squeeze(3) for conv in self.convs]  # batch_size, kernel_num, seq_len - C
        outputs = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in outputs]
        outputs = torch.cat(outputs, 1)
        return outputs


class MLP(nn.Module):
    """
    Multi-layer perceptron network
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers, hidden_p):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_p = hidden_p

        self.net = nn.Sequential()
        for l in range(num_layers):
            if hidden_p > 0:
                self.net.add_module(f'dropout-{l}', nn.Dropout(hidden_p))
            in_dim = input_size if l == 0 else hidden_size
            out_dim = output_size if l == num_layers - 1 else hidden_size
            self.net.add_module(f'linear-{l}', nn.Linear(in_dim, output_size))
            if (l != num_layers - 1):
                self.net.add_module(f'relu-{l}', nn.ReLU())

    def forward(self, inputs):
        return self.net(inputs)


class CNNClassifier(nn.Module):

    def __init__(self, vocab_size, emb_size, kernel_num, kernel_sizes, embed_p,
                 num_fc_layers, hidden_size, output_size, fc_p):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.embed_p = embed_p
        self.num_fc_layers = num_fc_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc_p = fc_p

        self.fc_in_dim = kernel_num * len(kernel_sizes)
        self.encoder = CNNEncoder(vocab_size, emb_size, kernel_num, kernel_sizes, embed_p)
        self.mlp = MLP(self.fc_in_dim, hidden_size, output_size, num_fc_layers, fc_p)

    def forward(self, inputs):
        return self.mlp(self.encoder(inputs))


class CNNMultiLabelClassifier(nn.Module):

    def __init__(self, vocab_size, emb_size, kernel_num, kernel_sizes, embed_p,
                 num_fc_layers, hidden_size, output_sizes, fc_p):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.embed_p = embed_p
        self.num_fc_layers = num_fc_layers
        self.hidden_size = hidden_size
        self.output_sizes = output_sizes
        self.fc_p = fc_p

        self.fc_in_dim = kernel_num * len(kernel_sizes)
        self.encoder = CNNEncoder(vocab_size, emb_size, kernel_num, kernel_sizes, embed_p)
        self.mlps = []
        for size in output_sizes:
            self.mlps.append(MLP(self.fc_in_dim, hidden_size, size, num_fc_layers, fc_p))
        self.mlps = nn.ModuleList(self.mlps)

    def forward(self, inputs):
        encoded = self.encoder(inputs)
        return [mlp(encoded) for mlp in self.mlps]
