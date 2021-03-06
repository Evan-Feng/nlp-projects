import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import floor, ceil
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


class PCNNEncoder(nn.Module):
    init_range = 0.1

    def __init__(self, vocab_size, emb_size, kernel_num, kernel_sizes, embed_p, token_ids):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.embed_p = embed_p
        self.token_ids = token_ids  # the IDs of the special tokens used to identify the locations of entities

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.emb.weight.data.uniform_(-self.init_range, self.init_range)
        self.emb_dp = EmbeddingDropout(self.emb, embed_p)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, emb_size)) for K in kernel_sizes])

    def forward(self, inputs):
        """
        inputs: torch.Tensor of shape (B, L)
        inputs: torch.Tensor of shape (B, L)
        """
        dev = inputs.device
        bs, sl = inputs.size()
        inputs_np = inputs.cpu().numpy()
        embeds = self.emb_dp(inputs)
        embeds = embeds.unsqueeze(1)  # batch_size, 1, seq_len, emb_size
        outputs = [conv(F.pad(embeds, [0, 0, floor((k - 1) / 2), ceil((k - 1) / 2)], value=0)).squeeze(3)
                   for conv, k in zip(self.convs, self.kernel_sizes)]  # (B, C, L)
        # outputs = [conv(embeds).squeeze(3) for conv in self.convs[:1]]
        outputs = [F.relu(x) for x in outputs]
        # outputs = [F.pad(x, (floor((k - 1) / 2), ceil((k - 1) / 2)), value=-1) for x, k in zip(outputs, self.kernel_sizes)]  # (B, C, L)
        outputs = torch.cat(outputs, 1)  # (B, C', L)

        pooled = []
        ai_np = np.arange(bs)
        ri = torch.arange(sl).to(dev).unsqueeze(0).expand(bs, sl)
        prev_ids = inputs.new_zeros(bs)
        # print(inputs == self.token_ids[0])
        # print(inputs == self.token_ids[-1])
        for t in self.token_ids:
            ids_np = (inputs_np == t).argmax(1)
            ids = torch.tensor(ids_np, device=dev)
            inputs_np[ai_np, ids_np] = -1
            lmask = (ri < prev_ids.unsqueeze(-1)).unsqueeze(1)
            rmask = (ri > ids.unsqueeze(-1)).unsqueeze(1)
            masked = outputs.masked_fill(lmask, float('-inf')).masked_fill(rmask, float('-inf'))
            pooled.append(F.max_pool1d(masked, masked.size(2)).squeeze(2))  # (B, C)
            prev_ids = ids
        outputs = torch.cat(pooled, -1)
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


class PCNNClassifier(nn.Module):

    def __init__(self, vocab_size, emb_size, kernel_num, kernel_sizes, embed_p,
                 num_fc_layers, hidden_size, output_size, fc_p, tok_ids):
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

        self.fc_in_dim = len(tok_ids) * kernel_num * len(kernel_sizes)
        self.encoder = PCNNEncoder(vocab_size, emb_size, kernel_num, kernel_sizes, embed_p, tok_ids)
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


class ChainedCNNMultiLabelClassifier(nn.Module):
    init_range = 0.1

    def __init__(self, vocab_size, emb_size, kernel_num, kernel_sizes, embed_p,
                 num_fc_layers, hidden_size, output_sizes, fc_p, rel_emb_size):
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
        self.rel_emb = nn.Embedding(output_sizes[0], rel_emb_size)
        self.mlps = []
        curr_dim = self.fc_in_dim
        for size in output_sizes:
            self.mlps.append(MLP(curr_dim, hidden_size, size, num_fc_layers, fc_p))
            curr_dim += rel_emb_size
        self.mlps = nn.ModuleList(self.mlps)

    def forward(self, inputs):
        encoded = self.encoder(inputs)
        history = []
        outputs = []
        for mlp in self.mlps:
            inputs = torch.cat([encoded] + history, -1)
            out = mlp(inputs)
            outputs.append(out)
            _, pred = out.max(-1)
            pred = pred.detach()
            history.append(self.rel_emb(pred))
        return outputs
