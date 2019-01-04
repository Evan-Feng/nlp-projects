
from __future__ import unicode_literals
import torch
import argparse
import os
import json
from train import PoemGenerater
import subprocess
from rnn_gen import BiRNNGenerator
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as mpm
import numpy as np
from utils import *

mpl.use('Agg')

FONT_PATH = "/home/fengyanlin/songti.ttf"
prop = mpm.FontProperties(fname=FONT_PATH, size=14)


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontproperties=prop)
    ax.set_yticklabels(row_labels, fontproperties=prop)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #          rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--c', default='export/model.ckpt', help='checkpoint file')
    parser.add_argument('-id', '--indices', type=int, nargs='+', default=[np.random.randint(500) for _ in range(4)], help='test sentences\' id to plot')
    parser.add_argument('--test', default='data/test.txt', help='test set')
    parser.add_argument('--savepath', default='export/generated_poem.txt', help='export file')
    parser.add_argument('--output', default='export/attention_map.pdf', help='output file')
    parser.add_argument('--grid', action='store_true')
    args = parser.parse_args()

    ckpt_dic = torch.load(args.c)
    vocab = ckpt_dic['vocab']
    test_x = load_test(args.test, vocab)

    model = ckpt_dic['model_state_dict']
    if next(model.parameters()).is_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model.eval()
    with torch.no_grad():
        output, weights = model.attention_weights(torch.tensor(test_x.T).to(device))
        test_pred = output.cpu().numpy().argmax(-1).T
        weights = np.transpose(weights.cpu().numpy(), (1, 0, 2))

    test_x = [[vocab[i] for i in row] for row in test_x]
    test_y = [[vocab[i] for i in row] for row in test_pred]

    if args.grid:
        n = 10
        args.indices = [np.random.randint(500) for _ in range(n)]
        fig = plt.figure()
        for i, idx in enumerate(args.indices):
            ax = fig.add_subplot(5, 2, i + 1)
            heatmap(weights[idx], test_x[idx], test_y[idx], ax, cmap='Blues', cbarlabel="Attention Weight")

        fig.set_size_inches(12, 16)
        fig.savefig(args.output, format='pdf', bbox_inches='tight')

    else:
        assert len(args.indices) < 10

        n = len(args.indices)
        fig = plt.figure()
        for i, idx in enumerate(args.indices):
            ax = fig.add_subplot('{}1{}'.format(n, i + 1))
            heatmap(weights[idx], test_x[idx], test_y[idx], ax, cmap='Blues', cbarlabel="Attention Weight")

        fig.set_size_inches(6, 3 * n)
        fig.savefig(args.output, format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
