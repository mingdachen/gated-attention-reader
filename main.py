#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import argparse
import torch
import numpy as np
import time
import os
import logging
from utils.data_preprocessor import data_preprocessor
from utils.minibatch_loader import minibatch_loader
from utils.misc import to_vars, check_dir, load_word2vec_embeddings
from model import GAReader


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_args():
    parser = argparse.ArgumentParser(
        description='Gated Attention Reader for \
        Text Comprehension Using PyTorch')
    parser.register('type', 'bool', str2bool)

    parser.add_argument('--resume', type='bool', default=False,
                        help='whether to keep training from previous model')
    parser.add_argument('--use_feat', type='bool', default=False,
                        help='whether to use extra features')
    parser.add_argument('--train_emb', type='bool', default=True,
                        help='whether to train embed')
    parser.add_argument('--cuda', type='bool', default=False,
                        help='whether to use CUDA')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='data directory containing input data')
    parser.add_argument('--log_file', type=str, default=None,
                        help='log file')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='directory to store checkpointed models')
    parser.add_argument('--embed_file', type=str, default=None,
                        help='word embedding initialization file')
    parser.add_argument('--gru_size', type=int, default=256,
                        help='size of word GRU hidden state')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='number of layers of the model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='mini-batch size')
    parser.add_argument('--n_epoch', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1,
                        help='save frequency')
    parser.add_argument('--print_every', type=int, default=50,
                        help='print frequency')
    parser.add_argument('--grad_clip', type=float, default=10,
                        help='clip gradients at this value')
    parser.add_argument('--init_learning_rate', type=float, default=5e-4,
                        help='initial learning rate')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for tensorflow')
    parser.add_argument('--char_dim', type=int, default=50,
                        help='size of character GRU hidden state')
    parser.add_argument('--gating_fn', type=str, default='torch.mul',
                        help='gating function')
    parser.add_argument('--drop_out', type=float, default=0.1,
                        help='dropout rate')
    args = parser.parse_args()
    return args


def train(args):
    use_chars = args.char_dim > 0
    # load data
    dp = data_preprocessor()
    data = dp.preprocess(
        args.data_dir, no_training_set=True, use_chars=use_chars)

    # build minibatch loader
    train_batch_loader = minibatch_loader(
        data.training, args.batch_size, sample=1.0)
    valid_batch_loader = minibatch_loader(
        data.validation, args.batch_size)
    test_batch_loader = minibatch_loader(
        data.test, args.batch_size)

    logging.info("loading word2vec file ...")
    embed_init, embed_dim = \
        load_word2vec_embeddings(data.dictionary[0], args.embed_file)
    logging.info("embedding dim: {}".format(embed_dim))
    logging.info("initialize model ...")
    model = GAReader(args.n_layers, data.vocab_size, data.n_chars,
                     args.drop_out, args.gru_size, embed_init, embed_dim,
                     args.train_emb, args.char_dim, args.use_feat,
                     args.gating_fn)
    if args.cuda:
        model.cuda()
    # training phase
    opt = torch.optim.Adam(model.parameters(), lr=args.init_learning_rate)
    logging.info('-' * 50)
    logging.info("Start training ...")
    model.train()
    for epoch in range(args.n_epoch):
        acc = loss = n_examples = it = 0
        start = time.time()
        for dw, dt, qw, qt, a, m_dw, m_qw, tt, \
                tm, c, m_c, cl, fnames in test_batch_loader:
            n_examples += dw.shape[0]
            dw, dt, qw, qt, a, m_dw, m_qw, tt, \
                tm, c, m_c, cl = to_vars([dw, dt, qw, qt, a, m_dw, m_qw, tt,
                                         tm, c, m_c, cl])
            loss_, acc_ = model(dw, dt, qw, qt, a, m_dw, m_qw, tt,
                                tm, c, m_c, cl, fnames)
            loss += loss_.data.numpy()[0]
            acc += acc_.data.numpy()[0]
            it += 1
            opt.zero_grad()
            loss_.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
            opt.step()
            if it % args.print_every == 0:
                spend = (time.time() - start) / 60
                statement = "epoch: {}, it: {} (max: {}), time: {:.3f}(m), "\
                    .format(epoch, it, test_batch_loader.n_batches, spend)
                statement += "loss: {:.3f}, acc: {:.3f}"\
                    .format(loss / n_examples, acc / n_examples)
                logging.info(statement)
                start = time.time()

    # test model
    logging.info("Final test ...")


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # check the existence of directories
    args.data_dir = os.path.join(os.getcwd(), args.data_dir)
    check_dir(args.data_dir, exit_function=True)
    # args.save_dir = os.path.join(os.getcwd(), args.save_dir)
    # check_dir(args.save_dir, exit_function=False)
    # initialize log file
    if args.log_file is None:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s',
                            datefmt='%m-%d %H:%M')
    else:
        logging.basicConfig(filename=args.log_file,
                            filemode='w', level=logging.DEBUG,
                            format='%(asctime)s %(message)s',
                            datefmt='%m-%d %H:%M')
    logging.info(args)
    train(args)
