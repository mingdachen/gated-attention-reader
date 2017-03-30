#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import os
import logging
from utils.data_preprocessor import data_preprocessor
from utils.minibatch_loader import minibatch_loader
from utils.misc import check_dir, load_word2vec_embeddings
from model import GAReader


def get_args():
    parser = argparse.ArgumentParser(
        description='Gated Attention Reader for \
        Text Comprehension Using TensorFlow')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='whether to keep training from previous model')
    parser.add_argument('--use_feat', action='store_true', default=False,
                        help='whether to use extra features')
    parser.add_argument('--train_emb', action='store_true', default=False,
                        help='whether to train embed')
    parser.add_argument('--data_dir', type=str, default='data/dailymail',
                        help='data directory containing input')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory containing tensorboard logs')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--embed_file', type=str,
                        default='data/word2vec_glove.txt',
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
    parser.add_argument('--gating_fn', type=str, default='tf.multiply',
                        help='gating function')
    parser.add_argument('--drop_out', type=float, default=0.1,
                        help='dropout rate')
    args = parser.parse_args()
    return args


def train(args):
    tf.set_random_seed(args.seed)
    use_chars = args.char_dim > 0
    # load data
    dp = data_preprocessor()
    data = dp.preprocess(
        args.data_dir, no_training_set=False, use_chars=use_chars)

    # build minibatch loader
    train_batch_loader = minibatch_loader(
        data.training, args.batch_size, sample=1.0)
    valid_batch_loader = minibatch_loader(
        data.validation, args.batch_size)
    test_batch_loader = minibatch_loader(
        data.test, args.batch_size)
    if not args.resume:
        logging.info("loading word2vec file ...")
        embed_init, embed_dim = \
            load_word2vec_embeddings(data.dictionary[0], args.embed_file)
        logging.info("embedding dim: {}".format(embed_dim))
        logging.info("initialize model ...")
        model = GAReader(args.n_layers, data.vocab_size, data.n_chars,
                         args.gru_size, embed_dim, args.train_emb,
                         args.char_dim, args.use_feat, args.gating_fn)
        model.build_graph(args.grad_clip, embed_init)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables())
    else:
        model = GAReader(args.n_layers, data.vocab_size, data.n_chars,
                         args.gru_size, 100, args.train_emb,
                         args.char_dim, args.use_feat, args.gating_fn)

    with tf.Session() as sess:
        # training phase
        if not args.resume:
            sess.run(init)
            logging.info('-' * 50)
            logging.info("Initial test ...")
            model.validate(sess, valid_batch_loader)
        else:
            model.restore(sess, args.save_dir)
            saver = tf.train.Saver(tf.global_variables())
        logging.info('-' * 50)
        logging.info("Start training ...")
        current_epoch = sess.run(model.n_epoch)
        while current_epoch < args.n_epoch:
            factor = 1 if current_epoch < 2 else 2 ** (current_epoch - 1)
            lr = args.init_learning_rate / factor
            model.train(sess, args.drop_out, lr, args.print_every,
                        args.save_every, args.save_dir,
                        train_batch_loader, saver)
            # validate model
            model.validate(sess, valid_batch_loader)
            current_epoch = sess.run(model.n_epoch)
        # test model
        logging.info("Final test ...")
        model.validate(sess, test_batch_loader)


if __name__ == "__main__":
    args = get_args()
    # check the existence of directories
    args.data_dir = os.path.join(os.getcwd(), args.data_dir)
    check_dir(args.data_dir, exit_function=True)
    args.log_dir = os.path.join(os.getcwd(), args.log_dir)
    args.save_dir = os.path.join(os.getcwd(), args.save_dir)
    check_dir(args.log_dir, args.save_dir, exit_function=False)
    # initialize log file
    log_file = os.path.join(args.log_dir, 'log')
    if args.log_dir is None:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s',
                            datefmt='%m-%d %H:%M')
    else:
        logging.basicConfig(filename=log_file,
                            filemode='w', level=logging.DEBUG,
                            format='%(asctime)s %(message)s',
                            datefmt='%m-%d %H:%M')
    logging.info(args)
    train(args)
