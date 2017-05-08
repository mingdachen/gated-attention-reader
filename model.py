#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from builtins import range

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell as GRU
import time
import os
import logging
from tqdm import trange
from utils.misc import prepare_input
from utils.model_helper import *
from utils.data_preprocessor import MAX_WORD_LEN


class GAReader:
    def __init__(self, n_layers, vocab_size, n_chars,
                 gru_size, embed_dim, train_emb, char_dim,
                 use_feat, gating_fn, save_attn=False):
        self.gru_size = gru_size
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.train_emb = train_emb
        self.char_dim = char_dim
        self.n_chars = n_chars
        self.use_feat = use_feat
        self.save_attn = save_attn
        self.gating_fn = gating_fn
        self.n_vocab = vocab_size
        self.use_chars = self.char_dim != 0

    def build_graph(self, grad_clip, embed_init):
        """
        define model variables
        """
        # word input
        self.doc = tf.placeholder(
            tf.int32, [None, None], name="doc")
        self.qry = tf.placeholder(
            tf.int32, [None, None], name="query")
        self.cand = tf.placeholder(
            tf.int32, [None, None, None], name="cand_ans")
        self.target = tf.placeholder(
            tf.int32, [None, ], name="answer")
        self.cloze = tf.placeholder(
            tf.int32, [None, ], name="cloze")
        # word mask
        self.doc_mask = tf.placeholder(
            tf.int32, [None, None], name="doc_mask")
        self.qry_mask = tf.placeholder(
            tf.int32, [None, None], name="query_mask")
        self.cand_mask = tf.placeholder(
            tf.int32, [None, None], name="cand_mask")
        # char input
        self.doc_char = tf.placeholder(
            tf.int32, [None, None], name="doc_char")
        self.qry_char = tf.placeholder(
            tf.int32, [None, None], name="qry_char")
        self.token = tf.placeholder(
            tf.int32, [None, MAX_WORD_LEN], name="token")
        # char mask
        self.char_mask = tf.placeholder(
            tf.int32, [None, MAX_WORD_LEN], name="char_mask")
        # extra features
        self.feat = tf.placeholder(
            tf.int32, [None, None], name="features")

        # model parameters
        self.lr = tf.placeholder(tf.float32, name="learning_rate")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        # word embedding
        if embed_init is None:
            word_embedding = tf.get_variable(
                "word_embedding", [self.n_vocab, self.embed_dim],
                initializer=tf.random_normal_initializer(stddev=0.1),
                trainable=self.train_emb)
        else:
            word_embedding = tf.Variable(embed_init, trainable=self.train_emb,
                                         name="word_embedding")
        doc_embed = tf.nn.embedding_lookup(word_embedding, self.doc)
        qry_embed = tf.nn.embedding_lookup(word_embedding, self.qry)

        # feature embedding
        feature_embedding = tf.get_variable(
            "feature_embedding", [2, 2],
            initializer=tf.random_normal_initializer(stddev=0.1),
            trainable=self.train_emb)
        feat_embed = tf.nn.embedding_lookup(feature_embedding, self.feat)

        # char embedding
        if self.use_chars:
            char_embedding = tf.get_variable(
                "char_embedding", [self.n_chars, self.char_dim],
                initializer=tf.random_normal_initializer(stddev=0.1))
            token_embed = tf.nn.embedding_lookup(char_embedding, self.token)
            fw_gru = GRU(self.char_dim)
            bk_gru = GRU(self.char_dim)
            # fw_states/bk_states: [batch_size, gru_size]
            # only use final state
            seq_length = tf.reduce_sum(self.char_mask, axis=1)
            _, (fw_final_state, bk_final_state) = \
                tf.nn.bidirectional_dynamic_rnn(
                fw_gru, bk_gru, token_embed, sequence_length=seq_length,
                dtype=tf.float32, scope="char_rnn")
            fw_embed = tf.layers.dense(
                fw_final_state, self.embed_dim // 2)
            bk_embed = tf.layers.dense(
                bk_final_state, self.embed_dim // 2)
            merge_embed = fw_embed + bk_embed
            doc_char_embed = tf.nn.embedding_lookup(merge_embed, self.doc_char)
            qry_char_embed = tf.nn.embedding_lookup(merge_embed, self.qry_char)

            doc_embed = tf.concat([doc_embed, doc_char_embed], axis=2)
            qry_embed = tf.concat([qry_embed, qry_char_embed], axis=2)

        self.attentions = []
        if self.save_attn:
            inter = pairwise_interaction(doc_embed, qry_embed)
            self.attentions.append(inter)

        for i in range(self.n_layers - 1):
            fw_doc = GRU(self.gru_size)
            bk_doc = GRU(self.gru_size)
            seq_length = tf.reduce_sum(self.doc_mask, axis=1)
            (fw_doc_states, bk_doc_states), _ = \
                tf.nn.bidirectional_dynamic_rnn(
                fw_doc, bk_doc, doc_embed, sequence_length=seq_length,
                dtype=tf.float32, scope="{}_layer_doc_rnn".format(i))
            doc_bi_embed = tf.concat([fw_doc_states, bk_doc_states], axis=2)

            fw_qry = GRU(self.gru_size)
            bk_qry = GRU(self.gru_size)
            seq_length = tf.reduce_sum(self.qry_mask, axis=1)
            (fw_qry_states, bk_qry_states), _ = \
                tf.nn.bidirectional_dynamic_rnn(
                fw_qry, bk_qry, qry_embed, sequence_length=seq_length,
                dtype=tf.float32, scope="{}_layer_qry_rnn".format(i))
            qry_bi_embed = tf.concat([fw_qry_states, bk_qry_states], axis=2)

            inter = pairwise_interaction(doc_bi_embed, qry_bi_embed)
            doc_inter_embed = gated_attention(
                doc_bi_embed, qry_bi_embed, inter, self.qry_mask,
                gating_fn=self.gating_fn)
            doc_embed = tf.nn.dropout(doc_inter_embed, self.keep_prob)
            if self.save_attn:
                self.attentions.append(inter)

        if self.use_feat:
            doc_embed = tf.concat([doc_embed, feat_embed], axis=2)
        # final layer
        fw_doc_final = GRU(self.gru_size)
        bk_doc_final = GRU(self.gru_size)
        seq_length = tf.reduce_sum(self.doc_mask, axis=1)
        (fw_doc_states, bk_doc_states), _ = tf.nn.bidirectional_dynamic_rnn(
            fw_doc_final, bk_doc_final, doc_embed, sequence_length=seq_length,
            dtype=tf.float32, scope="final_doc_rnn")
        doc_embed_final = tf.concat([fw_doc_states, bk_doc_states], axis=2)

        fw_qry_final = GRU(self.gru_size)
        bk_doc_final = GRU(self.gru_size)
        seq_length = tf.reduce_sum(self.qry_mask, axis=1)
        (fw_qry_states, bk_qry_states), _ = tf.nn.bidirectional_dynamic_rnn(
            fw_qry_final, bk_doc_final, qry_embed, sequence_length=seq_length,
            dtype=tf.float32, scope="final_qry_rnn")
        qry_embed_final = tf.concat([fw_qry_states, bk_qry_states], axis=2)

        if self.save_attn:
            inter = pairwise_interaction(doc_embed_final, qry_embed_final)
            self.attentions.append(inter)

        self.pred = attention_sum(
            doc_embed_final, qry_embed_final, self.cand,
            self.cloze, self.cand_mask)
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.target, logits=self.pred, name="cross_entropy"))
        self.pred_ans = tf.cast(tf.argmax(self.pred, axis=1), tf.int32)
        self.test = tf.cast(tf.equal(self.target, self.pred_ans), tf.float32)
        self.accuracy = tf.reduce_sum(
            tf.cast(tf.equal(self.target, self.pred_ans), tf.float32))
        vars_list = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(self.lr)
        # gradient clipping
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, vars_list), grad_clip)
        # for grad, var in zip(grads, vars_list):
        #     tf.summary.histogram(var.name + '/gradient', grad)
        self.updates = optimizer.apply_gradients(zip(grads, vars_list))
        self.save_vars()

    def save_vars(self):
        """
        for restoring model
        """
        tf.add_to_collection('doc', self.doc)
        tf.add_to_collection('qry', self.qry)
        tf.add_to_collection('doc_char', self.doc_char)
        tf.add_to_collection('qry_char', self.qry_char)
        tf.add_to_collection('target', self.target)
        tf.add_to_collection('doc_mask', self.doc_mask)
        tf.add_to_collection('qry_mask', self.qry_mask)
        tf.add_to_collection('token', self.token)
        tf.add_to_collection('char_mask', self.char_mask)
        tf.add_to_collection('cand', self.cand)
        tf.add_to_collection('cand_mask', self.cand_mask)
        tf.add_to_collection('cloze', self.cloze)
        tf.add_to_collection('feat', self.feat)
        tf.add_to_collection('keep_prob', self.keep_prob)
        tf.add_to_collection('loss', self.loss)
        tf.add_to_collection('accuracy', self.accuracy)
        tf.add_to_collection('updates', self.updates)
        tf.add_to_collection('learning_rate', self.lr)

    def train(self, sess, dw, dt, qw, qt, a, m_dw, m_qw, tt,
              tm, c, m_c, cl, fnames, dropout, learning_rate):
        """
        train model
        Args:
        - data: (object) containing training data
        """
        feed_dict = {self.doc: dw, self.qry: qw,
                     self.doc_char: dt, self.qry_char: qt, self.target: a,
                     self.doc_mask: m_dw, self.qry_mask: m_qw,
                     self.token: tt, self.char_mask: tm,
                     self.cand: c, self.cand_mask: m_c,
                     self.cloze: cl, self.keep_prob: 1 - dropout,
                     self.lr: learning_rate}
        if self.use_feat:
            feat = prepare_input(dw, qw)
            feed_dict += {self.feat: feat}

        loss, acc, _, = \
            sess.run([self.loss, self.accuracy, self.updates], feed_dict)
        return loss, acc

    def validate(self, sess, data):
        """
        test the model
        """
        loss = acc = n_exmple = 0
        tr = trange(
            len(data),
            desc="loss: {:.3f}, acc: {:.3f}".format(0.0, 0.0),
            leave=False)
        for dw, dt, qw, qt, a, m_dw, m_qw, tt, \
                tm, c, m_c, cl, fnames in data:
            start = time.time()
            feed_dict = {self.doc: dw, self.qry: qw,
                         self.doc_char: dt, self.qry_char: qt, self.target: a,
                         self.doc_mask: m_dw, self.qry_mask: m_qw,
                         self.token: tt, self.char_mask: tm,
                         self.cand: c, self.cand_mask: m_c,
                         self.cloze: cl, self.keep_prob: 1.,
                         self.lr: 0.}
            if self.use_feat:
                feat = prepare_input(dw, qw)
                feed_dict += {self.feat: feat}
            _loss, _acc = sess.run([self.loss, self.accuracy], feed_dict)
            n_exmple += dw.shape[0]
            loss += _loss
            acc += _acc
            tr.set_description("loss: {:.3f}, acc: {:.3f}".
                               format(_loss, _acc / dw.shape[0]))
            tr.update()
        tr.close()
        loss /= n_exmple
        acc /= n_exmple
        spend = (time.time() - start) / 60
        statement = "loss: {:.3f}, acc: {:.3f}, time: {:.1f}(m)"\
            .format(loss, acc, spend)
        logging.info(statement)
        return loss, acc

    def restore(self, sess, checkpoint_dir):
        """
        restore model
        """
        checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
        loader = tf.train.import_meta_graph(checkpoint_path + '.meta')
        loader.restore(sess, checkpoint_path)
        logging.info("model restored from {}".format(checkpoint_path))
        # restore variables from checkpoint
        self.doc = tf.get_collection('doc')[0]
        self.qry = tf.get_collection('qry')[0]
        self.doc_char = tf.get_collection('doc_char')[0]
        self.qry_char = tf.get_collection('qry_char')[0]
        self.target = tf.get_collection('target')[0]
        self.doc_mask = tf.get_collection('doc_mask')[0]
        self.qry_mask = tf.get_collection('qry_mask')[0]
        self.token = tf.get_collection('token')[0]
        self.char_mask = tf.get_collection('char_mask')[0]
        self.cand = tf.get_collection('cand')[0]
        self.cand_mask = tf.get_collection('cand_mask')[0]
        self.cloze = tf.get_collection('cloze')[0]
        self.feat = tf.get_collection('feat')[0]
        self.keep_prob = tf.get_collection('keep_prob')[0]
        self.loss = tf.get_collection('loss')[0]
        self.accuracy = tf.get_collection('accuracy')[0]
        self.updates = tf.get_collection('updates')[0]
        self.lr = tf.get_collection('learning_rate')[0]

    def save(self, sess, saver, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path)
        logging.info("model saved to {}".format(checkpoint_path))
