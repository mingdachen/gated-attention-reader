#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf


def tfconcat(t1, t2):
    return tf.concat([t1, t2], axis=2)


def tfsum(t1, t2):
    return t1 + t2


def gated_attention(doc, qry, inter, mask, gating_fn='tf.multiply'):
    # doc: B x N x D
    # qry: B x Q x D
    # inter: B x N x Q
    # mask (qry): B x Q
    alphas_r = tf.nn.softmax(inter) * \
        tf.cast(tf.expand_dims(mask, axis=1), tf.float32)
    alphas_r = alphas_r / \
        tf.expand_dims(tf.reduce_sum(alphas_r, axis=2), axis=-1)  # B x N x Q
    q_rep = tf.matmul(alphas_r, qry)  # B x N x D
    return eval(gating_fn)(doc, q_rep)


def pairwise_interaction(doc, qry):
    # doc: B x N x D
    # qry: B x Q x D
    shuffled = tf.transpose(qry, perm=[0, 2, 1])  # B x D x Q
    return tf.matmul(doc, shuffled)  # B x N x Q


def attention_sum(doc, qry, cand, cloze, cand_mask=None):
    # doc: B x N x D
    # qry: B x Q x D
    # cand: B x N x C
    # cloze: B x 1
    # cand_mask: B x N
    idx = tf.concat(
        [tf.expand_dims(tf.range(tf.shape(qry)[0]), axis=1),
         tf.expand_dims(cloze, axis=1)], axis=1)
    q = tf.gather_nd(qry, idx)  # B x D
    p = tf.squeeze(
        tf.matmul(doc, tf.expand_dims(q, axis=-1)), axis=-1)  # B x N
    pm = tf.nn.softmax(p) * tf.cast(cand_mask, tf.float32)  # B x N
    pm = pm / tf.expand_dims(tf.reduce_sum(pm, axis=1), axis=-1)  # B x N
    pm = tf.expand_dims(pm, axis=1)  # B x 1 x N
    return tf.squeeze(
        tf.matmul(pm, tf.cast(cand, tf.float32)), axis=1)  # B x C
