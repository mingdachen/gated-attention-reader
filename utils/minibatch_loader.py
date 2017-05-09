#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from builtins import range


import numpy as np
import random
MAX_WORD_LEN = 10


class minibatch_loader:
    def __init__(self, questions, batch_size, shuffle=True, sample=1.0):
        self.batch_size = batch_size
        if sample == 1.0:
            self.questions = questions
        else:
            self.questions = random.sample(
                questions, int(sample * len(questions)))
        self.max_num_cand = max(list(map(lambda x: len(x[3]), self.questions)))
        self.max_word_len = MAX_WORD_LEN
        self.shuffle = shuffle
        self.reset()

    def __len__(self):
        return len(self.batch_pool)

    def __iter__(self):
        """make the object iterable"""
        return self

    def reset(self):
        """new iteration"""
        self.ptr = 0

        self.batch_pool = []
        n = len(self.questions)
        idx_list = np.arange(0, n, self.batch_size)
        if self.shuffle:
            idx = np.arange(n)
            np.random.shuffle(idx)
            self.questions = self.questions[idx]
        for idx in idx_list:
            self.batch_pool.append(
                np.arange(idx, min(idx + self.batch_size, n)))

    def __next__(self):
        """load the next batch"""
        if self.ptr == len(self.batch_pool):
            self.reset()
            raise StopIteration()

        ixs = self.batch_pool[self.ptr]
        doc_len = [len(self.questions[idx][0]) for idx in ixs]
        qry_len = [len(self.questions[idx][1]) for idx in ixs]
        curr_max_doc_len = np.max(doc_len)
        curr_max_qry_len = np.max(qry_len)
        curr_batch_size = len(ixs)

        # document words
        dw = np.zeros(
            (curr_batch_size, curr_max_doc_len),
            dtype='int32')
        # query words
        qw = np.zeros(
            (curr_batch_size, curr_max_qry_len),
            dtype='int32')
        # candidate answers
        c = np.zeros(
            (curr_batch_size, curr_max_doc_len, self.max_num_cand),
            dtype='int16')
        # position of cloze in query
        cl = np.zeros(
            (curr_batch_size, ),
            dtype='int32')
        # document word mask
        m_dw = np.zeros(
            (curr_batch_size, curr_max_doc_len),
            dtype='int32')
        # query word mask
        m_qw = np.zeros(
            (curr_batch_size, curr_max_qry_len),
            dtype='int32')
        # candidate mask
        m_c = np.zeros(
            (curr_batch_size, curr_max_doc_len),
            dtype='int32')
        # correct answer
        a = np.zeros((curr_batch_size, ), dtype='int32')
        fnames = [''] * curr_batch_size

        types = {}

        for n, ix in enumerate(ixs):

            doc_w, qry_w, ans, cand, doc_c, \
                qry_c, cloze, fname = self.questions[ix]

            # document, query and candidates
            dw[n, : len(doc_w)] = np.array(doc_w)
            qw[n, : len(qry_w)] = np.array(qry_w)
            m_dw[n, : len(doc_w)] = 1
            m_qw[n, : len(qry_w)] = 1
            for it, word in enumerate(doc_c):
                wtuple = tuple(word)
                if wtuple not in types:
                    types[wtuple] = []
                types[wtuple].append((0, n, it))
            for it, word in enumerate(qry_c):
                wtuple = tuple(word)
                if wtuple not in types:
                    types[wtuple] = []
                types[wtuple].append((1, n, it))

            # search candidates in doc
            for it, cc in enumerate(cand):
                index = [ii for ii in range(len(doc_w)) if doc_w[ii] in cc]
                m_c[n, index] = 1
                c[n, index, it] = 1
                if ans == cc:
                    a[n] = it  # answer

            cl[n] = cloze
            fnames[n] = fname

        # create type character matrix and indices for doc, qry
        # document token index
        dt = np.zeros(
            (curr_batch_size, curr_max_doc_len),
            dtype='int32')
        # query token index
        qt = np.zeros(
            (curr_batch_size, curr_max_qry_len),
            dtype='int32')
        # type characters
        tt = np.zeros(
            (len(types), self.max_word_len),
            dtype='int32')
        # type mask
        tm = np.zeros(
            (len(types), self.max_word_len),
            dtype='int32')
        n = 0
        for k, v in types.items():
            tt[n, : len(k)] = np.array(k)
            tm[n, : len(k)] = 1
            for (sw, bn, sn) in v:
                if sw == 0:
                    dt[bn, sn] = n
                else:
                    qt[bn, sn] = n
            n += 1

        self.ptr += 1

        return dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, cl, fnames
