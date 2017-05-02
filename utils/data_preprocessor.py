#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from builtins import range

import glob
import os
import logging
from tqdm import tqdm
import numpy as np
from multiprocessing.dummy import Pool, Manager
from functools import partial
try:
    NUM_THREADS = int(os.environ['OMP_NUM_THREADS'])
except KeyError:
    NUM_THREADS = 4

MAX_WORD_LEN = 10

SYMB_BEGIN = "@begin"
SYMB_END = "@end"


class data_holder:
    def __init__(self, dictionary, n_entities, training, validation, test):
        self.dictionary = dictionary
        self.training = np.asarray(training)
        self.validation = validation
        self.test = test
        self.vocab_size = len(dictionary[0])
        self.n_chars = len(dictionary[1])
        self.n_entities = n_entities
        self.inv_dictionary = {v: k for k, v in dictionary[0].items()}


class data_preprocessor:
    def preprocess(self, question_dir, max_example=None,
                   no_training_set=False, use_chars=True):
        """
        preprocess all data into a standalone data object.
        the training set will be left out (to save debugging time)
        when no_training_set is True.
        """
        vocab_f = os.path.join(question_dir, "vocab.txt")
        word_dictionary, char_dictionary, num_entities = \
            self.make_dictionary(
                question_dir, vocab_file=vocab_f)
        dictionary = (word_dictionary, char_dictionary)
        if no_training_set:
            training = None
        else:
            logging.info("preparing training data ...")
            training = self.parse_all_files(
                question_dir + "/training", dictionary, max_example, use_chars)
        logging.info("preparing validation data ...")
        validation = self.parse_all_files(
            question_dir + "/validation", dictionary, None, use_chars)
        logging.info("preparing test data ...")
        test = self.parse_all_files(
            question_dir + "/test", dictionary, None, use_chars)

        data = data_holder(
            dictionary, num_entities, training, validation, test)
        return data

    def make_dictionary(self, question_dir, vocab_file):

        if os.path.exists(vocab_file):
            logging.info("loading vocabularies from " + vocab_file + " ...")
            vocabularies = list(
                map(lambda x: x.strip(),
                    open(vocab_file, encoding='utf-8').readlines()))
        else:
            logging.info("no " + vocab_file +
                         " found, constructing the vocabulary list ...")

            fnames = []
            fnames += glob.glob(
                os.path.join(question_dir, 'test', '*.question'))
            fnames += glob.glob(
                os.path.join(question_dir, 'validation', '*.question'))
            fnames += glob.glob(
                os.path.join(question_dir, 'training', '*.question'))

            vocab_set = set()

            def process(fname, vocab_set):
                with open(fname, encoding='utf-8') as fp:
                    fp.readline()
                    fp.readline()
                    document = fp.readline().split()
                    fp.readline()
                    query = fp.readline().split()

                    vocab_set |= set(document) | set(query)
                return 0

            with Pool(NUM_THREADS) as pool:
                for _ in tqdm(
                    pool.imap_unordered(
                        partial(
                            process,
                            vocab_set=vocab_set),
                        fnames), total=len(fnames)):
                    pass
            entities = set(e for e in vocab_set if e.startswith('@entity'))

            # @placehoder, @begin and @end are included in the vocabulary list
            tokens = vocab_set.difference(entities)
            tokens.add(SYMB_BEGIN)
            tokens.add(SYMB_END)

            vocabularies = list(entities) + list(tokens)

            logging.info("writing vocabularies to " + vocab_file + " ...")
            with open(vocab_file, "w", encoding='utf-8') as vocab_fp:
                vocab_fp.write('\n'.join(vocabularies))
        # print(type(vocabularies))
        vocab_size = len(vocabularies)
        # word dictionary: word -> index
        word_dictionary = dict(zip(vocabularies, range(vocab_size)))
        char_set = set([c for w in vocabularies for c in list(w)])
        char_set.add(' ')
        # char dictionary: char -> index
        char_dictionary = dict(zip(list(char_set), range(len(char_set))))
        num_entities = len(
            [v for v in vocabularies if v.startswith('@entity')])
        logging.info("vocab_size = %d" % vocab_size)
        logging.info("num characters = %d" % len(char_set))
        logging.info("%d anonymoused entities" % num_entities)
        logging.info("%d other tokens (including @placeholder, %s and %s)" % (
                     vocab_size - num_entities, SYMB_BEGIN, SYMB_END))

        return word_dictionary, char_dictionary, num_entities

    def parse_one_file(self, fname, w_dict, c_dict, use_chars):
        """
        parse a *.question file into tuple(document, query, answer, filename)
        and convert them into indices
        """
        with open(fname, encoding='utf-8') as fp:
            raw = fp.readlines()
        doc_raw = raw[2].split()  # document
        qry_raw = raw[4].split()  # query
        ans_raw = raw[6].strip()  # answer
        # candidate answers
        cand_raw = list(
            map(lambda x: x.strip().split(':')[0].split(), raw[8:]))

        # wrap the query with special symbols
        qry_raw.insert(0, SYMB_BEGIN)
        qry_raw.append(SYMB_END)
        try:
            cloze = qry_raw.index('@placeholder')
        except ValueError:
            logging.info('@placeholder not found in ', fname, '. Fixing...')
            at = qry_raw.index('@')
            qry_raw = qry_raw[: at] +\
                [''.join(qry_raw[at:at + 2])] + qry_raw[at + 2:]
            cloze = qry_raw.index('@placeholder')

        # tokens/entities --> indexes
        doc_words = list(map(lambda w: w_dict[w], doc_raw))
        qry_words = list(map(lambda w: w_dict[w], qry_raw))
        if use_chars:
            doc_chars = list(
                map(lambda w: map(lambda c: c_dict.get(c, c_dict[' ']),
                    list(w)[: MAX_WORD_LEN]), doc_raw))
            qry_chars = list(
                map(lambda w: map(lambda c: c_dict.get(c, c_dict[' ']),
                    list(w)[: MAX_WORD_LEN]), qry_raw))
        else:
            doc_chars, qry_chars = [], []
        # ans/cand --> index
        ans = list(map(lambda w: w_dict.get(w, 0), ans_raw.split()))
        cand = [list(map(lambda w:w_dict.get(w, 0), c)) for c in cand_raw]

        return doc_words, qry_words, ans, cand, \
            doc_chars, qry_chars, cloze, fname

    def parse_all_files(self, directory, dictionary, max_example, use_chars):
        """
        parse all files under the given directory into a list of questions,
        where each element is in the form of
        (document, query, answer, filename)
        """
        all_files = glob.glob(directory + '/*.question')[: 100]
        manager = Manager()
        w_dict, c_dict = dictionary[0], dictionary[1]
        w_dict_m = manager.dict(w_dict)
        c_dict_m = manager.dict(c_dict)
        with Pool(NUM_THREADS) as pool:
            questions = []
            for question in tqdm(
                pool.imap_unordered(
                    partial(
                        self.parse_one_file,
                        w_dict=w_dict_m,
                        c_dict=c_dict_m,
                        use_chars=use_chars),
                    all_files), total=len(all_files)):
                questions.append(question)
        return questions

    def gen_text_for_word2vec(self, question_dir, text_file):

            fnames = []
            fnames += glob.glob(question_dir + "/training/*.question")

            out = open(text_file, "w")

            for fname in fnames:

                fp = open(fname, encoding='utf-8')
                fp.readline()
                fp.readline()
                document = fp.readline()
                fp.readline()
                query = fp.readline()
                fp.close()

                out.write(document.strip())
                out.write(" ")
                out.write(query.strip())

            out.close()
