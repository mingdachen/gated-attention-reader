#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from builtins import range


import glob
import os
import logging
from tqdm import tqdm
import time

MAX_WORD_LEN = 10

SYMB_BEGIN = "@begin"
SYMB_END = "@end"


class data_holder:
    def __init__(self, dictionary, n_entities, training, validation, test):
        self.dictionary = dictionary
        self.training = training
        self.validation = validation
        self.test = test
        self.vocab_size = len(dictionary[0])
        self.n_chars = len(dictionary[1])
        self.n_entities = n_entities
        self.inv_dictionary = {v: k for k, v in dictionary[0].items()}


class data_preprocessor:
    def preprocess(self, question_dir, no_training_set=False, use_chars=True):
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
                question_dir + "/training", dictionary, use_chars)
        logging.info("preparing validation data ...")
        validation = self.parse_all_files(
            question_dir + "/validation", dictionary, use_chars)
        logging.info("preparing test data ...")
        test = self.parse_all_files(
            question_dir + "/test", dictionary, use_chars)

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
            n = 0.
            for fname in fnames:
                # print("processing: ", fname)
                fp = open(fname, encoding='utf-8')
                fp.readline()
                fp.readline()
                document = fp.readline().split()
                fp.readline()
                query = fp.readline().split()
                fp.close()

                vocab_set |= set(document) | set(query)

                # show progress
                n += 1
                if n % 10000 == 0:
                    logging.info('%3d%%' % int(100 * n / len(fnames)))

            entities = set(e for e in vocab_set if e.startswith('@entity'))

            # @placehoder, @begin and @end are included in the vocabulary list
            tokens = vocab_set.difference(entities)
            tokens.add(SYMB_BEGIN)
            tokens.add(SYMB_END)

            vocabularies = list(entities) + list(tokens)

            logging.info("writing vocabularies to " + vocab_file + " ...")
            vocab_fp = open(vocab_file, "w", encoding='utf-8')
            vocab_fp.write('\n'.join(vocabularies))
            vocab_fp.close()
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

    def parse_one_file(self, fname, dictionary, use_chars):
        """
        parse a *.question file into tuple(document, query, answer, filename)
        and convert them into indices
        """
        w_dict, c_dict = dictionary[0], dictionary[1]
        raw = open(fname, encoding='utf-8').readlines()
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

        return doc_words, qry_words, ans, cand, doc_chars, qry_chars, cloze

    def parse_all_files(self, directory, dictionary, use_chars):
        """
        parse all files under the given directory into a list of questions,
        where each element is in the form of
        (document, query, answer, filename)
        """
        all_files = glob.glob(directory + '/*.question')
        questions = []
        for f in tqdm(all_files):
            time.sleep(0.1)
            question = self.parse_one_file(f, dictionary, use_chars)
            questions.append(question + (f, ))
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
