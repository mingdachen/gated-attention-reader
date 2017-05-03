#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from builtins import range

import torch
import torch.nn as nn
from utils.misc import prepare_input
from utils.model_helper import *


class GAReader(nn.Module):
    def __init__(self, n_layers, vocab_size, n_chars, dropout,
                 gru_size, embed_init, embed_dim, train_emb, char_dim,
                 use_feat, gating_fn):
        super(GAReader, self).__init__()
        self.gru_size = gru_size
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.train_emb = train_emb
        self.char_dim = char_dim
        self.n_chars = n_chars
        self.use_feat = use_feat
        self.gating_fn = gating_fn
        self.n_vocab = vocab_size
        self.use_chars = self.char_dim != 0

        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        if use_feat:
            self.feat_embed = nn.Embedding(2, 2)
        if self.use_chars:
            self.char_embed = nn.Embedding(self.n_chars, self.char_dim)
            # (seq_len, batch, hidden_size * num_directions)
            self.char_gru = nn.GRU(input_size=self.char_dim,
                                   hidden_size=self.char_dim,
                                   dropout=dropout,
                                   batch_first=True,
                                   bidirectional=True)
            # (batch, seq_len, embed_dim)
            self.char_fw = nn.Linear(
                self.char_dim, self.embed_dim // 2)
            self.char_bk = nn.Linear(
                self.char_dim, self.embed_dim // 2)
        if embed_init is not None:
            self.embed.weight.data.copy_(torch.from_numpy(embed_init))
        if not train_emb:
            self.embed.weight.requires_grad = False
        self.main_doc_layers = nn.ModuleList()
        self.main_qry_layers = nn.ModuleList()
        if self.use_chars:
            main_input_feat = embed_dim + self.embed_dim // 2
        else:
            main_input_feat = embed_dim
        for layer in range(n_layers - 1):
            layer_doc = nn.GRU(
                input_size=main_input_feat if layer == 0 else 2 * self.gru_size,
                hidden_size=self.gru_size,
                batch_first=True,
                bidirectional=True)
            layer_qry = nn.GRU(
                input_size=main_input_feat,
                hidden_size=self.gru_size,
                batch_first=True,
                bidirectional=True)
            self.main_doc_layers.append(layer_doc)
            self.main_qry_layers.append(layer_qry)
        if use_feat:
            final_input_feat = self.gru_size * 2 + 2
        else:
            final_input_feat = self.gru_size * 2
        # final layer
        self.final_doc_layer = nn.GRU(
            input_size=final_input_feat,
            hidden_size=self.gru_size,
            batch_first=True,
            bidirectional=True)
        self.final_qry_layer = nn.GRU(
            input_size=main_input_feat,
            hidden_size=self.gru_size,
            batch_first=True,
            bidirectional=True)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, doc, doc_char, qry, qry_char, target,
                doc_mask, qry_mask, token, token_mask, cand,
                cand_mask, cloze, fnames):
        doc_embed = self.embed(doc.long())
        qry_embed = self.embed(qry.long())
        if self.use_chars:
            token_embed = self.char_embed(token.long())
            gru_out, gru_out_len = gru(token_embed, token_mask, self.char_gru)
            out_last_idx = (gru_out_len - 1)\
                .view(-1, 1).expand(
                    gru_out.size(0), gru_out.size(2)).unsqueeze(1)
            out_last = gru_out.gather(1, out_last_idx.long()).squeeze()
            token_fw_out = out_last[:, :self.char_dim]
            token_bk_out = out_last[:, self.char_dim:]
            token_fw_out = self.char_fw(token_fw_out)
            token_bk_out = self.char_bk(token_bk_out)
            merge_token_out = token_fw_out + token_bk_out
            doc_char_embed = merge_token_out.index_select(
                0, doc_char.long().view([-1])).view(
                list(doc_char.size()) + [self.embed_dim // 2])
            qry_char_embed = merge_token_out.index_select(
                0, qry_char.long().view([-1])).view(
                list(qry_char.size()) + [self.embed_dim // 2])
            doc_embed = torch.cat([doc_embed, doc_char_embed], dim=-1)
            qry_embed = torch.cat([qry_embed, qry_char_embed], dim=-1)
        for layer in range(self.n_layers - 1):
            doc_bi_embed, _ = gru(
                doc_embed, doc_mask, self.main_doc_layers[layer])
            qry_bi_embed, _ = gru(
                qry_embed, qry_mask, self.main_qry_layers[layer])
            interacted = pairwise_interaction(doc_bi_embed, qry_bi_embed)
            doc_inter_embed = gated_attention(
                doc_bi_embed, qry_bi_embed, interacted, qry_mask,
                gating_fn=self.gating_fn)
            doc_embed = self.dropout(doc_inter_embed)
        if self.use_feat:
            feat = prepare_input(doc, qry)
            feat_embed = self.feat_embed(feat)
            doc_embed = torch.cat([doc_embed, feat_embed], dim=-1)
        doc_final_embed, _ = gru(
            doc_embed, doc_mask, self.final_doc_layer)
        qry_final_embed, _ = gru(
            qry_embed, qry_mask, self.final_qry_layer)
        pred = attention_sum(
            doc_final_embed, qry_final_embed, cand, cloze, cand_mask)
        loss = self.criterion(pred, target.long())
        prob, pred_ans = torch.max(pred, dim=1)
        acc = torch.sum(torch.eq(pred_ans, target.long()))
        return loss, acc
