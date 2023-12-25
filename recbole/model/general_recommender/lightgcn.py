# -*- coding: utf-8 -*-
# @Time   : 2020/8/31
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

# UPDATE:
# @Time   : 2020/9/16, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com

r"""
LightGCN
################################################

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN
"""

import numpy as np
import scipy.sparse as sp
import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from recbole.model.layers import LightGCNConv

import torch.nn as nn
from einops import rearrange
import pdb
import torch.nn.functional as F
from collections import defaultdict
import pandas as pd

class LightGCN(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly 
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.require_pow = config['require_pow']
        self.ULRec = config['ULRec']
        self.alpha = config['alpha']
        self.tau = config['Tau']
        self.dropout = nn.Dropout(p=0.2,inplace=True)

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim).float()
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim).float()
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def dot_product(self, q, k, v):
        kv = torch.einsum("ld,lm->dm", k, v)
        qkv = torch.einsum("ld,dm->lm", q, kv)
        return qkv

    def Flow_UCNRec(self, queries, keys, values):
        ## input: B (L or S) D; output: B L D
        ## Note: queries, keys, values are not projected yet
        ## 1. Linear projection
        # print("queries",queries.size()) # torch.Size([2627, 64])
        # exit()
        L, _ = queries.shape  # B=1
        # S, _ = keys.shape
        # queries = self.query_projection(queries).view(B, L, self.n_heads, -1)
        # keys = self.key_projection(keys).view(B, S, self.n_heads, -1)
        # values = self.value_projection(values).view(B, S, self.n_heads, -1)
        self.eps = 1e-2
        queries = queries.transpose(0, 1)
        keys = keys.transpose(0, 1)
        values = values.transpose(0, 1)
        # 2. Non-negative projection
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        ## 3. Flow-Attention
        # (1) Calculate incoming and outgoing flow
        # print("queries",queries.size())
        # print("keys", keys.size())
        # exit()
        # print("keys.sum(dim=1) + self.eps",keys.sum(dim=1).size()) torch.Size([64])
        # print("queries",queries.size()) # torch.Size([64, 2627])
        # print("keys.sum(dim=0)", keys.sum(dim=0).size())
        # exit()
        sink_incoming = 1.0 / (torch.einsum("ld,d->l", queries + self.eps, keys.sum(dim=0) + self.eps))
        source_outgoing = 1.0 / (torch.einsum("ld,d->l", keys + self.eps, queries.sum(dim=0) + self.eps))
        # (2) conservation refine for source and sink
        conserved_sink = torch.einsum("ld,d->l", queries + self.eps,
                                      (keys * source_outgoing[:, None]).sum(dim=0) + self.eps)
        conserved_source = torch.einsum("ld,d->l", keys + self.eps,
                                        (queries * sink_incoming[:, None]).sum(dim=0) + self.eps)
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
        # (3) Competition & Allocation
        sink_allocation = torch.sigmoid(conserved_sink * (float(queries.shape[1]) / float(keys.shape[1])))
        source_competition = torch.softmax(conserved_source, dim=-1) * float(keys.shape[1])
        # (4) dot product
        x = (self.dot_product(queries * sink_incoming[:, None],  # for value normalization
                              keys,
                              values * source_competition[:, None])  # competition
             * sink_allocation[:, None]).transpose(0, 1)  # allocation
        ## (5) Final projection
        # x = x.reshape(B, L, -1)
        x = x.reshape(L, -1)
        # print("x",x.size())
        # exit()
        # x = self.out_projection(x)
        # x = self.dropout(x)
        return x

    def act_fun(self, act_fun):
        if act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return 1 + F.elu

    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

        return nn.Parameter(index, requires_grad=False)

    def UL_Rec(self, queries, keys, values):
        tgt_len, _ = queries.size()
        src_len = keys.size(0)
        # activation
        q = queries
        k = keys
        v = values
        q = F.relu(q)
        k = F.relu(k)
        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # (N * h, L, 2 * d)
        q_ = torch.cat(
            [q * torch.sin(weight_index[:, :tgt_len, :] / m), q * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        # (N * h, S, 2 * d)
        k_ = torch.cat(
            [k * torch.sin(weight_index[:, :src_len, :] / m), k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)
        # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
        q_ = torch.squeeze(q_)
        k_ = torch.squeeze(k_)
        # print("q_", k_.size())
        # print("k_",k_.size())
        # print("k_", v.size())
        # exit()
        kv_ = torch.einsum('ld,lm->dm', k_, v)
        # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
        eps = 1e-6
        z_ = 1 / torch.clamp_min(torch.einsum('ld,d->l', q_, torch.sum(k_, axis=0)), eps)
        # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
        attn_output = torch.einsum('ld,dm,l->lm', q_, kv_, z_)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        # print("attn_output",attn_output.size())
        # exit()
        # attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        return attn_output

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            if self.ULRec == 'Yes':
                x = all_embeddings
                alpha = self.alpha # 0.8
                tau = self.tau  # 0.2  # 0.5
                # norm_x = nn.functional.normalize(x, dim=1)
                if tau == 0.2:
                    norm_x = nn.functional.normalize(x, dim=1)
                    sim = norm_x @ norm_x.T / tau
                    sim = nn.functional.softmax(sim, dim=1)
                    x_neg = sim @ x
                    x = (1 + alpha) * x - alpha * x_neg
                if tau == 1:
                    norm_x = nn.functional.normalize(x, dim=1)
                    x_neg_d = self.UL_Rec(norm_x, norm_x, norm_x)
                    x = (1 + alpha) * x - alpha * x_neg_d
                else:
                    # contra-norm -dual:n >> d
                    norm_x = nn.functional.normalize(x, dim=1)
                    sim_d = norm_x.T @ norm_x  # / 0.2
                    sim_d = nn.functional.softmax(sim_d, dim=1)
                    x_neg_d = x @ sim_d
                    x = (1 + alpha) * x - alpha * x_neg_d
                    # x = self.LayerNorm(x)
                all_embeddings = x
                embeddings_list.append(all_embeddings)
            else:
                embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        #print("lightgcn_all_embeddings",lightgcn_all_embeddings.size())
        #exit()
        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings, require_pow=self.require_pow)

        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
