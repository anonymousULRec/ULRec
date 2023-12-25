# -*- coding: utf-8 -*-
# @Time   : 2021/10/12
# @Author : Tian Zhen
# @Email  : chenyuwuxinn@gmail.com

r"""
SGL
################################################
Reference:
    Jiancan Wu et al. "SGL: Self-supervised Graph Learning for Recommendation" in SIGIR 2021.

Reference code:
    https://github.com/wujcan/SGL
"""

import numpy as np
import scipy.sparse as sp
import torch
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
import torch.nn.functional as F

import torch.nn as nn
from einops import rearrange
import pdb
from collections import defaultdict
import pandas as pd

class SGL(GeneralRecommender):
    r"""SGL is a GCN-based recommender model.

    SGL supplements the classical supervised task of recommendation with an auxiliary 
    self supervised task, which reinforces node representation learning via self-
    discrimination.Specifically,SGL generates multiple views of a node, maximizing the
    agreement between different views of the same node compared to that of other nodes.
    SGL devises three operators to generate the views — node dropout, edge dropout, and 
    random walk — that change the graph structure in different manners.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SGL, self).__init__(config, dataset)
        self._user = dataset.inter_feat[dataset.uid_field]
        self._item = dataset.inter_feat[dataset.iid_field]
        self.embed_dim = config["embedding_size"]
        self.n_layers = int(config["n_layers"])
        self.type = config["type"]
        self.drop_ratio = config["drop_ratio"]
        self.ssl_tau = config["ssl_tau"]
        self.reg_weight = config["reg_weight"]
        self.ssl_weight = config["ssl_weight"]
        self.user_embedding = torch.nn.Embedding(self.n_users, self.embed_dim)
        self.item_embedding = torch.nn.Embedding(self.n_items, self.embed_dim)
        self.LayerNorm = nn.LayerNorm(config["embedding_size"], eps=1e-05)
        self.reg_loss = EmbLoss()
        self.train_graph = self.csr2tensor(self.create_adjust_matrix(is_sub=False))
        self.restore_user_e = None
        self.restore_item_e = None
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

        # contra
        self.ULRec = config['ULRec']
        self.alpha = config['alpha']
        self.tau = config['Tau']
        self.equlation = config['Equlation']

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
        L, _ = queries.shape # B=1
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
                                      (keys * source_outgoing[ :, None]).sum(dim=0) + self.eps)
        conserved_source = torch.einsum("ld,d->l", keys + self.eps,
                                        (queries * sink_incoming[:, None]).sum(dim=0) + self.eps)
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
        # (3) Competition & Allocation
        sink_allocation = torch.sigmoid(conserved_sink * (float(queries.shape[1]) / float(keys.shape[1])))
        source_competition = torch.softmax(conserved_source, dim=-1) * float(keys.shape[1])
        # (4) dot product
        x = (self.dot_product(queries * sink_incoming[ :, None],  # for value normalization
                              keys,
                              values * source_competition[ :, None])  # competition
             * sink_allocation[ :, None]).transpose(0, 1)  # allocation
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
        tgt_len , _ = queries.size()
        src_len =keys.size(0)
        # activation
        q= queries
        k =keys
        v=values
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
        q_=torch.squeeze(q_)
        k_ = torch.squeeze(k_)
        # print("q_", k_.size())
        # print("k_",k_.size())
        # print("k_", v.size())
        # exit()
        kv_ = torch.einsum('ld,lm->dm', k_, v)
        # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
        eps =1e-6
        z_ = 1 / torch.clamp_min(torch.einsum('ld,d->l', q_, torch.sum(k_, axis=0)), eps)
        # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
        attn_output = torch.einsum('ld,dm,l->lm', q_, kv_, z_)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        # print("attn_output",attn_output.size())
        # exit()
        # attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        return attn_output
    def graph_construction(self):
        r"""Devise three operators to generate the views — node dropout, edge dropout, and random walk of a node.

        """
        self.sub_graph1 = []
        if self.type == "ND" or self.type == "ED":
            self.sub_graph1 = self.csr2tensor(self.create_adjust_matrix(is_sub=True))
        elif self.type == "RW":
            for i in range(self.n_layers):
                _g = self.csr2tensor(self.create_adjust_matrix(is_sub=True))
                self.sub_graph1.append(_g)

        self.sub_graph2 = []
        if self.type == "ND" or self.type == "ED":
            self.sub_graph2 = self.csr2tensor(self.create_adjust_matrix(is_sub=True))
        elif self.type == "RW":
            for i in range(self.n_layers):
                _g = self.csr2tensor(self.create_adjust_matrix(is_sub=True))
                self.sub_graph2.append(_g)

    def rand_sample(self, high, size=None, replace=True):
        r"""Randomly discard some points or edges.

        Args:
            high (int): Upper limit of index value
            size (int): Array size after sampling

        Returns:
            numpy.ndarray: Array index after sampling, shape: [size]
        """

        a = np.arange(high)
        sample = np.random.choice(a, size=size, replace=replace)
        return sample

    def create_adjust_matrix(self, is_sub: bool):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.If it is a subgraph, it may be processed by
        node dropout or edge dropout.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            csr_matrix of the normalized interaction matrix.
        """
        matrix = None
        if not is_sub:
            ratings = np.ones_like(self._user, dtype=np.float32)
            matrix = sp.csr_matrix((ratings, (self._user, self._item + self.n_users)),
                                   shape=(self.n_users + self.n_items, self.n_users + self.n_items))
        else:
            if self.type == "ND":
                drop_user = self.rand_sample(self.n_users, size=int(self.n_users * self.drop_ratio), replace=False)
                drop_item = self.rand_sample(self.n_items, size=int(self.n_items * self.drop_ratio), replace=False)
                R_user = np.ones(self.n_users, dtype=np.float32)
                R_user[drop_user] = 0.
                R_item = np.ones(self.n_items, dtype=np.float32)
                R_item[drop_item] = 0.
                R_user = sp.diags(R_user)
                R_item = sp.diags(R_item)
                R_G = sp.csr_matrix((np.ones_like(self._user, dtype=np.float32), (self._user, self._item)),
                                    shape=(self.n_users, self.n_items))
                res = R_user.dot(R_G)
                res = res.dot(R_item)

                user, item = res.nonzero()
                ratings = res.data
                matrix = sp.csr_matrix((ratings, (user, item + self.n_users)), shape=(self.n_users + self.n_items, self.n_users + self.n_items))

            elif self.type == "ED" or self.type == "RW":
                keep_item = self.rand_sample(
                    len(self._user), size=int(len(self._user) * (1 - self.drop_ratio)), replace=False
                )
                user = self._user[keep_item]
                item = self._item[keep_item]

                matrix = sp.csr_matrix((np.ones_like(user), (user, item + self.n_users)),
                                       shape=(self.n_users + self.n_items, self.n_users + self.n_items))

        matrix = matrix + matrix.T
        D = np.array(matrix.sum(axis=1)) + 1e-7
        D = np.power(D, -0.5).flatten()
        D = sp.diags(D)
        return D.dot(matrix).dot(D)

    def csr2tensor(self, matrix: sp.csr_matrix):
        r"""Convert csr_matrix to tensor.

        Args:
            matrix (scipy.csr_matrix): Sparse matrix to be converted.

        Returns:
            torch.sparse.FloatTensor: Transformed sparse matrix.
        """
        matrix = matrix.tocoo()
        x = torch.sparse.FloatTensor(
            torch.LongTensor(np.array([matrix.row, matrix.col])),
            torch.FloatTensor(matrix.data.astype(np.float32)), matrix.shape
        ).to(self.device)
        return x

    def forward(self, graph):
        main_ego = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_ego = [main_ego]
        if isinstance(graph, list):
            for sub_graph in graph:
                main_ego = torch.sparse.mm(sub_graph, main_ego)
                all_ego.append(main_ego)
        else:
            for i in range(self.n_layers):
                main_ego = torch.sparse.mm(graph, main_ego)
                if self.ULRec is True:
                    x = main_ego
                    alpha = self.alpha  # 0.8
                    tau = self.tau  # 0.2  # 0.5
                    eq = self.equlation
                    # norm_x = nn.functional.normalize(x, dim=1)
                    # if tau == 0.2:
                    if eq == 6:
                        norm_x = nn.functional.normalize(x, dim=1)
                        sim = norm_x @ norm_x.T / 0.2
                        sim = nn.functional.softmax(sim, dim=1)
                        x_neg = sim @ x
                        x = (1 + alpha) * x - alpha * x_neg
                    if eq == 7:
                        norm_x = nn.functional.normalize(x, dim=1)
                        x_neg_d = self.UL_Rec(norm_x, norm_x, norm_x)
                        x = (1 + alpha) * x - alpha * x_neg_d
                    else:
                        norm_x = nn.functional.normalize(x, dim=1)
                        sim_d = norm_x.T @ norm_x  # / 0.2
                        sim_d = nn.functional.softmax(sim_d, dim=1)
                        x_neg_d = x @ sim_d
                        x = (1 + alpha) * x - alpha * x_neg_d
                    main_ego = x
                    all_ego.append(main_ego)
                else:
                    all_ego.append(main_ego)

        all_ego = torch.stack(all_ego, dim=1)
        all_ego = torch.mean(all_ego, dim=1, keepdim=False)
        user_emd, item_emd = torch.split(all_ego, [self.n_users, self.n_items], dim=0)

        return user_emd, item_emd

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        
        user_list = interaction[self.USER_ID]
        pos_item_list = interaction[self.ITEM_ID]
        neg_item_list = interaction[self.NEG_ITEM_ID]
        user_emd, item_emd = self.forward(self.train_graph)
        user_sub1, item_sub1 = self.forward(self.sub_graph1)
        user_sub2, item_sub2 = self.forward(self.sub_graph2)
        total_loss = self.calc_bpr_loss(user_emd,item_emd,user_list,pos_item_list,neg_item_list) + \
            self.calc_ssl_loss(user_list,pos_item_list,user_sub1,user_sub2,item_sub1,item_sub2)
        return total_loss

    def calc_bpr_loss(self, user_emd, item_emd, user_list, pos_item_list, neg_item_list):
        r"""Calculate the the pairwise Bayesian Personalized Ranking (BPR) loss and parameter regularization loss.

        Args:
            user_emd (torch.Tensor): Ego embedding of all users after forwarding.
            item_emd (torch.Tensor): Ego embedding of all items after forwarding.
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.
            neg_item_list (torch.Tensor): List of negative examples.

        Returns:
            torch.Tensor: Loss of BPR tasks and parameter regularization.
        """
        u_e = user_emd[user_list]
        pi_e = item_emd[pos_item_list]
        ni_e = item_emd[neg_item_list]
        p_scores = torch.mul(u_e, pi_e).sum(dim=1)
        n_scores = torch.mul(u_e, ni_e).sum(dim=1)

        l1 = torch.sum(-F.logsigmoid(p_scores - n_scores))

        u_e_p = self.user_embedding(user_list)
        pi_e_p = self.item_embedding(pos_item_list)
        ni_e_p = self.item_embedding(neg_item_list)

        l2 = self.reg_loss(u_e_p, pi_e_p, ni_e_p)

        return l1 + l2 * self.reg_weight

    def calc_ssl_loss(self, user_list, pos_item_list, user_sub1, user_sub2, item_sub1, item_sub2):
        r"""Calculate the loss of self-supervised tasks.

        Args:
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.
            user_sub1 (torch.Tensor): Ego embedding of all users in the first subgraph after forwarding.
            user_sub2 (torch.Tensor): Ego embedding of all users in the second subgraph after forwarding.
            item_sub1 (torch.Tensor): Ego embedding of all items in the first subgraph after forwarding.
            item_sub2 (torch.Tensor): Ego embedding of all items in the second subgraph after forwarding.

        Returns:
            torch.Tensor: Loss of self-supervised tasks.
        """

        u_emd1 = F.normalize(user_sub1[user_list], dim=1)
        u_emd2 = F.normalize(user_sub2[user_list], dim=1)
        all_user2 = F.normalize(user_sub2,dim=1)
        v1 = torch.sum(u_emd1 * u_emd2, dim=1)
        v2 = u_emd1.matmul(all_user2.T)
        v1 = torch.exp(v1 / self.ssl_tau)
        v2 = torch.sum(torch.exp(v2 / self.ssl_tau), dim=1)
        ssl_user = -torch.sum(torch.log(v1 / v2))

        i_emd1 = F.normalize(item_sub1[pos_item_list], dim=1)
        i_emd2 = F.normalize(item_sub2[pos_item_list], dim=1)
        all_item2 = F.normalize(item_sub2,dim=1)
        v3 = torch.sum(i_emd1 * i_emd2, dim=1)
        v4 = i_emd1.matmul(all_item2.T)
        v3 = torch.exp(v3 / self.ssl_tau)
        v4 = torch.sum(torch.exp(v4 / self.ssl_tau), dim=1)
        ssl_item = -torch.sum(torch.log(v3 / v4))

        return (ssl_item + ssl_user) * self.ssl_weight

    def predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward(self.train_graph)

        user = self.restore_user_e[interaction[self.USER_ID]]
        item = self.restore_item_e[interaction[self.ITEM_ID]]
        return torch.sum(user * item, dim=1)

    def full_sort_predict(self, interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward(self.train_graph)
        
        user = self.restore_user_e[interaction[self.USER_ID]]
        return user.matmul(self.restore_item_e.T)

    def train(self, mode: bool = True):
        r"""Override train method of base class.The subgraph is reconstructed each time it is called.

        """
        T = super().train(mode=mode)
        if mode:
            self.graph_construction()
        return T
