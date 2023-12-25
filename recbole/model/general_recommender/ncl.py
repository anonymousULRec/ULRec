# -*- coding: utf-8 -*-

r"""
NCL
################################################

Reference:
    Zihan Lin*, Changxin Tian*, Yupeng Hou*, Wayne Xin Zhao. "Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning." in WWW 2022.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

import torch.nn as nn
from einops import rearrange
import pdb
from collections import defaultdict
import pandas as pd


class NCL(GeneralRecommender):
    r"""NCL is a neighborhood-enriched contrastive learning paradigm for graph collaborative filtering.
    Both structural and semantic neighbors are explicitly captured as contrastive learning objects.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(NCL, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type: the embedding size of the base model
        self.n_layers = config['n_layers']          # int type: the layer num of the base model
        self.reg_weight = config['reg_weight']      # float32 type: the weight decay for l2 normalization

        self.ssl_temp = config['ssl_temp']
        self.ssl_reg = config['ssl_reg']
        self.hyper_layers = config['hyper_layers']

        self.alpha = config['alpha']

        self.proto_reg = config['proto_reg']
        self.k = config['num_clusters']

        # contra
        self.UL_Rec = config['UL_Rec']
        self.alpha = config['alpha']
        self.tau = config['Tau']

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        self.norm_adj_mat = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None

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

    def e_step(self):
        user_embeddings = self.user_embedding.weight.detach().cpu().numpy()
        item_embeddings = self.item_embedding.weight.detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x
        """
        import faiss
        kmeans = faiss.Kmeans(d=self.latent_dim, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster

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
        self.diag = torch.from_numpy(diag).to(self.device)
        D = sp.diags(diag)
        L = D @ A @ D
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
        for layer_idx in range(max(self.n_layers, self.hyper_layers*2)):
            all_embeddings = torch.sparse.mm(self.norm_adj_mat, all_embeddings)
            if self.UL_Rec == 'Yes':
                x = all_embeddings
                alpha = self.alpha  # 0.8
                tau = self.tau  # 0.2  # 0.5
                if tau == 0.2:
                    norm_x = nn.functional.normalize(x, dim=1)
                    sim = norm_x @ norm_x.T / tau
                    sim = nn.functional.softmax(sim, dim=1)
                    x_neg = sim @ x
                    x = (1 + alpha) * x - alpha * x_neg
                if tau == 0:
                    # flow
                    norm_x = nn.functional.normalize(x, dim=1)
                    x_neg_d = self.Flow_UCNRec(norm_x, norm_x, norm_x)
                    x = (1 + alpha) * x - alpha * x_neg_d
                    # print("x",x.size())
                if tau == 1:
                    # flow
                    norm_x = nn.functional.normalize(x, dim=1)
                    x_neg_d = self.UL_Rec(norm_x, norm_x, norm_x)
                    x = (1 + alpha) * x - alpha * x_neg_d
                else:
                    norm_x = nn.functional.normalize(x, dim=1)
                    sim_d = norm_x.T @ norm_x  # / 0.2
                    sim_d = nn.functional.softmax(sim_d, dim=1)
                    x_neg_d = x @ sim_d
                    x = (1 + alpha) * x - alpha * x_neg_d
                    # x = self.LayerNorm(x)
                # print("sim_d", x.size())
                # exit()

                all_embeddings = x
                embeddings_list.append(all_embeddings)
            else:
                # print("alpha", 0000)
                # # print("tau", 1111)
                # exit()
                embeddings_list.append(all_embeddings)
            #embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list[:self.n_layers+1], dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, embeddings_list

    def ProtoNCE_loss(self, node_embedding, user, item):
        user_embeddings_all, item_embeddings_all = torch.split(node_embedding, [self.n_users, self.n_items])

        user_embeddings = user_embeddings_all[user]     # [B, e]
        norm_user_embeddings = F.normalize(user_embeddings)

        user2cluster = self.user_2cluster[user]     # [B,]
        user2centroids = self.user_centroids[user2cluster]   # [B, e]
        pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.matmul(norm_user_embeddings, self.user_centroids.transpose(0, 1))
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        item_embeddings = item_embeddings_all[item]
        norm_item_embeddings = F.normalize(item_embeddings)

        item2cluster = self.item_2cluster[item]  # [B, ]
        item2centroids = self.item_centroids[item2cluster]  # [B, e]
        pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.matmul(norm_item_embeddings, self.item_centroids.transpose(0, 1))
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        proto_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

    def ssl_layer_loss(self, current_embedding, previous_embedding, user, item):
        current_user_embeddings, current_item_embeddings = torch.split(current_embedding, [self.n_users, self.n_items])
        previous_user_embeddings_all, previous_item_embeddings_all = torch.split(previous_embedding, [self.n_users, self.n_items])

        current_user_embeddings = current_user_embeddings[user]
        previous_user_embeddings = previous_user_embeddings_all[user]
        norm_user_emb1 = F.normalize(current_user_embeddings)
        norm_user_emb2 = F.normalize(previous_user_embeddings)
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        current_item_embeddings = current_item_embeddings[item]
        previous_item_embeddings = previous_item_embeddings_all[item]
        norm_item_emb1 = F.normalize(current_item_embeddings)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(previous_item_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        center_embedding = embeddings_list[0]
        context_embedding = embeddings_list[self.hyper_layers * 2]

        ssl_loss = self.ssl_layer_loss(context_embedding, center_embedding, user, pos_item)
        proto_loss = self.ProtoNCE_loss(center_embedding, user, pos_item)

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        return mf_loss + self.reg_weight * reg_loss, ssl_loss, proto_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, embedding_list = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
