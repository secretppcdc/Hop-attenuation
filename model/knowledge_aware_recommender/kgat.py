# -*- coding: utf-8 -*-
# @Time   : 2020/9/15
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
KGAT
##################################################
Reference:
    Xiang Wang et al. "KGAT: Knowledge Graph Attention Network for Recommendation." in SIGKDD 2019.

Reference code:
    https://github.com/xiangwang1223/knowledge_graph_attention_network
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class Aggregator(nn.Module):
    """ GNN Aggregator layer
    """

    def __init__(self, input_dim, output_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)

        if self.aggregator_type == 'gcn':
            self.W = nn.Linear(self.input_dim, self.output_dim)
        elif self.aggregator_type == 'graphsage':
            self.W = nn.Linear(self.input_dim * 2, self.output_dim)
        elif self.aggregator_type == 'bi':
            self.W1 = nn.Linear(self.input_dim, self.output_dim)
            self.W2 = nn.Linear(self.input_dim, self.output_dim)
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()

    def forward(self, norm_matrix, ego_embeddings):
        side_embeddings = torch.sparse.mm(norm_matrix, ego_embeddings)

        if self.aggregator_type == 'gcn':
            ego_embeddings = self.activation(self.W(ego_embeddings + side_embeddings))
        elif self.aggregator_type == 'graphsage':
            ego_embeddings = self.activation(self.W(torch.cat([ego_embeddings, side_embeddings], dim=1)))
        elif self.aggregator_type == 'bi':
            add_embeddings = ego_embeddings + side_embeddings
            sum_embeddings = self.activation(self.W1(add_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = self.activation(self.W2(bi_embeddings))
            ego_embeddings = bi_embeddings + sum_embeddings
        else:
            raise NotImplementedError

        ego_embeddings = self.message_dropout(ego_embeddings)

        return ego_embeddings


class KGAT(KnowledgeRecommender):
    r"""KGAT is a knowledge-based recommendation model. It combines knowledge graph and the user-item interaction
    graph to a new graph called collaborative knowledge graph (CKG). This model learns the representations of users and
    items by exploiting the structure of CKG. It adopts a GNN-based architecture and define the attention on the CKG.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(KGAT, self).__init__(config, dataset)

        # load dataset info
        self.ckg = dataset.ckg_graph(form='dgl', value_field='relation_id')
        self.all_hs = torch.LongTensor(dataset.ckg_graph(form='coo', value_field='relation_id').row).to(self.device)
        self.all_ts = torch.LongTensor(dataset.ckg_graph(form='coo', value_field='relation_id').col).to(self.device)
        self.all_rs = torch.LongTensor(dataset.ckg_graph(form='coo', value_field='relation_id').data).to(self.device)
        self.matrix_size = torch.Size([self.n_users + self.n_entities, self.n_users + self.n_entities])

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.kg_embedding_size = config['kg_embedding_size']
        self.layers = [self.embedding_size] + config['layers']
        self.aggregator_type = config['aggregator_type']
        self.mess_dropout = config['mess_dropout']
        self.reg_weight = config['reg_weight']

        # generate intermediate data
        self.A_in = self.init_graph()  # init the attention matrix by the structure of ckg

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.kg_embedding_size)
        self.trans_w = nn.Embedding(self.n_relations, self.embedding_size * self.kg_embedding_size)
        self.aggregator_layers = nn.ModuleList()
        for idx, (input_dim, output_dim) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.aggregator_layers.append(Aggregator(input_dim, output_dim, self.mess_dropout, self.aggregator_type))
        self.tanh = nn.Tanh()
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.restore_user_e = None
        self.restore_entity_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_entity_e']

    def init_graph(self):
        r"""Get the initial attention matrix through the collaborative knowledge graph

        Returns:
            torch.sparse.FloatTensor: Sparse tensor of the attention matrix
        """
        import dgl
        adj_list = []
        for rel_type in range(1, self.n_relations, 1):
            edge_idxs = self.ckg.filter_edges(lambda edge: edge.data['relation_id'] == rel_type)
            sub_graph = dgl.edge_subgraph(self.ckg, edge_idxs, preserve_nodes=True). \
                adjacency_matrix(transpose=False, scipy_fmt='coo').astype('float')
            rowsum = np.array(sub_graph.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(sub_graph).tocoo()
            adj_list.append(norm_adj)

        final_adj_matrix = sum(adj_list).tocoo()
        indices = torch.LongTensor([final_adj_matrix.row, final_adj_matrix.col])
        values = torch.FloatTensor(final_adj_matrix.data)
        adj_matrix_tensor = torch.sparse.FloatTensor(indices, values, self.matrix_size)
        return adj_matrix_tensor.to(self.device)

    def _get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        entity_embeddings = self.entity_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, entity_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(kgat_all_embeddings, [self.n_users, self.n_entities])
        return user_all_embeddings, entity_all_embeddings

    def _get_kg_embedding(self, h, r, pos_t, neg_t):
        h_e = self.entity_embedding(h).unsqueeze(1)
        pos_t_e = self.entity_embedding(pos_t).unsqueeze(1)
        neg_t_e = self.entity_embedding(neg_t).unsqueeze(1)
        r_e = self.relation_embedding(r)
        r_trans_w = self.trans_w(r).view(r.size(0), self.embedding_size, self.kg_embedding_size)

        h_e = torch.bmm(h_e, r_trans_w).squeeze()
        pos_t_e = torch.bmm(pos_t_e, r_trans_w).squeeze()
        neg_t_e = torch.bmm(neg_t_e, r_trans_w).squeeze()

        return h_e, r_e, pos_t_e, neg_t_e

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        # get loss for training rs
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, entity_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = entity_all_embeddings[pos_item]
        neg_embeddings = entity_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        
        #---new_loss_start----#
        h = interaction[self.HEAD_ENTITY_ID]
        r = interaction[self.RELATION_ID]
        pos_t = interaction[self.TAIL_ENTITY_ID]
        neg_t = interaction[self.NEG_TAIL_ENTITY_ID]

        h_e, r_e, pos_t_e, neg_t_e = self._get_kg_embedding(h, r, pos_t, neg_t)
        pos_tail_score = ((h_e + r_e - pos_t_e) ** 2).sum(dim=1)
        neg_tail_score = ((h_e + r_e - neg_t_e) ** 2).sum(dim=1)
        #---new_loss_end----#
        
        mf_loss = self.mf_loss(pos_scores, neg_scores)# + self.mf_loss(pos_tail_score, neg_tail_score)
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)
        
        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def calculate_kg_loss(self, interaction):
        r"""Calculate the training loss for a batch data of KG.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """

        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        # get loss for training kg
        h = interaction[self.HEAD_ENTITY_ID]
        r = interaction[self.RELATION_ID]
        pos_t = interaction[self.TAIL_ENTITY_ID]
        neg_t = interaction[self.NEG_TAIL_ENTITY_ID]

        h_e, r_e, pos_t_e, neg_t_e = self._get_kg_embedding(h, r, pos_t, neg_t)
        pos_tail_score = ((h_e + r_e - pos_t_e) ** 2).sum(dim=1)
        neg_tail_score = ((h_e + r_e - neg_t_e) ** 2).sum(dim=1)
        
        #---new_loss_start----#
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, entity_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = entity_all_embeddings[pos_item]
        neg_embeddings = entity_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        #---new_loss_end----#
        gamma = nn.Parameter(
            torch.Tensor([12]), 
            requires_grad=False
        )        
        #rotatE
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(h_e, 2, dim=2)
        re_tail_pos, im_tail_pos = torch.chunk(pos_t_e, 2, dim=2)
        re_tail_neg, im_tail_neg = torch.chunk(neg_t_e, 2, dim=2)
        embedding_range = nn.Parameter(
            torch.Tensor([( self.epsilon) / 2048]), 
            requires_grad=False
        )
        phase_relation = r_e/(embedding_range.item()/pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        
        re_score_pos = re_relation * re_tail_pos + im_relation * im_tail_pos
        im_score_pos = re_relation * im_tail_pos - im_relation * re_tail_pos
        re_score_pos = re_score_pos - re_head
        im_score_pos = im_score_pos - im_head

        re_score_neg = re_relation * re_tail_neg + im_relation * im_tail_neg
        im_score_neg = re_relation * im_tail_neg - im_relation * re_tail_neg
        re_score_neg = re_score_neg - re_head
        im_score_neg = im_score_neg - im_head          

        re_score_head = re_head * re_relation - im_head * im_relation
        im_score_head = re_head * im_relation + im_head * re_relation
        re_score_head = re_score_head - re_tail_pos
        im_score_head = im_score_head - im_tail_pos

        re_score_head_neg = re_head * re_relation - im_head * im_relation
        im_score_head_neg = re_head * im_relation + im_head * re_relation
        re_score_head_neg = re_score_head - re_tail_neg
        im_score_head_neg = im_score_head - im_tail_neg

        score_pos = torch.stack([re_score_head, im_score_head], dim = 0)
        score_pos = score_pos.norm(dim = 0)
        pos_tail_score =gamma.item() - score_pos.sum(dim = 2)


        score_neg = torch.stack([re_score_head_neg, im_score_head_neg], dim = 0)
        score_neg = score_neg.norm(dim = 0)
        neg_tail_score =gamma.item() - score_neg.sum(dim = 2)

        modulus = nn.Parameter(torch.Tensor([[0.5 * embedding_range.item()]]))

        #protatE
        phase_head = h_e#/(embedding_range.item()/pi)
        phase_relation = r_e#/(embedding_range.item()/pi)
        phase_tail_pos = pos_t_e#/(embedding_range.item()/pi)
        phase_tail_neg = neg_t_e#/(embedding_range.item()/pi)


        score_pos = phase_head + (phase_relation - phase_tail_pos)
        score_neg = phase_head + (phase_relation - phase_tail_neg)
        #else:
            #score = (phase_head + phase_relation) - phase_tail

        score_pos = torch.sin(score_pos)                    
        score_pos = torch.abs(score_pos)
        score_pos =  gamma.item() - score_pos.sum(dim = 2) * modulus


        score_neg = torch.sin(score_neg)                    
        score_neg = torch.abs(score_neg)
        score_neg =  gamma.item() - score_neg.sum(dim = 2) * modulus

        # #complex
        # re_head, im_head = torch.chunk(h_e, 2, dim=2)
        # re_relation, im_relation = torch.chunk(r_e, 2, dim=2)
        # re_tail_pos, im_tail_pos = torch.chunk(pos_t_e, 2, dim=2)
        # re_tail_neg, im_tail_neg = torch.chunk(neg_t_e, 2, dim=2)

        # #if mode == 'head-batch':
        #     #re_score = re_relation * re_tail + im_relation * im_tail
        #     #im_score = re_relation * im_tail - im_relation * re_tail
        #     #score = re_head * re_score + im_head * im_score
        
        # re_score = re_head * re_relation - im_head * im_relation
        # im_score = re_head * im_relation + im_head * re_relation
        # score_pos = re_score * re_tail_pos + im_score * im_tail_pos
        # score_neg = re_score * re_tail_neg + im_score * im_tail_neg

        # score_pos = score_pos.sum(dim = 2)
        # score_neg = score_neg.sum(dim = 2)


        negative_score = F.logsigmoid(-score_neg).mean(dim = 1)
        positive_score = F.logsigmoid(score_pos).squeeze(dim = 1)

        positive_sample_loss = - positive_score.mean()
        negative_sample_loss = - negative_score.mean()
        loss_protatE = (positive_sample_loss + negative_sample_loss)/2    


        kg_loss = F.softplus(pos_tail_score- neg_tail_score).mean()# + F.softplus(pos_scores - neg_scores).mean()
        kg_reg_loss = self.reg_loss(h_e, r_e, pos_t_e, neg_t_e)
        loss = loss_protatE + self.reg_weight * kg_reg_loss

        return loss

    def generate_transE_score(self, hs, ts, r):
        r"""Calculating scores for triples in KG.

        Args:
            hs (torch.Tensor): head entities
            ts (torch.Tensor): tail entities
            r (int): the relation id between hs and ts

        Returns:
            torch.Tensor: the scores of (hs, r, ts)
        """

        all_embeddings = self._get_ego_embeddings()
        h_e = all_embeddings[hs]
        t_e = all_embeddings[ts]
        r_e = self.relation_embedding.weight[r]
        r_trans_w = self.trans_w.weight[r].view(self.embedding_size, self.kg_embedding_size)

        h_e = torch.matmul(h_e, r_trans_w)
        t_e = torch.matmul(t_e, r_trans_w)

        kg_score = torch.mul(t_e, self.tanh(h_e + r_e)).sum(dim=1)

        return kg_score

    def update_attentive_A(self):
        r"""Update the attention matrix using the updated embedding matrix

        """

        kg_score_list, row_list, col_list = [], [], []
        # To reduce the GPU memory consumption, we calculate the scores of KG triples according to the type of relation
        for rel_idx in range(1, self.n_relations, 1):
            triple_index = torch.where(self.all_rs == rel_idx)
            kg_score = self.generate_transE_score(self.all_hs[triple_index], self.all_ts[triple_index], rel_idx)
            row_list.append(self.all_hs[triple_index])
            col_list.append(self.all_ts[triple_index])
            kg_score_list.append(kg_score)
        kg_score = torch.cat(kg_score_list, dim=0)
        row = torch.cat(row_list, dim=0)
        col = torch.cat(col_list, dim=0)
        indices = torch.cat([row, col], dim=0).view(2, -1)
        # Current PyTorch version does not support softmax on SparseCUDA, temporarily move to CPU to calculate softmax
        A_in = torch.sparse.FloatTensor(indices, kg_score, self.matrix_size).cpu()
        A_in = torch.sparse.softmax(A_in, dim=1).to(self.device)
        self.A_in = A_in

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, entity_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = entity_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_entity_e is None:
            self.restore_user_e, self.restore_entity_e = self.forward()
        u_embeddings = self.restore_user_e[user]
        i_embeddings = self.restore_entity_e[:self.n_items]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))

        return scores.view(-1)
