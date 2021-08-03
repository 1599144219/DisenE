import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

CUDA = torch.cuda.is_available()  # checking cuda availability


class ConvKB(nn.Module):
    def __init__(self, entity_emb, relation_emb, config=None):
        '''
        '''

        super(ConvKB, self).__init__()
        self.do_normalize = config.do_normalize

        self.entity_embeddings = nn.Parameter(entity_emb)
        self.relation_embeddings = nn.Parameter(relation_emb)

        self.num_nodes = entity_emb.shape[0]
        self.entity_in_dim = entity_emb.shape[1]

        # Properties of Relations
        self.num_relation = relation_emb.shape[0]
        self.relation_dim = relation_emb.shape[1]

        self.conv_layer = nn.Conv2d(1, config.out_channels, (1, 3))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(config.dropout)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((self.entity_in_dim) * config.out_channels, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

        # loss function
        self.loss = torch.nn.SoftMarginLoss()

    def forward(self, batch_inputs, batch_labels=None):
        if self.do_normalize:
            self.entity_embeddings.data = F.normalize(
                self.entity_embeddings.data, p=2, dim=1).detach()

        conv_input = torch.cat((self.entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)

        batch_size, length, dim = conv_input.size()
        conv_input = conv_input.transpose(1, 2)
        conv_input = conv_input.unsqueeze(1)

        out_conv = self.dropout(self.non_linearity(self.conv_layer(conv_input)))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)

        if batch_labels is not None:
            loss = self.loss(output.view(-1), batch_labels.view(-1))
            return loss, 0

        return output, 0


class TransE(nn.Module):
    def __init__(self, entity_emb, relation_emb, config=None):
        '''
        '''

        super(TransE, self).__init__()
        self.do_normalize = config.do_normalize
        self.valid_invalid_ratio = config.valid_invalid_ratio
        self.margin = config.margin

        self.entity_embeddings = nn.Parameter(entity_emb)
        self.relation_embeddings = nn.Parameter(relation_emb)

        self.num_nodes = entity_emb.shape[0]
        self.entity_in_dim = entity_emb.shape[1]

        # Properties of Relations
        self.num_relation = relation_emb.shape[0]
        self.relation_dim = relation_emb.shape[1]

        # loss function
        self.loss = nn.MarginRankingLoss(margin=self.margin, reduction='none')

    def forward(self, batch_inputs, batch_labels=None, batch_loss_weight=None):
        if self.do_normalize:
            self.entity_embeddings.data = F.normalize(
                self.entity_embeddings.data, p=2, dim=1).detach()

        len_pos_triples = int(batch_inputs.size(0) / (int(self.valid_invalid_ratio) + 1))
        pos_triples = batch_inputs[:len_pos_triples]
        neg_triples = batch_inputs[len_pos_triples:]
        # print("batch", batch_inputs.size(), self.valid_invalid_ratio, len_pos_triples, neg_triples.size())

        pos_triples = pos_triples.repeat(int(self.valid_invalid_ratio), 1)

        pos_head = self.entity_embeddings[pos_triples[:, 0], :]
        pos_rel = self.relation_embeddings[pos_triples[:, 1], :]
        pos_tail = self.entity_embeddings[pos_triples[:, 2], :]

        neg_head = self.entity_embeddings[neg_triples[:, 0], :]
        neg_rel = self.relation_embeddings[neg_triples[:, 1], :]
        neg_tail = self.entity_embeddings[neg_triples[:, 2], :]

        pos_x = pos_head + pos_rel - pos_tail
        neg_x = neg_head + neg_rel - neg_tail
        pos_norm = torch.norm(pos_x, p=1, dim=1)
        neg_norm = torch.norm(neg_x, p=1, dim=1)

        y = -torch.ones(int(self.valid_invalid_ratio) * len_pos_triples).cuda()
        output = (pos_norm, neg_norm, y)

        if batch_labels is not None:
            sep_loss = self.loss(pos_norm, neg_norm, y)
            if batch_loss_weight is not None:
                loss = torch.mean(sep_loss * batch_loss_weight.view(-1))
            else:
                loss = torch.mean(sep_loss)
            return loss, 0

        return output, 0

    def test(self, batch_inputs):
        head = self.entity_embeddings[batch_inputs[:, 0], :]
        rel = self.relation_embeddings[batch_inputs[:, 1], :]
        tail = self.entity_embeddings[batch_inputs[:, 2], :]

        x = head + rel - tail
        score = torch.norm(x, p=1, dim=1)

        # y = -torch.ones(int(batch_inputs.size(0))).cuda()
        score = - score

        return score, 0


class DisenE(nn.Module):
    def __init__(self, entity_emb, relation_emb, config=None):

        super(DisenE, self).__init__()
        self.do_normalize = config.do_normalize  # 0
        self.K = config.k_factors  # 6

        self.entity_embeddings = nn.Parameter(entity_emb)  # 可训练
        self.relation_embeddings = nn.Parameter(relation_emb)

        self.num_nodes = entity_emb.shape[0]  # 实体总数
        self.ent_emb_s = entity_emb.shape[1]  # 实体嵌入维度

        # Properties of Relations
        self.num_relation = relation_emb.shape[0]  # 关系总数
        self.emb_s = relation_emb.shape[1]  # 关系嵌入维度

        self.fc1 = torch.nn.Linear(self.emb_s * 3, 1)
        self.non_linearity = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

        self.conv_layer = nn.Conv2d(1, config.out_channels, (3, 3),
                                    padding=(1, 0))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout_conv = nn.Dropout(config.dropout)

        self.fc3 = torch.nn.Linear(self.emb_s * config.out_channels, 1)

        nn.init.xavier_uniform_(self.fc1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.fc3.weight, gain=1.414)

        # loss function
        self.loss = torch.nn.SoftMarginLoss()

    def forward(self, batch_inputs, batch_labels=None):
        if self.do_normalize:
            self.entity_embeddings.data = F.normalize(
                self.entity_embeddings.data, p=2, dim=1).detach()

        e1_embedded = self.entity_embeddings[batch_inputs[:, 0], :].view(-1, self.K, self.emb_s)
        rel_embedded = self.relation_embeddings[batch_inputs[:, 1]].unsqueeze(1)

        e2_embedded = self.entity_embeddings[batch_inputs[:, 2], :].view(-1, self.K, self.emb_s)
        ex_rel_emb = rel_embedded.expand(-1, self.K, self.emb_s)

        # calculate k attention
        e1_rel_e2 = torch.cat([e1_embedded, ex_rel_emb, e2_embedded], 2)  # [b_s, k, emb_s * 3]
        tmp = self.non_linearity(self.fc1(e1_rel_e2).squeeze(-1))  # [b_s, k]
        att_e1_e2 = self.softmax(tmp)

        # inside calculation
        x = e1_rel_e2.view(e1_rel_e2.size(0) * self.K, 3, self.emb_s)
        conv_input = x.transpose(1, 2)
        conv_input = conv_input.unsqueeze(1)
        x = self.dropout_conv(self.non_linearity(self.conv_layer(conv_input)))  # [b_s * k, channel, emb_s, 1]
        x = x.view(x.size(0), -1)  # [b_s * k, emb_s * channel]
        x = x.view(e1_rel_e2.size(0), self.K, -1)  # [b_s, k, emb_s * channel]

        e1_e2_atted = torch.mul(att_e1_e2.unsqueeze(-1), x)
        x4 = torch.sum(e1_e2_atted, 1)  # [b_s, emb_s *3]
        output = self.fc3(x4)

        if batch_labels is not None:
            loss = self.loss(output.view(-1), batch_labels.view(-1))
            return loss, att_e1_e2

        return output, att_e1_e2


class DisenE_Trans(nn.Module):
    def __init__(self, entity_emb, relation_emb, config=None):
        '''
        entity_emb : 初始化的实体embedding
        relation_emb : 初始化的关系embedding
        config : args
        '''
        super(DisenE_Trans, self).__init__()
        self.do_normalize = config.do_normalize  # 归一化参数
        self.K = config.k_factors  # 划分的因子数
        self.valid_invalid_ratio = config.valid_invalid_ratio  # 正样本对应负样本的比例

        self.entity_embeddings = nn.Parameter(entity_emb)
        self.relation_embeddings = nn.Parameter(relation_emb)

        self.emb_s = relation_emb.shape[1]  # 关系的嵌入维度 100

        self.fc1 = torch.nn.Linear(self.emb_s * 3, 1)
        self.non_linearity = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

        nn.init.xavier_uniform_(self.fc1.weight, gain=1.414)

        # loss function
        self.margin = config.margin
        self.loss = nn.MarginRankingLoss(margin=self.margin, reduction='none')

    def forward(self, batch_inputs, batch_labels=None):
        if self.do_normalize:
            # dim=1 对行操作
            self.entity_embeddings.data = F.normalize(
                self.entity_embeddings.data, p=2, dim=1).detach()

        # 正样本的三元组个数
        len_pos_triples = int(batch_inputs.size(0) / (int(self.valid_invalid_ratio) + 1))

        # [128*(40+1), 6, 100]
        head = self.entity_embeddings[batch_inputs[:, 0], :].view(-1, self.K, self.emb_s)
        print(head[0][0] == head[0][1])

        # [5248, 100] -> [5248, 1, 100] -> [5248, 6, 100]
        rel = self.relation_embeddings[batch_inputs[:, 1]]
        ex_rel_emb = rel.unsqueeze(1).expand(-1, self.K, self.emb_s)

        tail = self.entity_embeddings[batch_inputs[:, 2], :].view(-1, self.K, self.emb_s)

        # calculate k attention  [5248, 6, 100*3]
        e1_rel_e2 = torch.cat([head, ex_rel_emb, tail], 2)
        # [5248, 6]
        tmp = self.non_linearity(self.fc1(e1_rel_e2).squeeze(-1))
        att = self.softmax(tmp)

        # [128*40, 6, 100] 正样本
        pos_head = head[:len_pos_triples].repeat(int(self.valid_invalid_ratio), 1, 1)

        pos_tail = tail[:len_pos_triples].repeat(int(self.valid_invalid_ratio), 1, 1)
        pos_rel = ex_rel_emb[:len_pos_triples].repeat(int(self.valid_invalid_ratio), 1, 1)

        pos_att = att[:len_pos_triples].repeat(int(self.valid_invalid_ratio), 1)

        # [5120, 6, 100]  负样本
        neg_head = head[len_pos_triples:]
        neg_tail = tail[len_pos_triples:]
        neg_rel = ex_rel_emb[len_pos_triples:]
        neg_att = att[len_pos_triples:]

        pos_x = pos_head + pos_rel - pos_tail
        neg_x = neg_head + neg_rel - neg_tail

        pos_x = torch.sum(torch.mul(pos_att.unsqueeze(-1), pos_x), 1)  # [5120,100]
        neg_x = torch.sum(torch.mul(neg_att.unsqueeze(-1), neg_x), 1)

        pos_norm = torch.norm(pos_x, p=1, dim=1)
        neg_norm = torch.norm(neg_x, p=1, dim=1)

        y = -torch.ones(int(self.valid_invalid_ratio) * len_pos_triples).cuda()
        output = (pos_norm, neg_norm, y)

        if batch_labels is not None:
            # loss(x,y) = max(0,-y*(x1-x2)+margin)
            sep_loss = self.loss(pos_norm, neg_norm, y)
            loss = torch.mean(sep_loss)
            return loss, att
        return output, att

    def test(self, batch_inputs):
        head_ori = self.entity_embeddings[batch_inputs[:, 0], :].view(-1, self.K, self.emb_s)
        rel = self.relation_embeddings[batch_inputs[:, 1], :]
        tail_ori = self.entity_embeddings[batch_inputs[:, 2], :].view(-1, self.K, self.emb_s)
        ex_rel = rel.unsqueeze(1).expand(-1, self.K, self.emb_s)

        # calculate k attention
        e1_rel_e2 = torch.cat([head_ori, ex_rel, tail_ori], 2)  # [b_s, k, emb_s * 3]
        tmp = self.non_linearity(self.fc1(e1_rel_e2).squeeze(-1))  # [b_s, k]
        att = self.softmax(tmp)

        x = head_ori + ex_rel - tail_ori
        x = torch.sum(torch.mul(att.unsqueeze(-1), x), 1)

        score = torch.norm(x, p=1, dim=1)

        # y = -torch.ones(int(batch_inputs.size(0))).cuda()
        # score = score * y
        score = -score

        return score, att
