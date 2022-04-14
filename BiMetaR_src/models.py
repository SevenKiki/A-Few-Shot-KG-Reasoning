from embedding import *
from collections import OrderedDict
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict

class RelationMetaLearner(nn.Module):
    def __init__(self, few, embed_size=100, num_hidden1=500, num_hidden2=200, out_size=100, dropout_p=0.5):
        super(RelationMetaLearner, self).__init__()
        self.embed_size = embed_size
        self.few = few
        self.out_size = out_size
        self.rel_fc1 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(2*embed_size, num_hidden1)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc2 = nn.Sequential(OrderedDict([
            ('fc',   nn.Linear(num_hidden1, num_hidden2)),
            ('bn',   nn.BatchNorm1d(few)),
            ('relu', nn.LeakyReLU()),
            ('drop', nn.Dropout(p=dropout_p)),
        ]))
        self.rel_fc3 = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(num_hidden2, out_size)),
            ('bn', nn.BatchNorm1d(few)),
        ]))
        nn.init.xavier_normal_(self.rel_fc1.fc.weight)
        nn.init.xavier_normal_(self.rel_fc2.fc.weight)
        nn.init.xavier_normal_(self.rel_fc3.fc.weight)

    def forward(self, inputs):
        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1)  # torch.Size([1024, 5, 200])
        # x = torch.mean(x, 1)    # 方法1，平均再全连接，
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        x = self.rel_fc3(x) # torch.Size([1024, 5, 100])
        x = torch.mean(x, 1)    # 原始方法：先全连接再平均平均 torch.Size([1024, 100])
        return x.view(size[0], 1, 1, self.out_size) # torch.Size([1024, 1, 1, 100])


class RelationMetaLSTM(nn.Module):
    def __init__(self, few, embed_size=100, num_hidden=100, dropout_p=0.5, device=0):
        super(RelationMetaLSTM,self).__init__()
        self.rnn = nn.LSTMCell(2 * embed_size, 2 *num_hidden)
        # self.rnn1 = nn.LSTMCell(2 * embed_size, 2 * num_hidden)
        self.few = few
        self.dropout_p = dropout_p
        self.num_hidden = num_hidden
        self.device = device

        self.set_att_W = nn.Linear(2*embed_size, num_hidden)
        self.set_att_u = nn.Linear(num_hidden, 1)
        self.softmax = nn.Softmax(dim=1)


        self.fc = nn.Linear(2*num_hidden, num_hidden)
        self.activate = nn.LeakyReLU()
        

    def forward(self, inputs):
        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1)
        hx = torch.randn(size[0], 2 * self.num_hidden).to(self.device)
        cx = torch.randn(size[0], 2 * self.num_hidden).to(self.device)
        # hx1 = torch.randn(size[0], 2 * self.num_hidden).to(self.device)
        # cx1 = torch.randn(size[0], 2 * self.num_hidden).to(self.device)
        # output = []
        for i in range(self.few):
            hx, cx = self.rnn(x[:, i, :], (hx, cx))
            # hx1, cx1 = self.rnn1(x[:, 4-i, :], (hx1, cx1))
            if i==0:
                output = hx.unsqueeze(1)
                # output1 = hx1.unsqueeze(1)
            else:
                output = torch.cat((output , (hx + x[:,i,:]).unsqueeze(1)), 1)  # 残差
                # output1 = torch.cat((output1, (hx1 + x[:,4-i,:]).unsqueeze(1)), 1)
        # output, _ = self.rnn(x)
        # output = torch.cat((output, output1), 1)
        att = self.set_att_W(output).tanh()
        att_w = self.set_att_u(att)
        att_w = self.softmax(att_w)

        encode = torch.matmul(output.transpose(1, 2), att_w).transpose(1, 2)
        encode = self.fc(encode)
        encode = self.activate(encode)
        return encode.view(size[0], 1, 1, self.num_hidden)


class RelationMetaBiLSTM(nn.Module):
    def __init__(self, few, embed_size=100, num_hidden=100, dropout_p=0.5, device=0):
        super(RelationMetaBiLSTM,self).__init__()
        self.rnn = nn.LSTMCell(2 * embed_size, 2 *num_hidden)
        self.rnn1 = nn.LSTMCell(2 * embed_size, 2 * num_hidden)
        self.few = few
        self.dropout_p = dropout_p
        self.num_hidden = num_hidden
        self.device = device

        self.set_att_W = nn.Linear(2*embed_size, num_hidden)
        self.set_att_u = nn.Linear(num_hidden, 1)
        self.softmax = nn.Softmax(dim=1)


        self.fc = nn.Linear(2*num_hidden, num_hidden)
        self.activate = nn.LeakyReLU()

    def forward(self, inputs):
        size = inputs.shape
        x = inputs.contiguous().view(size[0], size[1], -1)
        hx = torch.randn(size[0], 2 * self.num_hidden).to(self.device)
        cx = torch.randn(size[0], 2 * self.num_hidden).to(self.device)
        hx1 = torch.randn(size[0], 2 * self.num_hidden).to(self.device)
        cx1 = torch.randn(size[0], 2 * self.num_hidden).to(self.device)
        for i in range(self.few):
            hx, cx = self.rnn(x[:, i, :], (hx, cx))
            hx1, cx1 = self.rnn1(x[:, 4-i, :], (hx1, cx1))
            if i==0:
                output = hx.unsqueeze(1)
                output1 = hx1.unsqueeze(1)
            else:
                output = torch.cat((output , (hx + x[:,i,:]).unsqueeze(1)), 1)  # 残差
                output1 = torch.cat((output1, (hx1 + x[:,4-i,:]).unsqueeze(1)), 1)
        # output, _ = self.rnn(x)
        output = torch.cat((output, output1), 1)
        att = self.set_att_W(output).tanh()
        att_w = self.set_att_u(att)
        att_w = self.softmax(att_w)

        encode = torch.matmul(output.transpose(1, 2), att_w).transpose(1, 2)
        encode = self.fc(encode)
        encode = self.activate(encode)
        return encode.view(size[0], 1, 1, self.num_hidden)


class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, t, r, pos_num):
        # print(h.shape)    # torch.Size([1024, 10, 1, 100])
        score = -torch.norm(h + r - t, 2, -1).squeeze(2)    # torch.Size([1024, 10])
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score


class EmbeddingLearnerLSTM(nn.Module):
    def __init__(self, embed_size=100, hidden_size=100, device=0):
        super(EmbeddingLearnerLSTM, self).__init__()
        self.rnn = nn.LSTMCell(2 * embed_size, 2 * embed_size)
        self.embed_size = embed_size
        self.device = device

    def forward(self, h, t, r, pos_num,  process_step=4):
        # print(h.shape)    # torch.Size([1024, 10, 1, 100])
        # print(h.shape, r.shape, t.shape)
        # print(torch.cat((h, t), 3).shape)
        # exit(0)
        size = h.shape
        input = torch.cat((h, t), 3).squeeze(2)
        for i in range(size[1]):
            ipt = input[:, i, :]
            print(ipt.shape)
            hx = torch.randn(size[0], 2 * self.embed_size).to(self.device)
            cx = torch.randn(size[0], 2 * self.embed_size).to(self.device)
            for _ in range(process_step):
                hx_, c = self.rnn(ipt, (hx, cx))
                h = ipt + hx_
                attn = F.softmax(torch.mul(h, r.squeeze(2)[:, i, :]))
                r = torch.mul(attn, r)
                hx = torch.cat()
                print(hx_.shape)
                exit(0)
            
        score = -torch.norm(h + r - t, 2, -1).squeeze(2)    # torch.Size([1024, 10])
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score


class MetaR(nn.Module):
    def __init__(self, dataset, parameter):
        super(MetaR, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.abla = parameter['ablation']
        if parameter['prefix'] in ('OneHop', 'OneHop_Pre_In_LSTM', 'OneHop_Pre_In_BiLSTM'):
            self.build_connection(dataset)
            self.embedding = EmbeddingOneHop(dataset, parameter, self.e1_degrees, self.connections)
        elif parameter['prefix'] in ('OneHopAtt', 'OneHopAtt_Pre_In_LSTM', 'OneHopAtt_Pre_In_BiLSTM'):
            self.build_connection(dataset)
            self.embedding = EmbeddingOneHopAttention(dataset, parameter, self.e1_degrees, self.connections)
        else:
            self.embedding = Embedding(dataset, parameter) 

        # if parameter['dataset'] == 'Wiki-One':
        #     self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=50, num_hidden1=250,
        #                                                 num_hidden2=100, out_size=50, dropout_p=self.dropout_p)
        # elif parameter['dataset'] == 'NELL-One':

        
        if parameter['prefix'] in ('LSTM', 'Pre_In_LSTM', 'OneHop_Pre_In_LSTM', 'OneHopAtt_Pre_In_LSTM', 'Pre_In_LSTM_Hit1'):
            self.relation_learner = RelationMetaLSTM(parameter['few'], embed_size=100, num_hidden=100, dropout_p=self.dropout_p, device=self.device)
        elif parameter['prefix'] in ('BiLSTM', 'Pre_In_BiLSTM', 'OneHop_Pre_In_BiLSTM', 'OneHopAtt_Pre_In_BiLSTM', 'Pre_In_BiLSTM_Hit1'):
            self.relation_learner = RelationMetaBiLSTM(parameter['few'], embed_size=100, num_hidden=100, dropout_p=self.dropout_p, device=self.device)
        else:
            self.relation_learner = RelationMetaLearner(parameter['few'], embed_size=100, num_hidden1=500,
                                                        num_hidden2=200, out_size=100, dropout_p=self.dropout_p)
        
        if parameter['prefix'] in ('LSTM_Matcher'):
            self.embedding_learner = EmbeddingLearnerLSTM(embed_size=100)
        else:
            self.embedding_learner = EmbeddingLearner()
        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()
        self.rel2id = dataset['rel2id']
        self.rel2emb = dataset['rel2emb']

    def build_connection(self, dataset, max_=20):
        print('Building connections ... ...')
        print("----------------------------")
        self.connections = np.zeros((len(dataset['ent2id']), max_, 2))
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)  # 节点的度
        with open(dataset['path_graph']) as f:
            lines = f.readlines()
            for line in lines:
                e1, rel, e2 = line.rstrip().split()
                self.e1_rele2[e1].append((dataset['rel2id'][rel], dataset['ent2id'][e2]))
                self.e1_rele2[e2].append((dataset['rel2id'][rel+'_inv'], dataset['ent2id'][e1]))
        
        degrees = {}
        for ent in dataset['ent2id'].keys():
            id = dataset['ent2id'][ent]
            neighbors = self.e1_rele2[ent]
            if len(neighbors) > max_:
                neighbors = neighbors[:max_]
            degrees[ent] = len(neighbors)
            self.e1_degrees[id] = len(neighbors)
            for idx, _ in enumerate(neighbors):
                self.connections[id, idx, 0] = _[0]
                self.connections[id, idx, 1] = _[1]
        return degrees

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def forward(self, task, iseval=False, curr_rel=''):
        # transfer task string into embedding
        support, support_negative, query, negative = [self.embedding(t) for t in task]

        few = support.shape[1]              # num of few
        num_sn = support_negative.shape[1]  # num of support negative
        num_q = query.shape[1]              # num of query
        num_n = negative.shape[1]           # num of query negative

        if iseval:
            self.relation_learner.eval()
        rel = self.relation_learner(support)  # torch.Size([1024, 1, 1, 100])
        rel.retain_grad()   # torch.Size([1024, 1, 1, 100])

        # relation for support 个数扩展到和support相同
        rel_s = rel.expand(-1, few+num_sn, -1, -1)  # torch.Size([1024, 10, 1, 100])

        # because in test and dev step, same relation uses same support,
        # so it's no need to repeat the step of relation-meta learning
        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]
        else:
            if not self.abla:   # 更新Relation Meta, 求解Gradient Meta
                # split on e1/e2 and concat on pos/neg
                # print(support[0].shape, support_negative[0].shape)  --> torch.Size([5, 2, 100]) torch.Size([5, 2, 100]) 
                
                sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)   # 头尾实体拆分，正负例拼接
                # print(sup_neg_e1[0].shape)  -->  torch.Size([10, 1, 100])
                
                # 正负例的分值 transE思想，h+r=t
                p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, few)

                y = torch.Tensor([1]).to(self.device)
                self.zero_grad()
                loss = self.loss_func(p_score, n_score, y)
                loss.backward(retain_graph=True)

                grad_meta = rel.grad
                rel_q = rel - self.beta*grad_meta
            else:
                rel_q = rel

            self.rel_q_sharing[curr_rel] = rel_q    # 记录每个relation对应的embedding

        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1) # 个数扩展到和query相同

        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)  # [bs, nq+nn, 1, es]
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_q)

        return p_score, n_score

    def predict(self, task, curr_rel=''):
        # transfer task string into embedding
        support, query = [self.embedding(t) for t in task]
        # print(support.shape, query.shape)   # torch.Size([1, 5, 2, 100]) torch.Size([1, 123, 2, 100])
        few = support.shape[1]              # num of few
        num_q = query.shape[1]              # num of query

        # self.relation_learner.eval()
        rel = self.relation_learner(support)  # torch.Size([1, 1, 1, 100])
        # rel.retain_grad()

        rel_q = rel.expand(-1, num_q, -1, -1) # 个数扩展到和query相同
        p_score, _ = self.embedding_learner(query[:,:,0,:].unsqueeze(2), query[:,:,1,:].unsqueeze(2), rel_q, num_q)

        return p_score
