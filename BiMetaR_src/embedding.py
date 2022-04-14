from numpy.lib.twodim_base import tri
import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, dataset, parameter):
        super(Embedding, self).__init__()
        self.device = parameter['device']
        self.ent2id = dataset['ent2id']
        self.es = parameter['embed_dim']

        num_ent = len(self.ent2id)
        self.embedding = nn.Embedding(num_ent, self.es)

        # if parameter['data_form'] == 'Pre-Train':
        self.ent2emb = dataset['ent2emb']
        self.embedding.weight.data.copy_(torch.from_numpy(self.ent2emb))
        # elif parameter['data_form'] in ['In-Train', 'Discard']:
        #     nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, triples):
        idx = [[[self.ent2id[t[0]], self.ent2id[t[2]]] for t in batch] for batch in triples]    # 头尾实体拼接
        idx = torch.LongTensor(idx).to(self.device)
        return self.embedding(idx)


class EmbeddingOneHop(nn.Module):
    def __init__(self, dataset, parameter, e1_degrees, connections):
        super(EmbeddingOneHop, self).__init__()
        self.device = parameter['device']
        self.ent2id = dataset['ent2id']
        self.es = parameter['embed_dim']
        self.e1_degrees = e1_degrees
        self.connections = connections

        num_ent = len(self.ent2id)
        self.embedding = nn.Embedding(num_ent, self.es)

        # if parameter['data_form'] == 'Pre-Train':
        self.ent2emb = dataset['ent2emb']
        self.embedding.weight.data.copy_(torch.from_numpy(self.ent2emb))
        # elif parameter['data_form'] in ['In-Train', 'Discard']:
        #     nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, triples):
        # print(len(triples))
        idx = [[[self.ent2id[t[0]], self.ent2id[t[2]]] for t in batch] for batch in triples]
        idx = torch.LongTensor(idx).to(self.device)
        ht_emb = self.embedding(idx).view(len(triples), len(triples[0]), 2, 1, self.es)

        idx_conn = [[[self.connections[self.ent2id[t[0]], :, 1], self.connections[self.ent2id[t[2]], :, 1]] for t in batch] for batch in triples]
        idx_conn = torch.LongTensor(idx_conn).to(self.device)
        ht_conn_emb = self.embedding(idx_conn)
        ht_emb = torch.cat((ht_emb, ht_conn_emb), dim=3)
        ht_emb_sum = torch.sum(ht_emb, dim=3)   # torch.Size([1024, 5, 2, 100])

        degrees = [[[self.e1_degrees[self.ent2id[t[0]]], self.e1_degrees[self.ent2id[t[2]]]] for t in batch] for batch in triples]
        degrees = torch.LongTensor(degrees).to(self.device)
        degrees = degrees.view(len(triples), len(triples[0]), -1, 1)    # torch.Size([1024, 5, 2, 1])
        
        return ht_emb_sum / degrees

class EmbeddingOneHopAttention(nn.Module):
    def __init__(self, dataset, parameter, e1_degrees, connections):
        super(EmbeddingOneHopAttention, self).__init__()
        self.device = parameter['device']
        self.ent2id = dataset['ent2id']
        self.es = parameter['embed_dim']
        self.rel2id = dataset['rel2id']
        self.e1_degrees = e1_degrees
        self.connections = connections
        self.Bilinear = nn.Bilinear(self.es, self.es, 1, bias=False)

        num_ent = len(self.ent2id)
        num_rel = len(self.rel2id)
        self.embedding = nn.Embedding(num_ent, self.es)
        self.rel_embedding = nn.Embedding(num_rel, self.es)
        self.linear1 = nn.Linear(self.es, self.es)
        self.linear2 = nn.Linear(self.es, self.es)

        # if parameter['data_form'] == 'Pre-Train':
        self.ent2emb = dataset['ent2emb']
        self.embedding.weight.data.copy_(torch.from_numpy(self.ent2emb))

        self.rel2emb = dataset['rel2emb']
        self.rel_embedding.weight.data.copy_(torch.from_numpy(self.rel2emb))
        # elif parameter['data_form'] in ['In-Train', 'Discard']:
        #     nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, triples):
        # print(len(triples))
        idx = [[[self.ent2id[t[0]], self.ent2id[t[2]]] for t in batch] for batch in triples]
        idx = torch.LongTensor(idx).to(self.device)
        ht_emb = self.embedding(idx)
        head_emb = ht_emb[:,:,0,:].view(len(triples), len(triples[0]), 1, self.es)
        tail_emb = ht_emb[:,:,1,:].view(len(triples), len(triples[0]), 1, self.es)
        weak_rel = (ht_emb[:, :, 1, :] - ht_emb[:, :, 0, :]).reshape(len(triples), len(triples[0]), -1, self.es)  # r = t - h, torch.Size([1024, 5, 1, 100])
        
        # ht_emb = ht_emb.view(len(triples), len(triples[0]), 2, 1, self.es)
        idx_conn = [[[self.connections[self.ent2id[t[0]], :, 1], self.connections[self.ent2id[t[2]], :, 1]] for t in batch] for batch in triples]
        idx_conn = torch.LongTensor(idx_conn).to(self.device)
        ht_conn_emb = self.embedding(idx_conn)

        idx_conn_rel = [[[self.connections[self.ent2id[t[0]], :, 0], self.connections[self.ent2id[t[2]], :, 0]] for t in batch] for batch in triples]
        idx_conn_rel = torch.LongTensor(idx_conn_rel).to(self.device)
        rel_conn_emb = self.rel_embedding(idx_conn_rel) # torch.Size([1024, 5, 2, 20, 100])
        head_rel_conn_emb = rel_conn_emb[:, :, 0, :, :].contiguous() # torch.Size([1024, 5, 20, 100])
        tail_rel_conn_emb = rel_conn_emb[:, :, 1, :, :].contiguous()

        weak_rel = weak_rel.expand(-1, -1, head_rel_conn_emb.shape[2], -1).contiguous()
        
        head_score = self.Bilinear(head_rel_conn_emb, weak_rel).squeeze(3)
        tail_score = self.Bilinear(tail_rel_conn_emb, weak_rel).squeeze(3)
        head_att = torch.softmax(head_score, dim=2).unsqueeze(2)    # torch.Size([1024, 5, 20, 100])
        tail_att = torch.softmax(tail_score, dim=2).unsqueeze(2)
        head = torch.matmul(head_att, head_rel_conn_emb)    # torch.Size([1024, 5, 1, 100])
        tail = torch.matmul(tail_att, tail_rel_conn_emb)
        
        h = torch.relu(self.linear1(head) + self.linear2(head_emb)) # torch.Size([1024, 5, 1, 100])
        t = torch.relu(self.linear1(tail) + self.linear2(tail_emb))

        return torch.cat((h, t), dim=2)