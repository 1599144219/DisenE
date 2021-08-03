import os
import numpy as np


def init_embeddings(entity_file, relation_file, k, emb_size):
    entity_emb, relation_emb = [], []

    with open(entity_file) as f:
        for line in f:
            ent_vec = []
            tmp = [float(val) for val in line.strip().split()]  # 一个实体的向量
            emb_k = int(emb_size / len(tmp))
            tot_k = int(emb_k * k)
            for i in range(tot_k):
                ent_vec += tmp
            entity_emb.append(ent_vec)

    with open(relation_file) as f:
        for line in f:
            rel_vec = []
            tmp = [float(val) for val in line.strip().split()]
            emb_k = int(emb_size / len(tmp))
            for i in range(emb_k):
                rel_vec += tmp
            relation_emb.append(rel_vec)
    return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)


def load_entity(filename):
    entity2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                line_split = line.strip().split()
                entity, entity_id = line_split[0].strip(), line_split[1].strip()
                entity2id[entity] = int(entity_id)
    return entity2id


def load_relation(filename):
    relation2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                line_split = line.strip().split()
                relation, relation_id = line_split[0].strip(), line_split[1].strip()
                relation2id[relation] = int(relation_id)
    return relation2id


def load_data(filename, entity2id, relation2id):
    with open(filename) as f:
        lines = f.readlines()

    triples_data = []
    for line in lines:
        line = line.strip().split()
        e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
        triples_data.append(
            (entity2id[e1], relation2id[relation], entity2id[e2]))

    return triples_data


def load_data2(filename, entity2id, relation2id):
    with open(filename) as f:
        lines = f.readlines()

    triples_data = []
    for line in lines:
        line = line.strip().split()
        e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
        if e1 == "?":
            triples_data.append(
                (-1, relation2id[relation], entity2id[e2]))
        elif e2 == "?":
            triples_data.append(
                (entity2id[e1], relation2id[relation], -1))
        else:
            triples_data.append(
                (entity2id[e1], relation2id[relation], entity2id[e2]))
    return triples_data


def build_data(path='./data/WN18RR/'):
    entity2id = load_entity(os.path.join(path, 'entity2id.txt'))
    relation2id = load_relation(os.path.join(path, 'relation2id.txt'))

    train_triples = load_data(os.path.join(
        path, 'train.txt'), entity2id, relation2id)
    validation_triples = load_data(
        os.path.join(path, 'valid.txt'), entity2id, relation2id)
    test_triples = load_data(os.path.join(
        path, 'test.txt'), entity2id, relation2id)

    link_triples = load_data2(os.path.join(
        path, 'link_prediction1.txt'), entity2id, relation2id)

    return train_triples, validation_triples, test_triples, link_triples, entity2id, relation2id
