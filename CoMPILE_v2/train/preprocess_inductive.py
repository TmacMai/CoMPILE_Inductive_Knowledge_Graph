import torch
import os
import numpy as np
from random import sample

def read_entity_from_id(filename='./data/WN18RR/entity2id.txt'):
    entity2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split('\t')) > 1:
                entity, entity_id = line.strip().split('\t'
                )[0].strip(), line.strip().split('\t')[1].strip()
                entity2id[entity] = int(entity_id)
    return entity2id


def read_relation_from_id(filename='./data/WN18RR/relation2id.txt'):
    relation2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split('\t')) > 1:
                relation, relation_id = line.strip().split('\t'
                )[0].strip(), line.strip().split('\t')[1].strip()
                relation2id[relation] = int(relation_id)
    return relation2id


def init_embeddings(entity_file, relation_file):
    entity_emb, relation_emb = [], []

    with open(entity_file) as f:
        for line in f:
            entity_emb.append([float(val) for val in line.strip().split('\t')])

    with open(relation_file) as f:
        for line in f:
            relation_emb.append([float(val) for val in line.strip().split('\t')])

    return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)


def parse_line(line):
    line = line.strip().split('\t')
    e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
    return e1, relation, e2


def get_id(filename,  is_unweigted=False, directed=True, saved_relation2id=None):

    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id

    triples_data = {}
    rows, cols, data = [], [], []
    unique_entities = set()

    ent = 0
    rel = 0

    for filename1 in filename:

        data = []
        with open(filename1) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if not saved_relation2id and triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

       # triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    return entity2id, relation2id, rel


def load_data(filename, entity2id, relation2id, is_unweigted=False, directed=True):
    with open(filename) as f:
        lines = [line.split() for line in f.read().split('\n')[:-1]]

    triples_data = []

    rows, cols, data = [], [], []
    unique_entities = set()
    for line in lines:
        e1, relation, e2 = line[0], line[1], line[2]
        unique_entities.add(e1)
        unique_entities.add(e2)
        triples_data.append(
            (entity2id[e1], relation2id[relation], entity2id[e2]))
        if not directed:
                # Connecting source and tail entity
            rows.append(entity2id[e1])
            cols.append(entity2id[e2])
            if is_unweigted:
                data.append(1)
            else:
                data.append(relation2id[relation])

        # Connecting tail and source entity
        rows.append(entity2id[e2])
        cols.append(entity2id[e1])
        if is_unweigted:
            data.append(1)
        else:
            data.append(relation2id[relation])

    print("number of unique_entities ->", len(unique_entities))
    return triples_data, (rows, cols, data), list(unique_entities)


def build_data(path='./data/FB15k-237-inductive-v3/', is_unweigted=False, directed=True):
   # entity2id = read_entity_from_id(path + 'entity2id.txt')
   # relation2id = read_relation_from_id(path + 'relation2id.txt')
    entity2id, relation2id, rel = get_id([os.path.join(path, 'train.txt'), os.path.join(path, 'valid.txt'), os.path.join(path, 'train_inductive.txt')])

    train_triples, train_adjacency_mat, unique_entities_train = load_data(os.path.join(path, 'train.txt'), entity2id, relation2id, is_unweigted, directed)
    validation_triples, valid_adjacency_mat, unique_entities_validation = load_data(os.path.join(path, 'valid.txt'), entity2id, relation2id, is_unweigted, directed)
    _, test_adjacency_mat, unique_entities_test = load_data(os.path.join(path, 'train_inductive.txt'), entity2id, relation2id, is_unweigted, directed)

    test_links, _, _ = load_data(os.path.join(path, 'test_inductive.txt'), entity2id, relation2id, is_unweigted, directed)

    test_triples = test_links

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    left_entity, right_entity = {}, {}

    return (train_triples, train_adjacency_mat), (validation_triples, valid_adjacency_mat), (test_triples, test_adjacency_mat), \
        entity2id, relation2id, rel, unique_entities_train,  unique_entities_validation, unique_entities_test
