import torch

from models4 import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy

from preprocess_inductive import read_entity_from_id, read_relation_from_id, init_embeddings, build_data
from create_batch_inductive2 import *
from utils import save_model, process_data
from torch.utils.data import DataLoader, Dataset
import random
import argparse
import os
import sys
import logging
import time
import pickle

import multiprocessing as mp
import time
import tqdm

def extract_directed_subgraph_batch(entity,  neighbors, edges, distance, inverse_neighbors, inverse_edges, inverse_distance, inverse_1hop, hop=3, undirected_neighbor=None, undirected_triple=None, undirected_distance=None, undirected_inverse_1hop=None):

        subgraph = {}
        if len(np.array(entity).shape) == 1:
            y = np.array([entity[i] for i in range(len(entity))]); y = list(np.tile(y, len(entity)))

            x = np.array([entity[i] for i in range(len(entity))])
            x = np.repeat(x, len(entity))

        else:
            x = list(entity[:,0]); y = list(entity[:,2]); re = list(entity[:,1])
        print(len(x), len(y), 'length')
        start = time.time()
       # pool = mp.Pool(mp.cpu_count())
       # print(mp.cpu_count())
        pool = mp.Pool(32)
     #   results = pool.map_async(parallel_worker, [(i, j, neighbors[i], edges[i], distance[i], inverse_neighbors[j], inverse_edges[j], inverse_distance[j], inverse_1hop[j], hop) for i, j in zip(x, y)])
        results = pool.map_async(parallel_worker, [(i, j, r, neighbors, edges, distance, inverse_neighbors, inverse_edges, inverse_distance, inverse_1hop, hop, undirected_neighbor, undirected_triple, undirected_distance, undirected_inverse_1hop) for i, j, r in zip(x, y, re)])
        remaining = results._number_left
 
        while True:
           # pbar.update(remaining - results._number_left)
            if results.ready(): break
            remaining = results._number_left
            time.sleep(1)
        results = results.get()
        pool.close()
      #  pbar.close()
        for g, source, target in results:
              if(source not in subgraph.keys()):
                     subgraph[source] = {}
              subgraph[source][target] = g
    
        end = time.time()
        print('extract batch subgraph time', end - start)
        return subgraph


def extract_directed_subgraph3(source, target, relation, neighbor, triple, distance, neighbor_inverse, triple_inverse, distance_inverse, inverse_1hop, hop, undirected_neighbor, undirected_triple, undirected_distance, undirected_inverse_1hop):
        subgraph = []
  
        node = set()
        neighbor_a = set(neighbor[source]); neighbor_b = set(neighbor_inverse[target]); inverse = set(inverse_1hop[target])
        common = list(neighbor_a.intersection(inverse))
        distance_a = distance[source]; distance_b = distance_inverse[target]
        graph = {}
        unfinished = True
        
        if len(common):
            common = list(neighbor_a.intersection(neighbor_b))
            if source not in common:
                common.append(source)
            if target not in common:
                common.append(target)
            distance_a = distance[source]; distance_b = distance_inverse[target]
            triple_a = triple[source]; triple_b = triple_inverse[target]          
            for j in  range(len(triple_a)):
                if (triple_a[j][0] in common) and (triple_a[j][1] in common) and (triple_a[j] != [source, target, relation]):
                      subgraph.append(triple_a[j])
                      node.add(triple_a[j][0])
                      node.add(triple_a[j][1])
                      if len(node)>50 and (source in node) and (target in node):    #########to perform graph pruncing
                           unfinished = False
                           break 
            if unfinished:
                for j in  range(len(triple_b)):
                    if (triple_b[j][1] == target)  and (triple_b[j][0] in common) and (triple_b[j] not in subgraph) and (triple_b[j] != [source, target, relation]):
                         subgraph.append(triple_b[j])
                         node.add(triple_b[j][0])
                         node.add(triple_b[j][1])
                         if len(node)>50 and (source in node) and (target in node):
                            break
       


        
        if len(subgraph)==0:
            triple_a = undirected_triple[source]; triple_b = undirected_triple[target] 
            neighbor_a = set(list(undirected_neighbor[source]))
            neighbor_b = set(list(undirected_neighbor[target]))
            common = list(neighbor_a.intersection(set(list(undirected_inverse_1hop[target]))))
            if len(common):
                distance_a = undirected_distance[source]; distance_b = undirected_distance[target]
                common = list(neighbor_a.intersection(neighbor_b))
                if source not in common:
                    common.append(source)
                if target not in common:
                    common.append(target) 
                for j in  range(len(triple_a)):
                     if (triple_a[j][0] in common) and (triple_a[j][1] in common) and (triple_a[j] != [source, target, relation]):
                         subgraph.append(triple_a[j])
                         node.add(triple_a[j][0])
                         node.add(triple_a[j][1])
                         if len(node)>50 and (source in node) and (target in node):    #########to perform graph pruncing
                            unfinished = False
                            break 
            if unfinished and len(common):
                for j in  range(len(triple_b)):
                    if (triple_b[j][1] in common)  and (triple_b[j][0] in common) and (triple_b[j] not in subgraph) and (triple_b[j] != [source, target, relation]):
                         subgraph.append(triple_b[j])
                         node.add(triple_b[j][1])
                         node.add(triple_b[j][0])
                         if len(node)>50 and (target in node) and (source in node):
                             break
        

        node.add(source)
        node.add(target)
      #  print(len(subgraph))
        node = list(node); distance_source = []; distance_target = []
        
        for nodes in node:
           
            if nodes == source:
                distance_source.append(0)
                distance_target.append(1)
                continue
            if nodes == target:
                distance_source.append(1)
                distance_target.append(0)
                continue
           
            if nodes not in distance_a.keys():
                distance_source.append(hop - distance_b[nodes])
              #  print(hop - distance_b[nodes])
            else:  
                distance_source.append(distance_a[nodes])
            if nodes not in distance_b.keys():
                distance_target.append(hop - distance_a[nodes])
              #  print(hop - distance_a[nodes])
            else:
                distance_target.append(distance_b[nodes])
        if len(node):
             distance_source = np.eye(hop+1)[distance_source]
           #  print(distance_source.shape)
             distance_target = np.eye(hop+1)[distance_target]
           #  print(distance_target)
             node = np.expand_dims(node, axis = 1)
             node_and_distance = np.concatenate([node, distance_source], axis = 1)
             node_and_distance = np.concatenate([node_and_distance, distance_target], axis = 1)
             graph['node'] = node_and_distance
        else:
             graph['node'] = np.array([])
       
        '''
        embedding_size = 32
        total_embed = []
        for nodes in node:
            if nodes not in distance_a.keys():
                u = hop - distance_b[nodes]
            else:  
                u = distance_a[nodes]
            if nodes not in distance_b.keys():
                s = hop - distance_a[nodes]
            else:
                s = distance_b[nodes]
        
            node_embed = np.random.normal(u, s, embedding_size)
            nodes = np.expand_dims(nodes, axis=0)
            node_and_embed = np.concatenate([nodes, node_embed], axis = 0)
            node_and_embed = np.expand_dims(node_and_embed, axis = 0)
            total_embed.append(node_and_embed)
        total_embed =  np.concatenate(total_embed, axis=0)
        graph['node'] = total_embed
        '''
        '''
        embedding_size = 8
        total_embed = []
        for nodes in node:
            if nodes not in distance_a.keys():
                u = hop - distance_b[nodes]
            else:  
                u = distance_a[nodes]
            if nodes not in distance_b.keys():
                s = hop - distance_a[nodes]
            else:
                s = distance_b[nodes]
        
            node_embed = np.random.normal(u, 1, embedding_size)
            node_embed2 = np.random.normal(s, 1, embedding_size)
            nodes = np.expand_dims(nodes, axis=0)
            node_and_embed = np.concatenate([nodes, node_embed, node_embed2], axis = 0)
            node_and_embed = np.expand_dims(node_and_embed, axis = 0)
            total_embed.append(node_and_embed)
        total_embed =  np.concatenate(total_embed, axis=0)
        graph['node'] = total_embed
        '''
        graph['edge'] = np.array(subgraph)
     #   graph['source'] = source; graph['target'] = target

        return graph, source, target


def parallel_worker(x):
    return extract_directed_subgraph3(*x)



###########################undirected######################################
def extract_undirected_subgraph_batch(entity,  neighbors, edges, distance, inverse_1hop, hop=3):
        subgraph = {}
        if len(np.array(entity).shape) == 1:
            y = np.array([entity[i] for i in range(len(entity))]); y = list(np.tile(y, len(entity)))

            x = np.array([entity[i] for i in range(len(entity))])
            x = np.repeat(x, len(entity))

        else:
            x = list(entity[:,0]); y = list(entity[:,2]); re = list(entity[:,1])
        print(len(x), len(y), 'length')
        start = time.time()
       # pool = mp.Pool(mp.cpu_count())
        pool = mp.Pool(32)
 
     #   results = pool.map_async(parallel_worker, [(i, j, neighbors[i], edges[i], distance[i], inverse_neighbors[j], inverse_edges[j], inverse_distance[j], inverse_1hop[j], hop) for i, j in zip(x, y)])
        results = pool.map_async(parallel_worker2, [(i, j, r, neighbors[i], edges[i], distance[i], neighbors[j], edges[j], distance[j], inverse_1hop[j], hop) for i, j, r in zip(x, y, re)])
        remaining = results._number_left
 
        while True:
           # pbar.update(remaining - results._number_left)
            if results.ready(): break
            remaining = results._number_left
            time.sleep(1)
        results = results.get()
        pool.close()
      #  pbar.close()
        for g, source, target in results:
              if(source not in subgraph.keys()):
                     subgraph[source] = {}
              subgraph[source][target] = g
        end = time.time()
        print('extract batch subgraph time', end - start)
        return subgraph


def extract_undirected_subgraph2(source, target, relation, neighbor_a, triple_a, distance_a, neighbor_b, triple_b, distance_b, inverse_1hop, hop=3):
        subgraph = []
        node = set()
        neighbor_a = set(neighbor_a); neighbor_b = set(neighbor_b); inverse = set(inverse_1hop)
        common = list(neighbor_a.intersection(inverse))
        graph = {}
        unfinished = True
        
        if len(common):
            common = list(neighbor_a.intersection(neighbor_b))
            if source not in common:
                common.append(source)
            if target not in common:
                common.append(target)      
            for j in  range(len(triple_a)):
                if (triple_a[j][0] in common) and (triple_a[j][1] in common) and (triple_a[j] != [source, target, relation]):
                      subgraph.append(triple_a[j])
                      node.add(triple_a[j][0])
                      node.add(triple_a[j][1])
                      if len(node)>100 and (source in node) and (target in node):
                           unfinished = False
                           break 
            if unfinished:
                for j in  range(len(triple_b)):
                    if (triple_b[j][1]  in common)  and (triple_b[j][0] in common) and (triple_b[j] not in subgraph) and (triple_b[j] != [source, target, relation]):
                         subgraph.append(triple_b[j])
                         node.add(triple_b[j][1])
                         node.add(triple_b[j][0])
                         if len(node)>100 and (source in node) and (target in node):
                              break

        '''
        if len(subgraph)==0:
            common = list(neighbor_a.intersection(neighbor_b))
            if len(common):
               # distance_a = undirected_distance[source]; distance_b = undirected_distance[target]
                common = list(neighbor_a.intersection(neighbor_b))
                if source not in common:
                    common.append(source)
                if target not in common:
                    common.append(target) 
                for j in  range(len(triple_a)):
                     if (triple_a[j][0] in common) and (triple_a[j][1] in common):
                         subgraph.append(triple_a[j])
                         node.add(triple_a[j][0])
                         node.add(triple_a[j][1])
                         if len(node)>50:
                              unfinished = False
                              break

            if common:
                for j in  range(len(triple_b)):
                    if (triple_b[j][1] in common)  and (triple_b[j][0] in common) and (triple_b[j] not in subgraph):
                         subgraph.append(triple_b[j])
                         node.add(triple_b[j][1])
                         node.add(triple_b[j][0])
                         if len(node)>100:
                              break
        '''
        node.add(source)
        node.add(target)
       # print(len(subgraph))
        node = list(node); distance_source = []; distance_target = []
       
        for nodes in node:
           
            if nodes == source:
                distance_source.append(0)
                distance_target.append(1)
                continue
            if nodes == target:
                distance_source.append(1)
                distance_target.append(0)
                continue
           
            if nodes not in distance_a.keys():
                distance_source.append(hop - distance_b[nodes])
            else:  
                distance_source.append(distance_a[nodes])
            if nodes not in distance_b.keys():
                distance_target.append(hop - distance_a[nodes])
            else:
                distance_target.append(distance_b[nodes])
        if len(node):
             distance_source = np.eye(hop+1)[distance_source]
             print(distance_source)
             distance_target = np.eye(hop+1)[distance_target]
             node = np.expand_dims(node, axis = 1)
             node_and_distance = np.concatenate([node, distance_source], axis = 1)
             node_and_distance = np.concatenate([node_and_distance, distance_target], axis = 1)
             graph['node'] = node_and_distance
        else:
             graph['node'] = np.array([])
        '''
        embedding_size = 32
        total_embed = []
        for nodes in node:
            if nodes not in distance_a.keys():
                u = hop - distance_b[nodes]
            else:  
                u = distance_a[nodes]
            if nodes not in distance_b.keys():
                s = hop - distance_a[nodes]
            else:
                s = distance_b[nodes]
        
            node_embed = np.random.normal(u, s, embedding_size)
            nodes = np.expand_dims(nodes, axis=0)
            node_and_embed = np.concatenate([nodes, node_embed], axis = 0)
            node_and_embed = np.expand_dims(node_and_embed, axis = 0)
            total_embed.append(node_and_embed)
        total_embed =  np.concatenate(total_embed, axis=0)
        graph['node'] = total_embed
        '''
        graph['edge'] = np.array(subgraph)
     #   graph['source'] = source; graph['target'] = target

        return graph, source, target


def parallel_worker2(x):
    return extract_undirected_subgraph2(*x)










class MyQUEUE: # just an implementation of a queue

    def __init__(self):
        self.holder = []

    def enqueue(self,val):
        self.holder.append(val)

    def dequeue(self):
        val = None
        try:
            val = self.holder[0]
            if len(self.holder) == 1:
                self.holder = []
            else:
                self.holder = self.holder[1:]   
        except:
            pass

        return val  

    def IsEmpty(self):
        result = False
        if len(self.holder) == 0:
            result = True
        return result

def BFS(graph,start,end, hop=3):
    q = MyQUEUE()
    all_path = []

    last_node = start
    if last_node not in graph.keys():
        return []
    for link_node in graph[last_node].keys():
       # if link_node not in tmp_path:
         #   new_path = []
        new_path = [[last_node, link_node, graph[last_node][link_node]]]
        q.enqueue(new_path)
          #  q.enqueue(temp_path)

    while q.IsEmpty() == False:
        tmp_path = q.dequeue()
        last_node = tmp_path[len(tmp_path)-1][1]
        #print tmp_path
        if last_node == end:
            print ("VALID_PATH : ",tmp_path)
            all_path.append(tmp_path)
            if len(all_path) > 100:
                break
            continue
        if last_node not in graph.keys():
            continue
        if len(tmp_path) > hop + 1:
            continue
        for link_node in graph[last_node].keys():
            if [last_node, link_node, graph[last_node][link_node]] not in tmp_path:
                new_path = []
                new_path = tmp_path + [[last_node, link_node, graph[last_node][link_node]]]
                q.enqueue(new_path)

    return all_path



def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-data", "--data",
                      default="./data/FB15k-237-inductive/", help="data directory")
    args.add_argument("-dataset", "--dataset",
                      default="FB15k-237", help="data set")

    args.add_argument("--dt", type=str,
                      default='now', help="datetime")
    args.add_argument("--prefix", type=str,
                      default='', help="prefix")

    args.add_argument("--hop", type=int,
                      default=3, help="hop")

    args.add_argument("--valid_invalid_ratio_conv", type=int,
                      default=1, help="valid_invalid_ratio_conv")

    args = args.parse_args()
    return args


args = parse_args()

def load_data(args):
    train_data, validation_data, test_data, entity2id, relation2id, relation_number, unique_entities_train, unique_entities_validation, unique_entities_test = build_data(
        args.data, is_unweigted=False, directed=True)

    print(relation2id)
    corpus = Corpus(args, train_data, validation_data, test_data, entity2id, relation2id, relation_number,
                    len(train_data), args.valid_invalid_ratio_conv, unique_entities_train, unique_entities_validation, unique_entities_test)

    print('finished loading corpus...')
    return corpus

import os.path
hop = args.hop
Corpus_ = load_data(args)
datapath = "./data/{}_{}_hop_new_data.pickle".format(args.dataset, hop)
datapath2 = "./subgraph/{}_{}_hop_undirected_subgraph.pickle".format(args.dataset, hop)
if os.path.isfile(datapath2):
    with open(datapath, 'rb') as handle:
        new_data = pickle.load(handle)

    datapath = "./subgraph/{}_{}_hop_all_subgraph.pickle".format(args.dataset, hop)  
    with open(datapath, 'rb') as handle:
        all_subgraph = pickle.load(handle)    

    new_val_entity = new_data['new_val_entity']; new_test_entity = new_data['new_test_entity']
    train_neighbors = all_subgraph['train_neighbors']; train_edges = all_subgraph['train_edges']
    train_distance = all_subgraph['train_distance']; train_inverse_neighbor = all_subgraph['train_inverse_neighbor']
    train_inverse_triples = all_subgraph['train_inverse_edges']; train_inverse_distance = all_subgraph['train_inverse_distance']


    test_neighbors = all_subgraph['test_neighbors']; test_edges = all_subgraph['test_edges']
    test_distance = all_subgraph['test_distance']; test_inverse_neighbor = all_subgraph['test_inverse_neighbor']
    test_inverse_triples = all_subgraph['test_inverse_edges']; test_inverse_distance = all_subgraph['test_inverse_distance']
    test_inverse_1hop = all_subgraph['test_inverse_1hop']; train_inverse_1hop = all_subgraph['train_inverse_1hop']
   # val_inverse_1hop = all_subgraph['val_inverse_1hop']


    datapath = "./subgraph/{}_{}_hop_undirected_subgraph.pickle".format(args.dataset, hop)  
    with open(datapath, 'rb') as handle:
        all_subgraph = pickle.load(handle)    

    undirected_train_neighbors = all_subgraph['train_neighbors']; undirected_train_edges = all_subgraph['train_edges']
    undirected_train_distance = all_subgraph['train_distance']

    undirected_test_neighbors = all_subgraph['test_neighbors']; undirected_test_edges = all_subgraph['test_edges']
    undirected_test_distance = all_subgraph['test_distance']
    undirected_test_inverse_1hop = all_subgraph['test_inverse_1hop']; undirected_train_inverse_1hop = all_subgraph['train_inverse_1hop']
  #  undirected_val_inverse_1hop = all_subgraph['val_inverse_1hop']


else:
    datapath = "./subgraph/{}_{}_hop_all_subgraph.pickle".format(args.dataset, hop)
    if 1:
        all_subgraph = {}
        train_neighbors, train_edges, train_distance, train_inverse_neighbor, train_inverse_triples, train_inverse_distance, train_inverse_1hop = \
                                Corpus_.get_batch_nhop_neighbors2(Corpus_.unique_entities_train, Corpus_.graph, Corpus_.inverse_graph, hop)
        print('start to creat train subgraph...')

        all_subgraph['train_neighbors'] = train_neighbors; all_subgraph['train_edges'] = train_edges
        all_subgraph['train_distance'] = train_distance; all_subgraph['train_inverse_neighbor'] = train_inverse_neighbor
        all_subgraph['train_inverse_edges'] = train_inverse_triples; all_subgraph['train_inverse_distance'] = train_inverse_distance
        all_subgraph['train_inverse_1hop'] = train_inverse_1hop

        test_neighbors, test_edges, test_distance, test_inverse_neighbor, test_inverse_triples, test_inverse_distance, test_inverse_1hop=\
                                        Corpus_.get_batch_nhop_neighbors2(Corpus_.unique_entities_test, Corpus_.test_graph, Corpus_.inverse_test_graph, hop)
        all_subgraph['test_neighbors'] = test_neighbors; all_subgraph['test_edges'] = test_edges
        all_subgraph['test_distance'] = test_distance; all_subgraph['test_inverse_neighbor'] = test_inverse_neighbor
        all_subgraph['test_inverse_edges'] = test_inverse_triples; all_subgraph['test_inverse_distance'] = test_inverse_distance
        all_subgraph['test_inverse_1hop'] = test_inverse_1hop
        print('start to creat test subgraph...')


        with open(datapath, 'wb') as handle:
            pickle.dump(all_subgraph, handle, protocol=pickle.HIGHEST_PROTOCOL)


        ############################# create undirected graph###############################################
        datapath = "./subgraph/{}_{}_hop_undirected_subgraph.pickle".format(args.dataset, hop)
        undirected_subgraph = {}
        undirected_train_neighbors, undirected_train_edges, undirected_train_distance, undirected_train_inverse_1hop = \
                                Corpus_.get_batch_nhop_neighbors_undirected(Corpus_.unique_entities_train, Corpus_.graph, Corpus_.inverse_graph, hop)
        new_train_entity, new_train_data = define_new_dataset( Corpus_.train_indices, undirected_train_neighbors, undirected_train_inverse_1hop)
        print('start to create undirected train subgraph...')
 
        undirected_subgraph['train_neighbors'] = undirected_train_neighbors; undirected_subgraph['train_edges'] = undirected_train_edges
        undirected_subgraph['train_distance'] = undirected_train_distance
        undirected_subgraph['train_inverse_1hop'] = undirected_train_inverse_1hop

        new_val_entity, new_val_data = define_new_dataset( Corpus_.validation_indices, undirected_train_neighbors, undirected_train_inverse_1hop)
 
        undirected_test_neighbors, undirected_test_edges, undirected_test_distance, undirected_test_inverse_1hop=\
                                        Corpus_.get_batch_nhop_neighbors_undirected(Corpus_.unique_entities_test, Corpus_.test_graph, Corpus_.inverse_test_graph, hop)

        new_test_entity, new_test_data = define_new_dataset(Corpus_.test_indices, undirected_test_neighbors, undirected_test_inverse_1hop)

        undirected_subgraph['test_neighbors'] = undirected_test_neighbors; undirected_subgraph['test_edges'] = undirected_test_edges
        undirected_subgraph['test_distance'] = undirected_test_distance
        undirected_subgraph['test_inverse_1hop'] = undirected_test_inverse_1hop
        print('start to create undirected test subgraph...')

        with open(datapath, 'wb') as handle:
            pickle.dump(undirected_subgraph, handle, protocol=pickle.HIGHEST_PROTOCOL)

        new_data = {}; new_data['new_train_entity'] = new_train_entity; new_data['new_train_data'] = new_train_data
        new_data['new_val_entity'] = new_val_entity; new_data['new_val_data'] = new_val_data
        new_data['new_test_entity'] = new_test_entity; new_data['new_test_data'] = new_test_data
        datapath = "./data/{}_{}_hop_new_data.pickle".format(args.dataset, hop)
        with open(datapath, 'wb') as handle:
            pickle.dump(new_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


Corpus_.unique_entities_train = new_data['new_train_entity']
print('unique_entity_train: ', len(Corpus_.unique_entities_train))
Corpus_.unique_entities_validation = new_data['new_val_entity']
Corpus_.unique_entities_test = new_data['new_test_entity']
Corpus_.train_indices = new_data['new_train_data'].astype(np.int32)
Corpus_.train_values = np.array([[1]] * Corpus_.train_indices.shape[0]).astype(np.float32)
Corpus_.validation_indices = new_data['new_val_data'].astype(np.int32)
Corpus_.validation_values = np.array([[1]] * Corpus_.validation_indices.shape[0]).astype(np.float32)
Corpus_.test_indices = new_data['new_test_data'].astype(np.int32)
Corpus_.test_values = np.array([[1]] * Corpus_.test_indices.shape[0]).astype(np.float32)



########################train################################
print('start to create negative sample...')
total_train_data, total_train_label = Corpus_.get_total_train_data(args.valid_invalid_ratio_conv, undirected_train_neighbors, undirected_train_inverse_1hop)
datapath = "./data/{}_{}_total_train_data.pickle".format(args.dataset, hop)
with open(datapath, 'wb') as handle:
     pickle.dump(total_train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

datapath = "./data/{}_{}_total_train_label.pickle".format(args.dataset, hop)
with open(datapath, 'wb') as handle:
     pickle.dump(total_train_label, handle, protocol=pickle.HIGHEST_PROTOCOL)


train_subgraph = extract_directed_subgraph_batch(total_train_data, train_neighbors, train_edges, train_distance, train_inverse_neighbor, train_inverse_triples, train_inverse_distance, train_inverse_1hop, hop,
undirected_train_neighbors, undirected_train_edges, undirected_train_distance, undirected_train_inverse_1hop)
datapath1 = "./subgraph/{}_{}_hop_train_subgraph.pickle".format(args.dataset, hop)
with open(datapath1, 'wb') as handle:
     pickle.dump(train_subgraph, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('finished creating train subgraph...')

'''
train_subgraph = extract_undirected_subgraph_batch(total_train_data, undirected_train_neighbors, undirected_train_edges, undirected_train_distance, undirected_train_inverse_1hop, hop)
datapath1 = "./subgraph/{}_{}_hop_undirected_train_subgraph.pickle".format(args.dataset, hop)
with open(datapath1, 'wb') as handle:
     pickle.dump(train_subgraph, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('finished creating train subgraph...')
'''

######################validation##########################

total_val_data, total_val_label = Corpus_.get_total_val_data(args.valid_invalid_ratio_conv, undirected_train_neighbors, undirected_train_inverse_1hop)
datapath = "./data/{}_{}_total_val_data.pickle".format(args.dataset, hop)
with open(datapath, 'wb') as handle:
     pickle.dump(total_val_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

datapath = "./data/{}_{}_total_val_label.pickle".format(args.dataset, hop)
with open(datapath, 'wb') as handle:
     pickle.dump(total_val_label, handle, protocol=pickle.HIGHEST_PROTOCOL)


validation_subgraph = extract_directed_subgraph_batch(total_val_data, train_neighbors, train_edges, train_distance, train_inverse_neighbor, train_inverse_triples, train_inverse_distance, train_inverse_1hop, hop, undirected_train_neighbors, undirected_train_edges, undirected_train_distance, undirected_train_inverse_1hop)
datapath1 = "./subgraph/{}_{}_hop_validation_subgraph.pickle".format(args.dataset, hop)
with open(datapath1, 'wb') as handle:
     pickle.dump(validation_subgraph, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
val_subgraph = extract_undirected_subgraph_batch(total_val_data, undirected_train_neighbors, undirected_train_edges, undirected_train_distance, undirected_train_inverse_1hop, hop)
datapath1 = "./subgraph/{}_{}_hop_undirected_validation_subgraph.pickle".format(args.dataset, hop)
with open(datapath1, 'wb') as handle:
     pickle.dump(val_subgraph, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''
################################test#######################

total_test_head, total_test_tail, total_test_triplet = Corpus_.get_test_all_data5(new_data['new_test_entity'], undirected_test_neighbors, undirected_test_inverse_1hop)


test_subgraph = extract_directed_subgraph_batch(total_test_triplet, test_neighbors, test_edges, test_distance, test_inverse_neighbor, test_inverse_triples, test_inverse_distance, test_inverse_1hop, hop,
undirected_test_neighbors, undirected_test_edges, undirected_test_distance, undirected_test_inverse_1hop)
datapath1 = "./subgraph/{}_{}_hop_test_subgraph.pickle".format(args.dataset, hop)
with open(datapath1, 'wb') as handle:
     pickle.dump(test_subgraph, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''
test_subgraph = extract_undirected_subgraph_batch(total_test_triplet, undirected_test_neighbors, undirected_test_edges, undirected_test_distance, undirected_test_inverse_1hop, hop)
datapath1 = "./subgraph/{}_{}_hop_undirected_test_subgraph.pickle".format(args.dataset, hop)
with open(datapath1, 'wb') as handle:
     pickle.dump(test_subgraph, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''

datapath1 = "./data/{}_{}_hop_total_test_head2.pickle".format(args.dataset, hop)
with open(datapath1, 'wb') as handle:
     pickle.dump(total_test_head, handle, protocol=pickle.HIGHEST_PROTOCOL)
datapath1 = "./data/{}_{}_hop_total_test_tail2.pickle".format(args.dataset, hop)
with open(datapath1, 'wb') as handle:
     pickle.dump(total_test_tail, handle, protocol=pickle.HIGHEST_PROTOCOL)



