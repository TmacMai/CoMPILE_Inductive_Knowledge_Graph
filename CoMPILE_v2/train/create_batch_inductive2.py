import torch
import numpy as np
from collections import defaultdict
import time
import queue
import random
import multiprocessing as mp
import time
import tqdm


def define_new_dataset(entity,  neighbors, inverse_neighbors):
        x = list(entity[:,0]); y = list(entity[:,2])
        print(len(x), 'original length')
        start = time.time()

        new_entity= set(); new_triplet = []; total=0
        for i in range(len(x)):
            neighbors_a =  neighbors[x[i]]; neighbors_b =  inverse_neighbors[y[i]]
            common = set([val for val in neighbors_a if val in neighbors_b])
            common2 = set([x[i], y[i]])
            if len(common) and (common != common2):
          #  if 1:
                new_triplet.append(entity[i, :]) 
                new_entity.add(x[i])
                new_entity.add(y[i])
                total += 1
         #   if i%20 == 0:
          #      print(i, total)
        end = time.time()
        print('new length: ', len(new_entity), 'new number of triplets: ', len(new_triplet))
        print('find new dataset time', end - start)
        return list(new_entity), np.array(new_triplet)




def parallel_worker2(x):
    return our_bfs3(*x)

def undirected_bfs3(graph, graph_inverse, source, nbd_size=2):
        visit = {}
        distance = {}
        visit[source] = 1
        distance[source] = 0
        q = queue.Queue()
        q.put((source, -1))
        all_neighbors = []
        all_neighbors.append(source)
        all_relations = []

        neighbors_inverse_1hop = []
        neighbors_inverse_1hop.append(source)

        while(not q.empty()):
            top = q.get()
            if top[0] in graph.keys() and distance[top[0]] < nbd_size:
                for target in graph[top[0]].keys():
                    if(target in visit.keys()):
                       # if np.array([top[0], target, graph[top[0]][target]]) not in all_relations:
                        if [top[0], target, graph[top[0]][target]] not in all_relations:
                              all_relations.append([top[0], target, graph[top[0]][target]])
                        
                    else:
                        visit[target] = 1
                        distance[target] = distance[top[0]] + 1
                        if distance[target] < 2:
                             neighbors_inverse_1hop.append(target)
                        all_neighbors.append(target)
                        all_relations.append([top[0], target, graph[top[0]][target]])
                        q.put((target, graph[top[0]][target]))

            if top[0] in graph_inverse.keys() and distance[top[0]] < nbd_size:
                for target in graph_inverse[top[0]].keys():
                    if(target in visit.keys()):
                        if [target, top[0], graph_inverse[top[0]][target]] not in all_relations:
                              all_relations.append([target, top[0], graph_inverse[top[0]][target]])
                        
                    else:
                        visit[target] = 1
                        distance[target] = distance[top[0]] + 1
                        if distance[target] < 2:
                             neighbors_inverse_1hop.append(target)
                        all_neighbors.append(target)
                        all_relations.append([target, top[0], graph_inverse[top[0]][target]])
                        q.put((target, graph_inverse[top[0]][target]))


        return np.array(all_neighbors), all_relations, distance, neighbors_inverse_1hop, source

def parallel_worker3(x):
    return undirected_bfs3(*x)

def our_bfs3(graph, graph_inverse, source, nbd_size=2):
        visit = {}
        distance = {}
        visit[source] = 1
        distance[source] = 0
        q = queue.Queue()
        q.put((source, -1))
        all_neighbors = []
        all_neighbors.append(source)
        all_relations = []
    #    nbd_size = nbd_size -1
        while(not q.empty()):
            top = q.get()
            if top[0] in graph.keys() and distance[top[0]] < nbd_size:
                for target in graph[top[0]].keys():
                    if(target in visit.keys()):
                       # if np.array([top[0], target, graph[top[0]][target]]) not in all_relations:
                        if [top[0], target, graph[top[0]][target]] not in all_relations:
                              all_relations.append([top[0], target, graph[top[0]][target]])
                        
                    else:
                        visit[target] = 1
                        distance[target] = distance[top[0]] + 1

                        all_neighbors.append(target)
                        all_relations.append([top[0], target, graph[top[0]][target]])
                        q.put((target, graph[top[0]][target]))


       # print('finished extracting forward neighbor...')
        visit = {}
        distance_inverse = {}
        visit[source] = 1
        distance_inverse[source] = 0
        qq = queue.Queue()
        qq.put((source, -1))
        all_neighbors_inverse = []
        all_neighbors_inverse.append(source)
        all_relations_inverse = []
        neighbors_inverse_1hop = []
       # nbd_size = 1
        while(not qq.empty()):
            top = qq.get()
            if top[0] in graph_inverse.keys() and distance_inverse[top[0]] < nbd_size:
                for target in graph_inverse[top[0]].keys():
                    if(target in visit.keys()):
                        if [target, top[0], graph_inverse[top[0]][target]] not in all_relations_inverse:
                              all_relations_inverse.append([target, top[0], graph_inverse[top[0]][target]])
                        
                    else:
                        visit[target] = 1
                        distance_inverse[target] = distance_inverse[top[0]] + 1
                        if distance_inverse[target] < 2:
                             neighbors_inverse_1hop.append(target)
                        all_neighbors_inverse.append(target)
                        all_relations_inverse.append([target, top[0], graph_inverse[top[0]][target]])
                        qq.put((target, graph_inverse[top[0]][target]))
       # print('finished extracting backward neighbor...')
        return np.array(all_neighbors), all_relations, distance, np.array(all_neighbors_inverse), all_relations_inverse, distance_inverse, neighbors_inverse_1hop, source





class Corpus:
    def __init__(self, args, train_data, validation_data, test_data, entity2id,
                 relation2id, headTailSelector, batch_size, valid_to_invalid_samples_ratio, unique_entities_train, unique_entities_validation, unique_entities_test, get_2hop=False):
        self.train_triples = train_data[0]

        # Converting to sparse tensor
        adj_indices = torch.LongTensor(
            [train_data[1][0], train_data[1][1]])  # rows and columns
        adj_values = torch.LongTensor(train_data[1][2])
        self.train_adj_matrix = (adj_indices, adj_values)
        self.unique_entities_train = unique_entities_train
        self.graph = self.get_graph(self.train_adj_matrix)
        self.inverse_graph = self.get_inverse_graph(self.train_adj_matrix)

        adj_indices_val = torch.LongTensor(
            [validation_data[1][0], validation_data[1][1]])  # rows and columns
        adj_values_val = torch.LongTensor(validation_data[1][2])
        self.val_adj_matrix = (adj_indices_val, adj_values_val)
        self.val_graph = self.get_graph(self.val_adj_matrix)
        self.inverse_val_graph = self.get_inverse_graph(self.val_adj_matrix)
        self.unique_entities_validation = unique_entities_validation

        adj_indices_test = torch.LongTensor(
            [test_data[1][0], test_data[1][1]])  # rows and columns
        adj_values_test = torch.LongTensor(test_data[1][2])
        self.test_adj_matrix = (adj_indices_test, adj_values_test)
        self.test_graph = self.get_graph(self.test_adj_matrix)
        self.inverse_test_graph = self.get_inverse_graph(self.test_adj_matrix)
        self.unique_entities_test = unique_entities_test

        # adjacency matrix is needed for train_data only, as GAT is trained for
        # training data
        self.validation_triples = validation_data[0]
        self.test_triples = test_data[0]

        self.headTailSelector = headTailSelector  # for selecting random entities
        self.entity2id = entity2id
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.relation2id = relation2id
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        self.batch_size = batch_size
        # ratio of valid to invalid samples per batch for training ConvKB Model
        self.invalid_valid_ratio = int(valid_to_invalid_samples_ratio)

        if(get_2hop):
           # self.graph = self.get_graph(self.train_adj_matrix)
            self.node_neighbors_2hop = self.get_further_neighbors()

        self.unique_entities_train = [self.entity2id[i]
                                      for i in unique_entities_train]

        self.unique_entities_validation = [self.entity2id[i]
                                      for i in unique_entities_validation]

        self.unique_entities_test = [self.entity2id[i]
                                      for i in unique_entities_test]



        self.train_indices = np.array(
            list(self.train_triples)).astype(np.int32)
        # These are valid triples, hence all have value 1
        self.train_values = np.array(
            [[1]] * len(self.train_triples)).astype(np.float32)

        self.validation_indices = np.array(
            list(self.validation_triples)).astype(np.int32)
        self.validation_values = np.array(
            [[1]] * len(self.validation_triples)).astype(np.float32)

        self.test_indices = np.array(list(self.test_triples)).astype(np.int32)
        self.test_values = np.array(
            [[1]] * len(self.test_triples)).astype(np.float32)

        self.valid_triples_dict = {j: i for i, j in enumerate(
            self.train_triples + self.validation_triples + self.test_triples)}
        print("Total triples count {}, training triples {}, validation_triples {}, test_triples {}".format(len(self.valid_triples_dict), len(self.train_indices),
                                                                                                           len(self.validation_indices), len(self.test_indices)))
     #   self.total_train_data, self.train_label = self.get_total_train_data(8)
        # For training purpose
        self.batch_indices = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
        self.batch_values = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)


    def get_total_train_data(self, invalid_valid_ratio, neighbor_a, neighbor_b):
            batch_size = len(self.train_indices)
            self.batch_indices = np.empty((batch_size * (invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty((batch_size * (invalid_valid_ratio + 1), 1)).astype(np.float32)

            self.batch_indices[:batch_size, :] = self.train_indices
            self.batch_values[:batch_size, :] = self.train_values

            last_index = batch_size
            
            if invalid_valid_ratio > 0:

                self.batch_indices[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(self.batch_indices[:last_index, :], (invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(self.batch_values[:last_index, :], (invalid_valid_ratio, 1))
                negative_sample = 0
                while negative_sample < last_index * invalid_valid_ratio:
                    if negative_sample%20 == 0:
                        print(negative_sample)
                    now_indice = random.choice([i for i in range(last_index)])
                    random_entity = random.choice(self.unique_entities_train)
                    if ((random_entity, self.batch_indices[now_indice, 1], self.batch_indices[now_indice, 2]) not in self.valid_triples_dict.keys()) and  self.common_neighbor(neighbor_a, neighbor_b, random_entity, self.batch_indices[now_indice, 2]) and (self.batch_indices[now_indice, 2]!=random_entity):
                        self.batch_indices[last_index + negative_sample, 0] = random_entity
                        self.batch_indices[last_index + negative_sample, 1] = self.batch_indices[now_indice, 1]
                        self.batch_indices[last_index + negative_sample, 2] = self.batch_indices[now_indice, 2]
                        self.batch_values[last_index + negative_sample, :] = [-1]
                        negative_sample += 1
                    '''
                    elif np.random.uniform() < 0.:  ###################sample some negative zero graph
                        self.batch_indices[last_index + negative_sample, 0] = random_entity
                        self.batch_indices[last_index + negative_sample, 1] = self.batch_indices[now_indice, 1]
                        self.batch_indices[last_index + negative_sample, 2] = self.batch_indices[now_indice, 2]
                        self.batch_values[last_index + negative_sample, :] = [-1]
                        negative_sample += 1
                    '''
                    if negative_sample>= last_index * invalid_valid_ratio:
                        break
                    random_entity = random.choice(self.unique_entities_train)
                    if ((self.batch_indices[now_indice, 0], self.batch_indices[now_indice, 1], random_entity) not in 
                      self.valid_triples_dict.keys())  and self.common_neighbor(neighbor_a, neighbor_b, self.batch_indices[now_indice, 0], random_entity) and (self.batch_indices[now_indice, 0]!=random_entity):
                        self.batch_indices[last_index + negative_sample, 2] = random_entity
                        self.batch_indices[last_index + negative_sample, 1] = self.batch_indices[now_indice, 1]
                        self.batch_indices[last_index + negative_sample, 0] = self.batch_indices[now_indice, 0]
                        self.batch_values[last_index + negative_sample, :] = [-1]
                        negative_sample += 1
                    '''
                    elif np.random.uniform() < 0.:
                        self.batch_indices[last_index + negative_sample, 2] = random_entity
                        self.batch_indices[last_index + negative_sample, 1] = self.batch_indices[now_indice, 1]
                        self.batch_indices[last_index + negative_sample, 0] = self.batch_indices[now_indice, 0]
                        self.batch_values[last_index + negative_sample, :] = [-1]
                        negative_sample += 1
                    '''
            return self.batch_indices, self.batch_values


    def get_total_train_data2(self, invalid_valid_ratio, neighbor_a, neighbor_b):
            batch_size = len(self.train_indices)
            self.batch_indices = np.empty((batch_size * (invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty((batch_size * (invalid_valid_ratio + 1), 1)).astype(np.float32)

            self.batch_indices[:batch_size, :] = self.train_indices
            self.batch_values[:batch_size, :] = self.train_values

            last_index = batch_size
            
            if invalid_valid_ratio > 0:
 
                self.batch_indices[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(self.batch_indices[:last_index, :], (invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(self.batch_values[:last_index, :], (invalid_valid_ratio, 1))
                negative_sample = 0
                while negative_sample < last_index * invalid_valid_ratio:
                    if negative_sample%20 == 0:
                        print(negative_sample)
                    now_indice = random.choice([i for i in range(last_index)])
                    random_entity = random.choice(self.unique_entities_train)

                    if (random_entity, self.batch_indices[now_indice, 1], self.batch_indices[now_indice, 2]) not in self.valid_triples_dict.keys() and (not random_entity==self.batch_indices[now_indice, 2]):
                        self.batch_indices[last_index + negative_sample, 0] = random_entity
                        self.batch_indices[last_index + negative_sample, 1] = self.batch_indices[now_indice, 1]
                        self.batch_indices[last_index + negative_sample, 2] = self.batch_indices[now_indice, 2]
                        self.batch_values[last_index + negative_sample, :] = [-1]
                        negative_sample += 1
                    if negative_sample>= last_index * invalid_valid_ratio:
                        break
                    random_entity = random.choice(self.unique_entities_train)

                    if (self.batch_indices[now_indice, 0], self.batch_indices[now_indice, 1], random_entity) not in  self.valid_triples_dict.keys() and (not random_entity==self.batch_indices[now_indice, 0]):
                        self.batch_indices[last_index + negative_sample, 2] = random_entity
                        self.batch_indices[last_index + negative_sample, 1] = self.batch_indices[now_indice, 1]
                        self.batch_indices[last_index + negative_sample, 0] = self.batch_indices[now_indice, 0]
                        self.batch_values[last_index + negative_sample, :] = [-1]
                        negative_sample += 1
            return self.batch_indices, self.batch_values



    def get_total_val_data(self, invalid_valid_ratio, neighbor_a, neighbor_b):
            batch_size = len(self.validation_indices)
            self.batch_indices = np.empty((batch_size * (invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty((batch_size * (invalid_valid_ratio + 1), 1)).astype(np.float32)

            self.batch_indices[:batch_size, :] = self.validation_indices
            self.batch_values[:batch_size, :] = self.validation_values

            last_index = batch_size
            
            if invalid_valid_ratio > 0:

                self.batch_indices[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(self.batch_indices[:last_index, :], (invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(self.batch_values[:last_index, :], (invalid_valid_ratio, 1))
                negative_sample = 0
                while negative_sample < last_index * invalid_valid_ratio:
                    if negative_sample%20 == 0:
                        print(negative_sample)
                    now_indice = random.choice([i for i in range(last_index)])
                    random_entity = random.choice(self.unique_entities_train)
                    if (random_entity, self.batch_indices[now_indice, 1], self.batch_indices[now_indice, 2]) not in self.valid_triples_dict.keys() and (self.batch_indices[now_indice, 2]!=random_entity)   and self.common_neighbor(neighbor_a, neighbor_b, random_entity, self.batch_indices[now_indice, 2]):
                        self.batch_indices[last_index + negative_sample, 0] = random_entity
                        self.batch_indices[last_index + negative_sample, 1] = self.batch_indices[now_indice, 1]
                        self.batch_indices[last_index + negative_sample, 2] = self.batch_indices[now_indice, 2]
                        self.batch_values[last_index + negative_sample, :] = [-1]


                    if negative_sample>= last_index * invalid_valid_ratio:
                        break
                    random_entity = random.choice(self.unique_entities_train)
                    if (self.batch_indices[now_indice, 0], self.batch_indices[now_indice, 1], random_entity) not in self.valid_triples_dict.keys()   and (self.batch_indices[now_indice, 0]!=random_entity)   and self.common_neighbor(neighbor_a, neighbor_b, self.batch_indices[now_indice, 0], random_entity):
                        self.batch_indices[last_index + negative_sample, 2] = random_entity
                        self.batch_indices[last_index + negative_sample, 1] = self.batch_indices[now_indice, 1]
                        self.batch_indices[last_index + negative_sample, 0] = self.batch_indices[now_indice, 0]
                        self.batch_values[last_index + negative_sample, :] = [-1]
                        negative_sample += 1


            return self.batch_indices, self.batch_values




    def get_total_val_data2(self, invalid_valid_ratio, neighbor_a, neighbor_b):
            batch_size = len(self.validation_indices)
            self.batch_indices = np.empty((batch_size * (invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty((batch_size * (invalid_valid_ratio + 1), 1)).astype(np.float32)

            self.batch_indices[:batch_size, :] = self.validation_indices
            self.batch_values[:batch_size, :] = self.validation_values

            last_index = batch_size
            
            if invalid_valid_ratio > 0:

                self.batch_indices[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(self.batch_indices[:last_index, :], (invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(self.batch_values[:last_index, :], (invalid_valid_ratio, 1))
                negative_sample = 0
                while negative_sample < last_index * invalid_valid_ratio:
                    if negative_sample%20 == 0:
                        print(negative_sample)
                    now_indice = random.choice([i for i in range(last_index)])
                    random_entity = random.choice(self.unique_entities_train)
                    if (random_entity, self.batch_indices[now_indice, 1], self.batch_indices[now_indice, 2]) not in self.valid_triples_dict.keys() and (self.batch_indices[now_indice, 2]!=random_entity):
                        self.batch_indices[last_index + negative_sample, 0] = random_entity
                        self.batch_indices[last_index + negative_sample, 1] = self.batch_indices[now_indice, 1]
                        self.batch_indices[last_index + negative_sample, 2] = self.batch_indices[now_indice, 2]
                        self.batch_values[last_index + negative_sample, :] = [-1]


                    if negative_sample>= last_index * invalid_valid_ratio:
                        break
                    random_entity = random.choice(self.unique_entities_train)
                    if (self.batch_indices[now_indice, 0], self.batch_indices[now_indice, 1], random_entity) not in self.valid_triples_dict.keys()   and (self.batch_indices[now_indice, 0]!=random_entity):
                        self.batch_indices[last_index + negative_sample, 2] = random_entity
                        self.batch_indices[last_index + negative_sample, 1] = self.batch_indices[now_indice, 1]
                        self.batch_indices[last_index + negative_sample, 0] = self.batch_indices[now_indice, 0]
                        self.batch_values[last_index + negative_sample, :] = [-1]
                        negative_sample += 1


            return self.batch_indices, self.batch_values




    def get_test_all_data(self, unique_entities, neighbor_a, neighbor_b):

            start_time = time.time()

            indices = [i for i in range(len(self.test_indices))]
            batch_indices = self.test_indices[indices, :]
            print("Sampled indices")
            print("test set length ", len(self.test_indices))
            entity_list = list(unique_entities)   ########
            total_test_head = [];  total_test_tail = []
            for i in range(batch_indices.shape[0]):
                start_time_it = time.time()
                new_x_batch_head = np.tile(batch_indices[i, :], (len(entity_list), 1))    ############## to be determined
                new_x_batch_tail = np.tile(batch_indices[i, :], (len(entity_list), 1))

                if(batch_indices[i, 0] not in unique_entities or batch_indices[i, 2] not in unique_entities):
                    continue

                new_x_batch_head[:, 0] = entity_list
                new_x_batch_tail[:, 2] = entity_list

                last_index_head = []  # array of already existing triples
                last_index_tail = []
                for tmp_index in range(len(new_x_batch_head)):
                    temp_triple_head = (new_x_batch_head[tmp_index][0], new_x_batch_head[tmp_index][1], new_x_batch_head[tmp_index][2])
                    if temp_triple_head in self.valid_triples_dict.keys() or (not self.common_neighbor(neighbor_a, neighbor_b, new_x_batch_head[tmp_index][0], new_x_batch_head[tmp_index][2])):
                        last_index_head.append(tmp_index)

                    temp_triple_tail = (new_x_batch_tail[tmp_index][0], new_x_batch_tail[tmp_index][1],  new_x_batch_tail[tmp_index][2])
                    if temp_triple_tail in self.valid_triples_dict.keys() or (not self.common_neighbor(neighbor_a, neighbor_b, new_x_batch_tail[tmp_index][0], new_x_batch_tail[tmp_index][2])):
                        last_index_tail.append(tmp_index)

                # Deleting already existing triples, leftover triples are invalid, according
                # to train, validation and test data
                # Note, all of them maynot be actually invalid
                new_x_batch_head = np.delete(new_x_batch_head, last_index_head, axis=0)
                new_x_batch_tail = np.delete(new_x_batch_tail, last_index_tail, axis=0)

                # adding the current valid triples to the top, i.e, index 0
                new_x_batch_head = np.insert(new_x_batch_head, 0, batch_indices[i], axis=0)
                new_x_batch_tail = np.insert(new_x_batch_tail, 0, batch_indices[i], axis=0)
                total_test_head.append(new_x_batch_head)
                total_test_tail.append(new_x_batch_tail)
            end_time = time.time()
            print('time for extracting all testing data: ', end_time - start_time)
            return total_test_head, total_test_tail




    def get_test_all_data2(self, unique_entities, neighbor_a, neighbor_b, percent = 1, rank_number=100):

            start_time = time.time()
            indices = [i for i in range(len(self.test_indices))]
            indices = np.random.choice(indices, int(len(self.test_indices) * percent), replace=False)
            batch_indices = self.test_indices[indices, :]
            print("Sampled indices")
            print("test set length ", len(batch_indices))
            entity_list = list(unique_entities)   ########
            total_test_head = [];  total_test_tail = []

            for i in range(batch_indices.shape[0]):
                start_time_it = time.time()
                new_x_batch_head = []    ############## to be determined
                new_x_batch_tail = []
                current_triple = batch_indices[i]
                new_x_batch_head.append(current_triple)
                new_x_batch_tail.append(current_triple)
                kk = 0
                for j in entity_list:
                    if (j, current_triple[1], current_triple[2]) not in self.valid_triples_dict.keys() and self.common_neighbor(neighbor_a, neighbor_b, j,  current_triple[2]):
                       new_x_batch_head.append([j, current_triple[1], current_triple[2]])
                       kk += 1
                       if kk>=rank_number:
                          break
                new_x_batch_head = np.array(new_x_batch_head)
                print(kk)
                kk = 0
                for j in entity_list:
                    if (current_triple[0], current_triple[1], j) not in self.valid_triples_dict.keys() and self.common_neighbor(neighbor_a, neighbor_b,  current_triple[0], j):
                       new_x_batch_tail.append([current_triple[0], current_triple[1], j])
                       kk += 1
                       if kk>=rank_number:
                          break
                new_x_batch_tail = np.array(new_x_batch_tail)
                print(kk)
                total_test_head.append(new_x_batch_head)
                total_test_tail.append(new_x_batch_tail)
            end_time = time.time()
            print('time for extracting all testing data: ', end_time - start_time)
            return total_test_head, total_test_tail



    def get_test_all_data3(self, unique_entities, neighbor_a, neighbor_b, percent = 1, rank_number=1000):

            start_time = time.time()
            indices = [i for i in range(len(self.test_indices))]
            indices = np.random.choice(indices, int(len(self.test_indices) * percent), replace=False)
            batch_indices = self.test_indices[indices, :]
            print("Sampled indices")
            print("test set length ", len(batch_indices))
            entity_list = list(unique_entities)   ########
            total_test_head = [];  total_test_tail = []

            for i in range(batch_indices.shape[0]):
                start_time_it = time.time()
                new_x_batch_head = []    ############## to be determined
                new_x_batch_tail = []
                current_triple = batch_indices[i]
                new_x_batch_head.append(current_triple)
                new_x_batch_tail.append(current_triple)
                kk = 0; existed = []
               
                for j in entity_list:
                    if (j, current_triple[1], current_triple[2]) not in self.valid_triples_dict.keys() and self.common_neighbor(neighbor_a, neighbor_b, j,  current_triple[2]) and (j != current_triple[2]):
                   # if (j, current_triple[1], current_triple[2]) not in self.valid_triples_dict.keys():
                       new_x_batch_head.append([j, current_triple[1], current_triple[2]])
                       kk += 1; existed.append(j)
                       if kk>=rank_number:
                          break
                '''
                while (kk < rank_number):
                     j = np.random.choice(len(entity_list))
                     j = entity_list[j]
                     if ((j, current_triple[1], current_triple[2]) not in self.valid_triples_dict.keys()) and (j not in existed) and (j != current_triple[2]):
                         new_x_batch_head.append([j, current_triple[1], current_triple[2]])
                         kk += 1; existed.append(j)
                '''
                if len(new_x_batch_head)>50:
                       new_x_batch_head2 = []
                       new_x_batch_head2.append(new_x_batch_head[0])
                       indices = [i for i in range(1, len(new_x_batch_head))]
                       index = list(np.random.choice(indices, 49, replace=False))
                       for kkk in index:
                           new_x_batch_head2.append(new_x_batch_head[kkk])
                      # new_x_batch_head2 = new_x_batch_head2 + new_x_batch_head[index]
                       print(len(new_x_batch_head2))
                       new_x_batch_head = new_x_batch_head2
                     
                new_x_batch_head = np.array(new_x_batch_head)
             #   print(kk)

                kk = 0; existed = []
                
                for j in entity_list:
                    if (current_triple[0], current_triple[1], j) not in self.valid_triples_dict.keys() and self.common_neighbor(neighbor_a, neighbor_b,  current_triple[0], j) and (j != current_triple[0]):
                   # if (current_triple[0], current_triple[1], j) not in self.valid_triples_dict.keys():
                       new_x_batch_tail.append([current_triple[0], current_triple[1], j])
                       kk += 1; existed.append(j)
                       if kk>=rank_number:
                          break
 


                if len(new_x_batch_tail)>50:
                       new_x_batch_tail2 = []
                       new_x_batch_tail2.append(new_x_batch_tail[0])
                       indices = [i for i in range(1, len(new_x_batch_tail))]
                       index = list(np.random.choice(indices, 49, replace=False))
                       for kkk in index:
                           new_x_batch_tail2.append(new_x_batch_tail[kkk])
                       print(len(new_x_batch_tail2))
                       new_x_batch_tail = new_x_batch_tail2


                new_x_batch_tail = np.array(new_x_batch_tail)
               # print(kk)
                total_test_head.append(new_x_batch_head)
                total_test_tail.append(new_x_batch_tail)
            end_time = time.time()
            print('time for extracting all testing data: ', end_time - start_time)
            return total_test_head, total_test_tail




    def get_test_all_data4(self, unique_entities, neighbor_a, neighbor_b, percent = 1, rank_number=50):

            start_time = time.time()
            indices = [i for i in range(len(self.test_indices))]
            indices = np.random.choice(indices, int(len(self.test_indices) * percent), replace=False)
            batch_indices = self.test_indices[indices, :]
            print("Sampled indices")
            print("test set length ", len(batch_indices))
            entity_list = list(unique_entities)   ########
            total_test_head = [];  total_test_tail = []

            for i in range(batch_indices.shape[0]):
                start_time_it = time.time()
                new_x_batch_head = []    ############## to be determined
                new_x_batch_tail = []
                current_triple = batch_indices[i]
                new_x_batch_head.append(current_triple)
                new_x_batch_tail.append(current_triple)
                kk = 1; existed = []
               
  
               
                while (kk < rank_number):
                     j = np.random.choice(len(entity_list))
                     j = entity_list[j]
                     if ((j, current_triple[1], current_triple[2]) not in self.valid_triples_dict.keys()) and (j not in existed) and (j != current_triple[2]):
                         new_x_batch_head.append([j, current_triple[1], current_triple[2]])
                         kk += 1; existed.append(j)
               
                       
                new_x_batch_head = np.array(new_x_batch_head)
             #   print(kk)

                kk = 1; existed = []
                
                
                while (kk < rank_number):
                     j = np.random.choice(len(entity_list))
                     j = entity_list[j]
                     if ((current_triple[0], current_triple[1], j) not in self.valid_triples_dict.keys()) and (j not in existed) and (j != current_triple[0]):
                         new_x_batch_tail.append([current_triple[0], current_triple[1], j])
                         kk += 1; existed.append(j)
               
                new_x_batch_tail = np.array(new_x_batch_tail)
               # print(kk)
                total_test_head.append(new_x_batch_head)
                total_test_tail.append(new_x_batch_tail)
            end_time = time.time()
            print('time for extracting all testing data: ', end_time - start_time)
            return total_test_head, total_test_tail




    def get_test_all_data5(self, unique_entities, neighbor_a, neighbor_b, percent = 1, rank_number=1000):

            start_time = time.time()
            indices = [i for i in range(len(self.test_indices))]
            indices = np.random.choice(indices, int(len(self.test_indices) * percent), replace=False)
            batch_indices = self.test_indices[indices, :]
            print("Sampled indices")
            print("test set length ", len(batch_indices))
            entity_list = list(unique_entities)   ########
            total_test_head = [];  total_test_tail = []
            all_test_triplet = []

            for i in range(batch_indices.shape[0]):
                start_time_it = time.time()
                new_x_batch_head = []    ############## to be determined
                new_x_batch_tail = []
                current_triple = batch_indices[i]
                new_x_batch_head.append(current_triple)
                new_x_batch_tail.append(current_triple)
                kk = 0; existed = []
               
                for j in entity_list:
                    if (j, current_triple[1], current_triple[2]) not in self.valid_triples_dict.keys() and self.common_neighbor(neighbor_a, neighbor_b, j,  current_triple[2]) and (j != current_triple[2]):
                   # if (j, current_triple[1], current_triple[2]) not in self.valid_triples_dict.keys():
                       new_x_batch_head.append([j, current_triple[1], current_triple[2]])
                       kk += 1; existed.append(j)
                       if kk>=rank_number:
                          break
                '''
                while (kk < rank_number):
                     j = np.random.choice(len(entity_list))
                     j = entity_list[j]
                     if ((j, current_triple[1], current_triple[2]) not in self.valid_triples_dict.keys()) and (j not in existed) and (j != current_triple[2]):
                         new_x_batch_head.append([j, current_triple[1], current_triple[2]])
                         kk += 1; existed.append(j)
                '''
                if len(new_x_batch_head)>50:
                       new_x_batch_head2 = []
                       new_x_batch_head2.append(new_x_batch_head[0])
                       all_test_triplet.append(new_x_batch_head[0])
                       indices = [i for i in range(1, len(new_x_batch_head))]
                       index = list(np.random.choice(indices, 49, replace=False))
                       for kkk in index:
                           new_x_batch_head2.append(new_x_batch_head[kkk])
                           all_test_triplet.append(new_x_batch_head[kkk])
                      # new_x_batch_head2 = new_x_batch_head2 + new_x_batch_head[index]
                       print(len(new_x_batch_head2))
                       new_x_batch_head = new_x_batch_head2
                     
                new_x_batch_head = np.array(new_x_batch_head)
             #   print(kk)

                kk = 0; existed = []
                
                for j in entity_list:
                    if (current_triple[0], current_triple[1], j) not in self.valid_triples_dict.keys() and self.common_neighbor(neighbor_a, neighbor_b,  current_triple[0], j) and (j != current_triple[0]):
                   # if (current_triple[0], current_triple[1], j) not in self.valid_triples_dict.keys():
                       new_x_batch_tail.append([current_triple[0], current_triple[1], j])
                       kk += 1; existed.append(j)
                       if kk>=rank_number:
                          break
                '''
                while (kk < rank_number):
                     j = np.random.choice(len(entity_list))
                     j = entity_list[j]
                     if ((current_triple[0], current_triple[1], j) not in self.valid_triples_dict.keys()) and (j not in existed) and (j != current_triple[0]):
                         new_x_batch_tail.append([current_triple[0], current_triple[1], j])
                         kk += 1; existed.append(j)
                '''


                if len(new_x_batch_tail)>50:
                       new_x_batch_tail2 = []
                       new_x_batch_tail2.append(new_x_batch_tail[0])
                       all_test_triplet.append(new_x_batch_tail[0])
                       indices = [i for i in range(1, len(new_x_batch_tail))]
                       index = list(np.random.choice(indices, 49, replace=False))
                       for kkk in index:
                           new_x_batch_tail2.append(new_x_batch_tail[kkk])
                           all_test_triplet.append(new_x_batch_tail[kkk])

                       print(len(new_x_batch_tail2))
                       new_x_batch_tail = new_x_batch_tail2


                new_x_batch_tail = np.array(new_x_batch_tail)
               # print(kk)
                total_test_head.append(new_x_batch_head)
                total_test_tail.append(new_x_batch_tail)
            end_time = time.time()
            print('time for extracting all testing data: ', end_time - start_time)
            return total_test_head, total_test_tail, np.array(all_test_triplet)






    def common_neighbor(self, neighbor_a, neighbor_b, a, b):
     #   common = [val for val in neighbor_a if val in neighbor_b]
        neighbor_a = neighbor_a[a]; neighbor_b = neighbor_b[b]
        common = False
        for x in neighbor_a:
            for y in neighbor_b:
                if x==y and x != a and x != b:
                   common = True
                   break
            if common:
                break
        return common
        '''
        if len(common):
           return True
        else:
           return False
        ''' 
 
    def get_batch_nhop_neighbors_undirected(self, batch_sources, graph, graph_inverse, nbd_size=2):
        batch_source_triples = {}
        batch_neighbor = {}
        batch_distance = {}; neighbor_inverse_1hop = {}
        print("length of unique_entities ", len(batch_sources))

        start = time.time()
        pool = mp.Pool(mp.cpu_count())
        results = pool.map_async(parallel_worker3, [(graph, graph_inverse, source, nbd_size) for source in batch_sources])
        remaining = results._number_left
       # pbar = tqdm(total=remaining)
        while True:
           # pbar.update(remaining - results._number_left)
            if results.ready(): break
            remaining = results._number_left
            time.sleep(1)
        results = results.get()
        pool.close()
      #  pbar.close()
        for neighbor, triples, distance, inverse_1hop, source in results:
             batch_source_triples[source] = triples
             batch_neighbor[source] = neighbor
             batch_distance[source] = distance
             neighbor_inverse_1hop[source] = inverse_1hop
        end = time.time()

        print('Time for extracting batch neighbor: ', end - start)

        return batch_neighbor, batch_source_triples, batch_distance,  neighbor_inverse_1hop


    def get_graph2(self, adj_matrix, unique_entities):
        graph = np.zeros([unique_entities, unique_entities])
        
        all_tiples = torch.cat([adj_matrix[0].transpose(
            0, 1), adj_matrix[1].unsqueeze(1)], dim=1)

        for data in all_tiples:
            source = data[1].data.item()
            target = data[0].data.item()
            value = data[2].data.item()
            graph[source][target] = value

        print("Graph created")
        return graph


    def get_graph(self, adj_matrix):
        graph = {}
        
        all_tiples = torch.cat([adj_matrix[0].transpose(
            0, 1), adj_matrix[1].unsqueeze(1)], dim=1)

        for data in all_tiples:
            source = data[1].data.item()
            target = data[0].data.item()
            value = data[2].data.item()

            if(source not in graph.keys()):
                graph[source] = {}
                graph[source][target] = value
            else:
                graph[source][target] = value
        print("Graph created")
        return graph




    def get_inverse_graph(self, adj_matrix):
        graph = {}
        
        all_tiples = torch.cat([adj_matrix[0].transpose(
            0, 1), adj_matrix[1].unsqueeze(1)], dim=1)

        for data in all_tiples:
            source = data[1].data.item()
            target = data[0].data.item()
            value = data[2].data.item()

            if(target not in graph.keys()):
                graph[target] = {}
                graph[target][source] = value
            else:
                graph[target][source] = value
        print("inverse graph created")
        return graph



    def get_batch_nhop_neighbors2(self, batch_sources, graph, graph_inverse, nbd_size=2):
        batch_source_triples = {}; inverse_source_triples = {}
        batch_neighbor = {}; inverse_neighbor = {}
        batch_distance = {}; inverse_distance = {}; neighbor_inverse_1hop = {}
        print("length of unique_entities ", len(batch_sources))

        start = time.time()
        pool = mp.Pool(mp.cpu_count())
        results = pool.map_async(parallel_worker2, [(graph, graph_inverse, source, nbd_size) for source in batch_sources])
        remaining = results._number_left
       # pbar = tqdm(total=remaining)
        while True:
           # pbar.update(remaining - results._number_left)
            if results.ready(): break
            remaining = results._number_left
            time.sleep(1)
        results = results.get()
        pool.close()
      #  pbar.close()
        for neighbor, triples, distance, neighbor_inverse, triples_inverse, distance_inverse, inverse_1hop, source in results:
             batch_source_triples[source] = triples
             batch_neighbor[source] = neighbor
             batch_distance[source] = distance
             inverse_source_triples[source] = triples_inverse
             inverse_neighbor[source] = neighbor_inverse
             inverse_distance[source] = distance_inverse
             neighbor_inverse_1hop[source] = inverse_1hop
        end = time.time()

        print('Time for extracting batch neighbor: ', end - start)

        return batch_neighbor, batch_source_triples, batch_distance, inverse_neighbor, inverse_source_triples, inverse_distance, neighbor_inverse_1hop



    def bfs(self, graph, source, nbd_size=2):
        visit = {}
        distance = {}
        parent = {}
        distance_lengths = {}

        visit[source] = 1
        distance[source] = 0
        parent[source] = (-1, -1)

        q = queue.Queue()
        q.put((source, -1))

        while(not q.empty()):
            top = q.get()
            if top[0] in graph.keys():
                for target in graph[top[0]].keys():
                    if(target in visit.keys()):
                        continue
                    else:
                        q.put((target, graph[top[0]][target]))

                        distance[target] = distance[top[0]] + 1

                        visit[target] = 1
                        if distance[target] > 2:
                            break
                        parent[target] = (top[0], graph[top[0]][target])

                        if distance[target] not in distance_lengths.keys():
                            distance_lengths[distance[target]] = 1

        neighbors = {}
        for target in visit.keys():
            if(distance[target] != nbd_size):
                continue
            edges = [-1, parent[target][1]]
            relations = []
            entities = [target]
            temp = target
            while(parent[temp] != (-1, -1)):       ###### if not source
                relations.append(parent[temp][1])
                entities.append(parent[temp][0])
                temp = parent[temp][0]

            if(distance[target] in neighbors.keys()):
                neighbors[distance[target]].append(
                    (tuple(relations), tuple(entities[:-1])))
            else:
                neighbors[distance[target]] = [
                    (tuple(relations), tuple(entities[:-1]))]    #######([edge_value], [target])

        return neighbors

    def get_further_neighbors(self, nbd_size=2):
        neighbors = {}
        start_time = time.time()
        print("length of graph keys is ", len(self.graph.keys()))
        for source in self.graph.keys():
            # st_time = time.time()
            temp_neighbors = self.bfs(self.graph, source, nbd_size)
            for distance in temp_neighbors.keys():
                if(source in neighbors.keys()):
                    if(distance in neighbors[source].keys()):
                        neighbors[source][distance].append(
                            temp_neighbors[distance])
                    else:
                        neighbors[source][distance] = temp_neighbors[distance]
                else:
                    neighbors[source] = {}
                    neighbors[source][distance] = temp_neighbors[distance]

        print("time taken ", time.time() - start_time)

        print("length of neighbors dict is ", len(neighbors))
        return neighbors

    def get_batch_nhop_neighbors_all(self, args, batch_sources, node_neighbors, nbd_size=2):
        batch_source_triples = []
        print("length of unique_entities ", len(batch_sources))
        count = 0
        for source in batch_sources:
            # randomly select from the list of neighbors
            if source in node_neighbors.keys():
                nhop_list = node_neighbors[source][nbd_size]

                for i, tup in enumerate(nhop_list):
                    if(args.partial_2hop and i >= 2):
                        break

                    count += 1
                    batch_source_triples.append([source, nhop_list[i][0][-1], nhop_list[i][0][0],
                                                 nhop_list[i][1][0]])     ######source, first_relation, last_relation, last_target 

        return np.array(batch_source_triples).astype(np.int32)

    def transe_scoring(self, batch_inputs, entity_embeddings, relation_embeddings):
        source_embeds = entity_embeddings[batch_inputs[:, 0]]
        relation_embeds = relation_embeddings[batch_inputs[:, 1]]
        tail_embeds = entity_embeddings[batch_inputs[:, 2]]
        x = source_embeds + relation_embed - tail_embeds
        x = torch.norm(x, p=1, dim=1)
        return x


from torch.autograd import Variable
def get_our_validation_pred_random(args, model, total_test_head, total_test_tail, subgraph, eval_type='random'):
    average_hits_at_100_head, average_hits_at_100_tail = [], []
    average_hits_at_ten_head, average_hits_at_ten_tail = [], []
    average_hits_at_three_head, average_hits_at_three_tail = [], []
    average_hits_at_one_head, average_hits_at_one_tail = [], []
    average_mean_rank_head, average_mean_rank_tail = [], []
    average_mean_recip_rank_head, average_mean_recip_rank_tail = [], []
    log = open('./log/%s_error_%s.log' % (args.prefix, args.dt), 'w')
    for iters in range(1):
        start_time = time.time()

        ranks_head, ranks_tail = [], []
        reciprocal_ranks_head, reciprocal_ranks_tail = [], []
        hits_at_100_head, hits_at_100_tail = 0, 0
        hits_at_ten_head, hits_at_ten_tail = 0, 0
        hits_at_three_head, hits_at_three_tail = 0, 0
        hits_at_one_head, hits_at_one_tail = 0, 0

        for i in range(len(total_test_head)):
          #  print(len(ranks_head))
            start_time_it = time.time()
            new_x_batch_head = total_test_head[i].astype(np.int64)
            new_x_batch_tail = total_test_tail[i].astype(np.int64)

            if new_x_batch_tail.shape[0] < 2 or new_x_batch_head.shape[0] < 2:
                 continue
            true_teiple = new_x_batch_tail[0, :]
            new_x_batch_head = new_x_batch_head[1:, :]
            new_x_batch_tail = new_x_batch_tail[1:, :]
            if eval_type == 'org':
                new_x_batch_head = np.insert(new_x_batch_head, 0, true_teiple, axis=0)
                new_x_batch_tail = np.insert(new_x_batch_tail, 0, true_teiple, axis=0)

            elif eval_type in ['last']:
                new_x_batch_head = np.concatenate([new_x_batch_head, [true_teiple]], 0)
                new_x_batch_tail = np.concatenate([new_x_batch_tail, [true_teiple]], 0)

            elif eval_type in ['random']:
                rand_head = np.random.randint(new_x_batch_head.shape[0])
                rand_tail = np.random.randint(new_x_batch_tail.shape[0])
                new_x_batch_head = np.insert(new_x_batch_head, rand_head, true_teiple, axis=0)
                new_x_batch_tail = np.insert(new_x_batch_tail, rand_tail, true_teiple, axis=0)

            import math
            # Have to do this, because it doesn't fit in memory

            scores_head = model(Variable(torch.LongTensor(new_x_batch_head)).cuda(), subgraph)
            # scores_head = scores_head.cpu().data.numpy()

            sorted_scores_head, sorted_indices_head = torch.sort(scores_head.view(-1), dim=-1, descending=True)
            # Just search for zeroth index in the sorted scores, we appended valid triple at top

            #  ranks_head.append(np.where(sorted_indices_head.cpu().numpy() == 0)[0][0] + 1)
            if eval_type == 'org':
                ranks_head.append(np.where(sorted_indices_head.cpu().numpy() == 0)[0][0] + 1)

            elif eval_type == 'last':
                ranks_head.append(np.where(sorted_indices_head.cpu().numpy() == sorted_indices_head.shape[0] - 1)[0][0] + 1)

            elif eval_type == 'random':
                ranks_head.append(np.where(sorted_indices_head.cpu().numpy() == rand_head)[0][0] + 1)

            reciprocal_ranks_head.append(1.0 / ranks_head[-1])

            scores_tail = model(Variable(torch.LongTensor(new_x_batch_tail)).cuda(), subgraph)
            #   scores_tail = scores_tail.cpu().data.numpy()

            sorted_scores_tail, sorted_indices_tail = torch.sort(scores_tail.view(-1), dim=-1, descending=True)

            # Just search for zeroth index in the sorted scores, we appended valid triple at top

            #    ranks_tail.append(np.where(sorted_indices_tail.cpu().numpy() == 0)[0][0] + 1)

            if eval_type == 'org':
                ranks_tail.append(np.where(sorted_indices_tail.cpu().numpy() == 0)[0][0] + 1)

            elif eval_type == 'last':
                ranks_tail.append(np.where(sorted_indices_tail.cpu().numpy() == sorted_indices_tail.shape[0] - 1)[0][0] + 1)

            elif eval_type == 'random':
                ranks_tail.append(np.where(sorted_indices_tail.cpu().numpy() == rand_tail)[0][0] + 1)

            reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])
          #  print("sample - ", ranks_head[-1], ranks_tail[-1])

        #     print('score tail ', scores_tail.data.cpu().numpy())
        #    print('score head ', scores_head.data.cpu().numpy())
        for i in range(len(ranks_head)):
            if ranks_head[i] <= 100:
                hits_at_100_head = hits_at_100_head + 1
            if ranks_head[i] <= 10:
                hits_at_ten_head = hits_at_ten_head + 1
            if ranks_head[i] <= 3:
                hits_at_three_head = hits_at_three_head + 1
            if ranks_head[i] == 1:
                hits_at_one_head = hits_at_one_head + 1

        for i in range(len(ranks_tail)):
            if ranks_tail[i] <= 100:
                hits_at_100_tail = hits_at_100_tail + 1
            if ranks_tail[i] <= 10:
                hits_at_ten_tail = hits_at_ten_tail + 1
            if ranks_tail[i] <= 3:
                hits_at_three_tail = hits_at_three_tail + 1
            if ranks_tail[i] == 1:
                hits_at_one_tail = hits_at_one_tail + 1
        '''
        assert len(ranks_head) == len(reciprocal_ranks_head)
        assert len(ranks_tail) == len(reciprocal_ranks_tail)
        print("here {}".format(len(ranks_head)))
        print("\nCurrent iteration time {}".format(time.time() - start_time))
        print("Stats for replacing head are -> ")
        print("Current iteration Hits@100 are {}".format(
            hits_at_100_head / float(len(ranks_head))))
        print("Current iteration Hits@10 are {}".format(
            hits_at_ten_head / len(ranks_head)))
        print("Current iteration Hits@3 are {}".format(
            hits_at_three_head / len(ranks_head)))
        print("Current iteration Hits@1 are {}".format(
            hits_at_one_head / len(ranks_head)))
        print("Current iteration Mean rank {}".format(
            sum(ranks_head) / len(ranks_head)))
        print("Current iteration Mean Reciprocal Rank {}".format(
            sum(reciprocal_ranks_head) / len(reciprocal_ranks_head)))

        print("\nStats for replacing tail are -> ")
        print("Current iteration Hits@100 are {}".format(
            hits_at_100_tail / len(ranks_head)))
        print("Current iteration Hits@10 are {}".format(
            hits_at_ten_tail / len(ranks_head)))
        print("Current iteration Hits@3 are {}".format(
            hits_at_three_tail / len(ranks_head)))
        print("Current iteration Hits@1 are {}".format(
            hits_at_one_tail / len(ranks_head)))
        print("Current iteration Mean rank {}".format(
            sum(ranks_tail) / len(ranks_tail)))
        print("Current iteration Mean Reciprocal Rank {}".format(
            sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)))
        '''
        average_hits_at_100_head.append(
            hits_at_100_head / len(ranks_head))
        average_hits_at_ten_head.append(
            hits_at_ten_head / len(ranks_head))
        average_hits_at_three_head.append(
            hits_at_three_head / len(ranks_head))
        average_hits_at_one_head.append(
            hits_at_one_head / len(ranks_head))
        average_mean_rank_head.append(sum(ranks_head) / len(ranks_head))
        average_mean_recip_rank_head.append(
            sum(reciprocal_ranks_head) / len(reciprocal_ranks_head))

        average_hits_at_100_tail.append(
            hits_at_100_tail / len(ranks_head))
        average_hits_at_ten_tail.append(
            hits_at_ten_tail / len(ranks_head))
        average_hits_at_three_tail.append(
            hits_at_three_tail / len(ranks_head))
        average_hits_at_one_tail.append(
            hits_at_one_tail / len(ranks_head))
        average_mean_rank_tail.append(sum(ranks_tail) / len(ranks_tail))
        average_mean_recip_rank_tail.append(
            sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail))
    '''
    print("\nAveraged stats for replacing head are -> ")
    print("Hits@100 are {}".format(
        sum(average_hits_at_100_head) / len(average_hits_at_100_head)))
    print("Hits@10 are {}".format(
        sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)))
    print("Hits@3 are {}".format(
        sum(average_hits_at_three_head) / len(average_hits_at_three_head)))
    print("Hits@1 are {}".format(
        sum(average_hits_at_one_head) / len(average_hits_at_one_head)))
    print("Mean rank {}".format(
        sum(average_mean_rank_head) / len(average_mean_rank_head)))
    print("Mean Reciprocal Rank {}".format(
        sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head)))

    print("\nAveraged stats for replacing tail are -> ")
    print("Hits@100 are {}".format(
        sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)))
    print("Hits@10 are {}".format(
        sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)))
    print("Hits@3 are {}".format(
        sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)))
    print("Hits@1 are {}".format(
        sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)))
    print("Mean rank {}".format(
        sum(average_mean_rank_tail) / len(average_mean_rank_tail)))
    print("Mean Reciprocal Rank {}".format(
        sum(average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)))
    '''
    cumulative_hits_100 = (sum(average_hits_at_100_head) / len(average_hits_at_100_head)
                           + sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)) / 2
    cumulative_hits_ten = (sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)
                           + sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)) / 2
    cumulative_hits_three = (sum(average_hits_at_three_head) / len(average_hits_at_three_head)
                             + sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)) / 2
    cumulative_hits_one = (sum(average_hits_at_one_head) / len(average_hits_at_one_head)
                           + sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)) / 2
    cumulative_mean_rank = (sum(average_mean_rank_head) / len(average_mean_rank_head)
                            + sum(average_mean_rank_tail) / len(average_mean_rank_tail)) / 2
    cumulative_mean_recip_rank = (sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head) + sum(
        average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)) / 2

    print("\nCumulative stats are -> ")
    print("Hits@100 are {}".format(cumulative_hits_100))
    print("Hits@10 are {}".format(cumulative_hits_ten))
    print("Hits@3 are {}".format(cumulative_hits_three))
    print("Hits@1 are {}".format(cumulative_hits_one))
    print("Mean rank {}".format(cumulative_mean_rank))
    print("Mean Reciprocal Rank {}".format(cumulative_mean_recip_rank))

    log.close()

