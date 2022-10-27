import torch
# from torchviz import make_dot, make_dot_from_trace
#from models import SpKBGATModified, SpKBGATConvOnly
from layers import ConvKB
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy

from create_batch_inductive2 import Corpus
from torch.utils.data import Dataset
import random
import argparse
import os
import logging
import time
import pickle


CUDA = torch.cuda.is_available()



def process_data(data, label, indice):
    # parse the input args
    class pharmKG(Dataset):
        '''
        PyTorch Dataset for pharmKG, don't need to change this
        '''

        def __init__(self, triple, labels):
            self.triple = triple
            self.labels = labels

        def __getitem__(self, idx):
            return [self.triple[idx, :],  self.labels[idx, :]]

        def __len__(self):
            return self.triple.shape[0]


    print('Before tranforming: ', data.shape)
    data = data.astype(np.int64)
    label = label.astype(np.float32)
    '''
    positive_triple = np.expand_dims(data[:indice, :], axis = 0)
    negative_triple = data[indice:, :]
    ratio = negative_triple.shape[0]//indice
    negative_triple = np.reshape(negative_triple, (ratio, indice, 3))
    total_triple = np.concatenate([positive_triple, negative_triple], axis=0)
    total_triple = np.swapaxes(total_triple, 0 ,1)


    positive_label = np.expand_dims(label[:indice, :], axis = 0)
    negative_label = label[indice:, :]
    negative_label = np.reshape(negative_label, (ratio, indice, 1))
    total_label = np.concatenate([positive_label, negative_label], axis=0)
    total_label = np.swapaxes(total_label, 0 ,1)
    ''' 
    total_triple = data; total_label = label
    train_set = pharmKG(total_triple, total_label)


    print('After tranforming: ', total_triple.shape)
  #  print(total_triple[0, :, :], total_label[0, :, :])
    
    return train_set





def process_data3(data, label, indice):
    # parse the input args
    class pharmKG(Dataset):
        '''
        PyTorch Dataset for pharmKG, don't need to change this
        '''

        def __init__(self, triple, labels):
            self.triple = triple
            self.labels = labels

        def __getitem__(self, idx):
            return [self.triple[idx, :],  self.labels[idx, :]]

        def __len__(self):
            return self.triple.shape[0]


    print('Before tranforming: ', data.shape)
    data = data.astype(np.int64)
    label = label.astype(np.float32)
    '''
    positive_triple = np.expand_dims(data[:indice, :], axis = 0)
    negative_triple = data[indice:, :]
    ratio = negative_triple.shape[0]//indice
    negative_triple = np.reshape(negative_triple, (ratio, indice, 3))
    total_triple = np.concatenate([positive_triple, negative_triple], axis=0)
    total_triple = np.swapaxes(total_triple, 0 ,1)


    positive_label = np.expand_dims(label[:indice, :], axis = 0)
    negative_label = label[indice:, :]
    negative_label = np.reshape(negative_label, (ratio, indice, 1))
    total_label = np.concatenate([positive_label, negative_label], axis=0)
    total_label = np.swapaxes(total_label, 0 ,1)
    ''' 
    total_triple = data; total_label = label

    total_relation = list(set(list(total_triple[:, 1])))
    
    total_triple2 = {}
    for i in range(len(total_triple)):
        re = total_triple[i, 1]
        if re not in total_triple2.keys():
            total_triple2[re] = []
        triple_label = np.concatenate((np.expand_dims(total_triple[i, :], axis=0), np.expand_dims(total_label[i, :], axis=0)), axis = -1)
        total_triple2[re].append(triple_label)

    threshold = len(total_triple)/len(total_relation) * 0.1
  #  threshold = 20
    large_relation = []; few_relation = []

    for i in range(len(total_relation)):
        relation_new = total_relation[i]
        relation = len(total_triple2[relation_new])
        if relation > threshold:
            large_relation.append(relation_new)
        else:
            few_relation.append(relation_new)
    
    return total_triple2, large_relation, few_relation



def process_data4(data, label, indice):
    # parse the input args
    class pharmKG(Dataset):
        '''
        PyTorch Dataset for pharmKG, don't need to change this
        '''

        def __init__(self, triple, labels):
            self.triple = triple
            self.labels = labels

        def __getitem__(self, idx):
            return [self.triple[idx, :],  self.labels[idx, :]]

        def __len__(self):
            return self.triple.shape[0]


    print('Before tranforming: ', data.shape)
    data = data.astype(np.int64)
    label = label.astype(np.float32)

    total_triple = data; total_label = label

    total_relation = list(set(list(total_triple[:, 1])))
    
    total_triple2 = {}
    for i in range(len(total_triple)):
        re = total_triple[i, 1]
        if re not in total_triple2.keys():
            total_triple2[re] = []
        triple_label = np.concatenate((np.expand_dims(total_triple[i, :], axis=0), np.expand_dims(total_label[i, :], axis=0)), axis = -1)
        total_triple2[re].append(triple_label)

  #  threshold = len(total_triple)/len(total_relation) * 0.1
    threshold = 20
    large_relation = []; few_relation = []

    for i in range(len(total_relation)):
        relation_new = total_relation[i]
        relation = len(total_triple2[relation_new])
        if relation < 50:
            large_relation.append(relation_new)
            if relation < threshold:
                few_relation.append(relation_new)
        else:
            print('large relation exists!!!!!!!!!!!!!!!!!!!!')
    
    return total_triple2, large_relation, few_relation




def process_data5(data, label, indice):
    # parse the input args
    class pharmKG(Dataset):
        '''
        PyTorch Dataset for pharmKG, don't need to change this
        '''

        def __init__(self, triple, labels):
            self.triple = triple
            self.labels = labels

        def __getitem__(self, idx):
            return [self.triple[idx, :],  self.labels[idx, :]]

        def __len__(self):
            return self.triple.shape[0]


    print('Before tranforming: ', data.shape)
    data = data.astype(np.int64)
    label = label.astype(np.float32)

    total_triple = data; total_label = label

    total_relation = list(set(list(total_triple[:, 1])))
    
    total_triple2 = {}
    for i in range(len(total_triple)):
        re = total_triple[i, 1]
        if re not in total_triple2.keys():
            total_triple2[re] = []
        triple_label = np.concatenate((np.expand_dims(total_triple[i, :], axis=0), np.expand_dims(total_label[i, :], axis=0)), axis = -1)
        total_triple2[re].append(triple_label)

    threshold = len(total_triple)/len(total_relation) * 0.2
    k = 6
  #  threshold = 20
    large_relation = []; few_relation = []

    for i in range(len(total_relation)):
        relation_new = total_relation[i]
        relation = len(total_triple2[relation_new])
        if relation > threshold:
            large_relation.append(relation_new)
        else:
            few_relation.append(relation_new)
            if relation > k-1:
                index = np.random.choice(len(total_triple2[relation_new]), k, replace=False)
                train_now = total_triple2[relation_new][index]
                total_triple2[relation_new] = train_now

    
    return total_triple2, large_relation, few_relation






def process_data_test(data, label, indice):
    # parse the input args
    class pharmKG(Dataset):
        '''
        PyTorch Dataset for pharmKG, don't need to change this
        '''

        def __init__(self, triple, labels):
            self.triple = triple
            self.labels = labels

        def __getitem__(self, idx):
            return [self.triple[idx, :],  self.labels[idx, :]]

        def __len__(self):
            return self.triple.shape[0]


    print('Before tranforming: ', data.shape)
    data = data.astype(np.int64)
    label = label.astype(np.float32)

    total_triple = data; total_label = label

    total_relation = list(set(list(total_triple[:, 1])))
    
    total_triple2 = {}
    for i in range(len(total_triple)):
        re = total_triple[i, 1]
        if re not in total_triple2.keys():
            total_triple2[re] = []
        triple_label = np.concatenate((np.expand_dims(total_triple[i, :], axis=0), np.expand_dims(total_label[i, :], axis=0)), axis = -1)
        total_triple2[re].append(triple_label)

    threshold = len(total_triple)/len(total_relation) * 0.1  #####################0.04 for nell v2   0.4 fb v1??
    print(threshold)
    threshold = 10
    large_relation = []; few_relation = []

    for i in range(len(total_relation)):
        relation_new = total_relation[i]
        relation = len(total_triple2[relation_new])
        if relation > threshold:
            large_relation.append(relation_new)
        else:
            few_relation.append(relation_new)
    
    return total_triple2, large_relation, few_relation




def process_data_test2(data, label, indice):
    # parse the input args
    class pharmKG(Dataset):
        '''
        PyTorch Dataset for pharmKG, don't need to change this
        '''

        def __init__(self, triple, labels):
            self.triple = triple
            self.labels = labels

        def __getitem__(self, idx):
            return [self.triple[idx, :],  self.labels[idx, :]]

        def __len__(self):
            return self.triple.shape[0]


    print('Before tranforming: ', data.shape)
    data = data.astype(np.int64)
    label = label.astype(np.float32)

    total_triple = data; total_label = label

    total_relation = list(set(list(total_triple[:, 1])))
    
    total_triple2 = {}
    for i in range(len(total_triple)):
        re = total_triple[i, 1]
        if re not in total_triple2.keys():
            total_triple2[re] = []
        triple_label = np.concatenate((np.expand_dims(total_triple[i, :], axis=0), np.expand_dims(total_label[i, :], axis=0)), axis = -1)
        total_triple2[re].append(triple_label)

    threshold = len(total_triple)/len(total_relation) * 0.4 #####################0.04 for nell v2
    print(threshold)
   # threshold = 5
    k = 4
    large_relation = []; few_relation = []

    for i in range(len(total_relation)):
        relation_new = total_relation[i]
        relation = len(total_triple2[relation_new])
        if relation > threshold:
            large_relation.append(relation_new)
        if relation > 9 and relation <= threshold:
            few_relation.append(relation_new)
    
    return total_triple2, large_relation, few_relation





def process_data2(data, label, indice):
    # parse the input args
    class pharmKG(Dataset):
        '''
        PyTorch Dataset for pharmKG, don't need to change this
        '''

        def __init__(self, triple, labels):
            self.triple = triple
            self.labels = labels

        def __getitem__(self, idx):
            return [self.triple[idx, :, :],  self.labels[idx, :, :]]

        def __len__(self):
            return self.triple.shape[0]


    print('Before tranforming: ', data.shape)
    data = data.astype(np.int64)
    label = label.astype(np.float32)
   
    positive_triple = np.expand_dims(data[:indice, :], axis = 0)
    negative_triple = data[indice:, :]
    ratio = negative_triple.shape[0]//indice
    negative_triple = np.reshape(negative_triple, (ratio, indice, 3))
    total_triple = np.concatenate([positive_triple, negative_triple], axis=0)
    total_triple = np.swapaxes(total_triple, 0 ,1)


    positive_label = np.expand_dims(label[:indice, :], axis = 0)
    negative_label = label[indice:, :]
    negative_label = np.reshape(negative_label, (ratio, indice, 1))
    total_label = np.concatenate([positive_label, negative_label], axis=0)
    total_label = np.swapaxes(total_label, 0 ,1)
 
    #total_triple = data; total_label = label
    train_set = pharmKG(total_triple, total_label)


    print('After tranforming: ', total_triple.shape)
  #  print(total_triple[0, :, :], total_label[0, :, :])
    
    return train_set





def save_model(model, name, epoch, folder_name):
    print("Saving Model")
    torch.save(model.state_dict(),
               (folder_name + "trained_{}.pth").format(epoch))
    print("Done saving Model")

def save_model2(model, name, epoch, folder_name):
    print("Saving Model")
    torch.save(model,
               (folder_name + "trained_{}.pkl").format(epoch))
    print("Done saving Model")


gat_loss_func = nn.MarginRankingLoss(margin=0.5)


def GAT_Loss(train_indices, valid_invalid_ratio):
    len_pos_triples = train_indices.shape[0] // (int(valid_invalid_ratio) + 1)

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(valid_invalid_ratio), 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=2, dim=1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=2, dim=1)

    y = torch.ones(int(args.valid_invalid_ratio)
                   * len_pos_triples).cuda()
    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss


def render_model_graph(model, Corpus_, train_indices, relation_adj, averaged_entity_vectors):
    graph = make_dot(model(Corpus_.train_adj_matrix, train_indices, relation_adj, averaged_entity_vectors),
                     params=dict(model.named_parameters()))
    graph.render()


def print_grads(model):
    print(model.relation_embed.weight.grad)
    print(model.relation_gat_1.attention_0.a.grad)
    print(model.convKB.fc_layer.weight.grad)
    for name, param in model.named_parameters():
        print(name, param.grad)


def clip_gradients(model, gradient_clip_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, "norm before clipping is -> ", param.grad.norm())
            torch.nn.utils.clip_grad_norm_(param, args.gradient_clip_norm)
            print(name, "norm beafterfore clipping is -> ", param.grad.norm())


def plot_grad_flow(named_parameters, parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in zip(named_parameters, parameters):
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="r")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="g")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="r", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="g", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig('initial.png')
    plt.close()


def plot_grad_flow_low(named_parameters, parameters):
    ave_grads = []
    layers = []
    for n, p in zip(named_parameters, parameters):
        # print(n)
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig('initial.png')
    plt.close()
