import torch

from models4 import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy

from preprocess2 import read_entity_from_id, read_relation_from_id, init_embeddings, build_data
from create_batch_inductive2 import *
from utils import save_model, process_data3, save_model2, process_data, process_data_test
from torch.utils.data import DataLoader, Dataset
import random
import argparse
import os
import sys
import logging
import time
import pickle
import pdb
from torch.nn.utils import weight_norm

# %%
# %%from torchviz import make_dot, make_dot_from_trace


def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-data", "--data",
                      default="./data/FB15k-237/", help="data directory")
    args.add_argument("-dataset", "--dataset",
                      default="FB15k-237", help="data set")
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=3600, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=200, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=5e-6, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-5, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=str,
                      default='False', help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=50, help="Size of embeddings (if pretrained not used)")

    args.add_argument("-re_emb", "--relation_emb", type=int,
                      default=32, help="Size of relation embeddings ")


    args.add_argument("-l", "--lr", type=float, default=1e-3)

    args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/FB15k-237/out/", help="Folder name to save the models.")

    # arguments for convolution network
    args.add_argument("-b_conv", "--batch_size_conv", type=int,
                      default=128, help="Batch size for conv")   ########128
    args.add_argument("-alpha_conv", "--alpha_conv", type=float,
                      default=0.2, help="LeakyRelu alphas for conv layer")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=1,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=500,
                      help="Number of output channels in conv layer")
    args.add_argument("-drop_conv", "--drop_conv", type=float,
                      default=0.0, help="Dropout probability for convolution layer")

    args.add_argument("--dt", type=str,
                      default='now', help="datetime")

    args.add_argument("--model", type=str,
                      default='compile', help="datetime")

    args.add_argument("--prefix", type=str,
                      default='', help="prefix")
    args.add_argument("--train", type=str,
                      default='True', help="True")

    args.add_argument("--test", type=str,
                      default='True', help="test")

    args.add_argument("--explain", type=str,
                      default='True', help="explain")

    args.add_argument("--hop", type=int,
                      default=3, help="hop")

    args.add_argument("--direct", type=str,
                      default="True", help="directed graph or not")
    args.add_argument('--ensemble_size', type=int, default=1,
                        help='Number of models in ensemble')
    args.add_argument('--hidden_size', type=int, default=300,
                        help='Dimensionality of hidden layers in MPN')
    args.add_argument('--bias', action='store_true', default=False,
                        help='Whether to add bias to linear layers')
    args.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    args.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')
    args.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')
    args.add_argument('--undirected', action='store_true', default=False,
                        help='Undirected edges (always sum the two relevant bond vectors)')                     
    args.add_argument('--ffn_hidden_size', type=int, default=None,
                        help='Hidden dim for higher-capacity FFN (defaults to hidden_size)')
    args.add_argument('--ffn_num_layers', type=int, default=2,
                        help='Number of layers in FFN after MPN encoding')
    args.add_argument('--atom_messages', action='store_true', default=False,
                        help='Use messages on atoms instead of messages on bonds')


    args = args.parse_args()
    return args

args = parse_args()
torch.set_num_threads(64)


import os.path
hop = args.hop
datapath = "./data/{}_{}_hop_new_data.pickle".format(args.dataset, hop)
if os.path.isfile(datapath):
    with open(datapath, 'rb') as handle:
        new_data = pickle.load(handle)
    if args.direct == "True":
         datapath1 = "./subgraph/{}_{}_hop_train_subgraph.pickle".format(args.dataset, hop)
    else:
         datapath1 = "./subgraph/{}_{}_hop_undirected_train_subgraph.pickle".format(args.dataset, hop)
    with open(datapath1, 'rb') as handle:
        train_subgraph = pickle.load(handle)

    if args.direct == "True":
         datapath1 = "./subgraph/{}_{}_hop_test_subgraph.pickle".format(args.dataset, hop)
    else:
         datapath1 = "./subgraph/{}_{}_hop_undirected_test_subgraph.pickle".format(args.dataset, hop)
    with open(datapath1, 'rb') as handle:
        test_subgraph = pickle.load(handle)
    print('finished loading subgraph...')

    with open("./data/{}_{}_total_train_data.pickle".format(args.dataset, hop), 'rb') as handle:
        train_data = pickle.load(handle)
    with open("./data/{}_{}_total_train_label.pickle".format(args.dataset, hop), 'rb') as handle:
        train_label = pickle.load(handle)

    if args.direct == "True":
         datapath1 = "./subgraph/{}_{}_hop_validation_subgraph.pickle".format(args.dataset, hop)
    else:
         datapath1 = "./subgraph/{}_{}_hop_undirected_validation_subgraph.pickle".format(args.dataset, hop)
    with open(datapath1, 'rb') as handle:
        val_subgraph = pickle.load(handle)
    print('finished loading subgraph...')

    with open("./data/{}_{}_total_val_data.pickle".format(args.dataset, hop), 'rb') as handle:
        val_data = pickle.load(handle)
    with open("./data/{}_{}_total_val_label.pickle".format(args.dataset, hop), 'rb') as handle:
        val_label = pickle.load(handle)
 

CUDA = torch.cuda.is_available()
np.random.seed(1111)
torch.manual_seed(1234)

def train_compile(args):

    print("Defining model")
    train_triple = new_data['new_train_data']
    relation = set(list(train_triple[:, 1]))
    print('total_number_of_train_relation: ', len(relation))   
    val_relation = set(list(new_data['new_val_data'][:,1]))
    relation = relation.union(val_relation)
    test_relation = set(list(new_data['new_test_data'][:,1]))
    relation = relation.union(test_relation)
    print('total_number_of_relation: ', len(relation))
   
    entity = set(list(train_data[:, 0])).union(set(list(train_data[:, 2])))
    print('total entity train:', len(entity))
 
    train_entity = set(list(new_data['new_train_entity']))
    test_entity = set(list(new_data['new_test_entity']))   

    node_emb = 2*(hop+1)
    output_dim=1; latent_dim=[128, 64, 32, 16]; relation_emb =  args.relation_emb  # 32 #50

    model_conv = CoMPILE(args, np.max(np.array(list(relation)))+1, relation_emb, latent_dim, output_dim, node_emb)

    save_iteration = int(train_data.shape[0]//args.batch_size_conv//3)
    train_set = process_data(train_data, train_label, train_triple.shape[0])
    train_iterator = DataLoader(train_set, batch_size=args.batch_size_conv, num_workers=8, shuffle=True)

    if CUDA:
        model_conv.cuda()

    optimizer = torch.optim.Adam(
        model_conv.parameters(), lr=args.lr, weight_decay=args.weight_decay_conv)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, last_epoch=-1)

    margin_loss = torch.nn.SoftMarginLoss()

    epoch_losses = []   # losses of all epochs
    print("Number of epochs {}".format(args.epochs_conv))
    best_loss = 100000; best_auc = 0
    for epoch in range(args.epochs_conv):
        print("\nepoch-> ", epoch)

        model_conv.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []


        for iters, batch in enumerate(train_iterator):
            start_time_iter = time.time()
            train_indices, train_values = batch
          
            if CUDA:
                train_indices = Variable(torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            train_indices = train_indices.view(-1, 3)

            preds = model_conv(train_indices, train_subgraph)

            optimizer.zero_grad()
            loss = margin_loss(preds.view(-1), train_values.view(-1))

            loss.backward()

            optimizer.step()

            epoch_loss.append(loss.data.item())
            end_time_iter = time.time()

            if (iters+1)%save_iteration ==0:
                 directed = "_directed_" if args.direct=="True" else "_undirected_"
                 save_model2(model_conv, args.data, epoch + 1000*int((iters+1)//save_iteration),
                   args.output_folder + "ourconv2/"+directed)
                 with torch.no_grad():
                     val_loss = valid(model_conv, margin_loss, 128)
                 if best_auc < val_loss:
                     best_auc = val_loss
                     print('find best model, saving...')
                     torch.save(model_conv, args.output_folder + "ourconv2/"+ args.dataset + directed + "best.pkl")

        scheduler.step()
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))
        directed = "_directed_" if args.direct=="True" else "_undirected_"
        save_model2(model_conv, args.data, epoch,
                   args.output_folder + "ourconv2/"+directed)
        with torch.no_grad():
              val_loss = valid(model_conv, margin_loss, 128)
        if best_auc < val_loss:
              best_auc = val_loss
              print('find best model, saving...')
              torch.save(model_conv, args.output_folder + "ourconv2/" + args.dataset + directed +  "best.pkl")




from sklearn import metrics
def valid(model, loss, batch_size = 128):
    losses = 0
    total_iter = int(len(val_data)//batch_size) + 1
    if len(val_data)%batch_size ==0:
       total_iter -=1
    all_preds = []
    all_labels = []
    for i in range(total_iter):
        if i == total_iter - 1:
            train_indices = val_data[i*batch_size:, :]
            train_values = val_label[i*batch_size:, :]
        else:
            train_indices = val_data[i*batch_size:(i+1)*batch_size, :]
            train_values = val_label[i*batch_size:(i+1)*batch_size, :]
          
        if CUDA:
            train_indices = Variable(torch.LongTensor(train_indices)).cuda()
            train_values = Variable(torch.FloatTensor(train_values)).cuda()
        else:
            train_indices = Variable(torch.LongTensor(train_indices))
            train_values = Variable(torch.FloatTensor(train_values))

     #   print(train_indices.shape)
        preds = model(train_indices, val_subgraph)
        all_preds += preds.squeeze(1).detach().cpu().tolist()
        all_labels += train_values.squeeze(1).tolist()
        losses += loss(preds.view(-1), train_values.view(-1))* len(train_indices)
    auc = metrics.roc_auc_score(all_labels, all_preds)
   # return losses//len(val_data)
    return auc



def evaluate(args):
    directed = "_directed_" if args.direct=="True" else "_undirected_"
    if args.epochs_conv < 0:
        model_conv = torch.load(args.output_folder + "ourconv2/" + args.dataset + directed +  "best.pkl")
    else:
        model_conv = torch.load('{0}ourconv2/{1}trained_{2}.pkl'.format(args.output_folder, directed, args.epochs_conv - 1))
    model_conv.cuda()
    #model_conv = torch.nn.DataParallel(model_conv)
    model_conv.eval()


    with open("./data/{}_{}_hop_total_test_head2.pickle".format(args.dataset, hop), 'rb') as handle:
        total_test_head = pickle.load(handle)
    with open("./data/{}_{}_hop_total_test_tail2.pickle".format(args.dataset, hop), 'rb') as handle:
        total_test_tail = pickle.load(handle)

    train_triple = new_data['new_train_data']
    relation = set(list(train_triple[:, 1]))
    
    new_total_test_head = []; true_triplets = []
    new_total_test_tail = []; neg_triplets = []
    for i in range(len(total_test_head)):
       if total_test_head[i][0, 1] in relation and total_test_head[i].shape[0]>=50 and total_test_tail[i].shape[0]>=50:
           new_total_test_head.append(total_test_head[i])
     #  if total_test_tail[i][0, 1] in relation and total_test_tail[i].shape[0]>50:

           new_total_test_tail.append(total_test_tail[i])
           true_triplets.append(total_test_tail[i][0, :])
           xx = np.random.uniform()
           if xx < 0.5: 
                neg = total_test_tail[i][1:, :]
                random_entity = random.choice([i for i in range(49)])
                neg_triplets.append(neg[random_entity, :])   
           else:
                neg = total_test_head[i][1:, :]
                random_entity = random.choice([i for i in range(49)])
                neg_triplets.append(neg[random_entity, :])   
                
    true_label = np.expand_dims(np.array([1 for i in range(len(true_triplets))]), 1)
    neg_label = np.expand_dims(np.array([-1 for i in range(len(neg_triplets))]), 1)
    true_triplets = np.array(true_triplets); neg_triplets = np.array(neg_triplets)


    with torch.no_grad():
        get_our_validation_pred_random(args, model_conv, new_total_test_head, new_total_test_tail, test_subgraph)
  #  auc, auc_pr = get_auc(model_conv, true_triplets, true_label, neg_triplets, neg_label)  
  #  print('auc: ', auc, 'auc_pr: ', auc_pr)

    mean_auc_pr = 0.
    for kk in range(10):
        true_triplets = []; neg_triplets = []
        for i in range(len(new_total_test_head)):
            true_triplets.append(new_total_test_head[i][0, :])
            xx = np.random.uniform()
            if xx < 0.5: 
                neg = new_total_test_head[i][1:, :]
                random_entity = random.choice([i for i in range(49)])
                neg_triplets.append(neg[random_entity, :])   
            else:
                neg = new_total_test_tail[i][1:, :]
                random_entity = random.choice([i for i in range(49)])
                neg_triplets.append(neg[random_entity, :])   
                
        true_label = np.expand_dims(np.array([1 for i in range(len(true_triplets))]), 1)
        neg_label = np.expand_dims(np.array([-1 for i in range(len(neg_triplets))]), 1)
        true_triplets = np.array(true_triplets); neg_triplets = np.array(neg_triplets)

   
        auc, auc_pr = get_auc(model_conv, true_triplets, true_label, neg_triplets, neg_label)  

        mean_auc_pr += auc_pr
    print('mean_auc_pr: ', mean_auc_pr/10.0)





def get_auc(model, true_triplets, true_label, neg_triplets, neg_label, batch_size = 128):
   
    total_iter = int(len(true_triplets)//batch_size) + 1
    if len(true_triplets)%batch_size ==0:
       total_iter -=1
    all_preds = []
    all_labels = []
    for i in range(total_iter):
        if i == total_iter - 1:
            true_indices = true_triplets[i*batch_size:, :]
            true_values = true_label[i*batch_size:, :]
            neg_indices = neg_triplets[i*batch_size:, :]
            neg_values = neg_label[i*batch_size:, :]
        else:
            true_indices = true_triplets[i*batch_size:(i+1)*batch_size, :]
            true_values = true_label[i*batch_size:(i+1)*batch_size, :]
            neg_indices = neg_triplets[i*batch_size:(i+1)*batch_size, :]
            neg_values = neg_label[i*batch_size:(i+1)*batch_size, :]
          
        if CUDA:
            true_indices = Variable(torch.LongTensor(true_indices)).cuda()
            neg_indices = Variable(torch.LongTensor(neg_indices)).cuda()
          #  train_values = Variable(torch.FloatTensor(train_values)).cuda()
        else:
            true_indices = Variable(torch.LongTensor(true_indices))
            neg_indices = Variable(torch.LongTensor(neg_indices))
           # train_values = Variable(torch.FloatTensor(train_values))

        preds = model(true_indices, test_subgraph)
        all_preds += preds.squeeze(1).detach().cpu().tolist()
        all_labels += true_values.tolist()

        preds = model(neg_indices, test_subgraph)
        all_preds += preds.squeeze(1).detach().cpu().tolist()
        all_labels += neg_values.tolist()

    auc = metrics.roc_auc_score(all_labels, all_preds)
    auc_pr = metrics.average_precision_score(all_labels, all_preds)
    return auc, auc_pr

     
if args.train=='True':
    train_compile(args)
 

if args.test=='True':
    evaluate(args)



