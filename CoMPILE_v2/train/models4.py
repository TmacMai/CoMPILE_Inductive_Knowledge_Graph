import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from layers2 import SpGraphAttentionLayer, ConvKB, OurSpGraphAttentionLayer

CUDA = torch.cuda.is_available()  # checking cuda availability

import numpy as np



class MySpMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sp_mat, dense_mat):
        ctx.save_for_backward(sp_mat, dense_mat)

        return torch.mm(sp_mat, dense_mat)

    @staticmethod
    def backward(ctx, grad_output):        
        sp_mat, dense_mat = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        assert not ctx.needs_input_grad[0]
        if ctx.needs_input_grad[1]:
            grad_matrix2 = Variable(torch.mm(sp_mat.data.t(), grad_output.data))
        
        return grad_matrix1, grad_matrix2

def gnn_spmm(sp_mat, dense_mat):
    return MySpMM.apply(sp_mat, dense_mat)






def get_activation_function(activation):
    """
    Gets an activation function module given the name of the activation.

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    else:
        raise ValueError('Activation "{}" not supported.'.format(activation))

import math







     






class CoMPILE(nn.Module):               
    def __init__(self, args, num_relation, relation_emb, latent_dim, output_dim,
                 node_emb):

        super().__init__()  
        self.params = args
        self.latent_dim = latent_dim
        self.output_dim = 1
        self.node_emb = node_emb
        self.relation_emb = relation_emb
        self.edge_emb = self.node_emb * 2 + self.relation_emb 
        self.hidden_size = relation_emb
        self.num_relation = num_relation

        self.final_relation_embeddings = nn.Parameter(torch.randn(num_relation, relation_emb))
      #  self.relation_to_edge = nn.Linear(relation_emb, self.hidden_size)

        self.linear1 = nn.Linear(self.hidden_size, 1)

        self.node_fdim = self.node_emb
        self.edge_fdim = self.edge_emb
        
        self.bias = False
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = False
        self.node_messages = False
        self.args = args
        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)  #################### ELU relu tanh

        # Cached zeros
     #   self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size),  requires_grad=False)
        # Input
        input_dim = self.node_fdim
        self.W_i_node = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        input_dim = self.edge_fdim
        self.W_i_edge = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        
        w_h_input_size_atom = self.hidden_size + self.edge_fdim
      #  self.W_h_atom = nn.Linear(w_h_input_size_atom, self.hidden_size, bias=self.bias)

        self.input_attention1 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=self.bias)
        self.input_attention2 = nn.Linear(self.hidden_size, 1, bias=self.bias)
        
        w_h_input_size_bond = self.hidden_size
        for depth in range(self.depth-1):
            self._modules['W_h_bond_{}'.format(depth)] = nn.Linear(w_h_input_size_bond, self.hidden_size, bias=self.bias)
          #  self._modules['W_h_bond_{}'.format(depth)] = nn.Linear(w_h_input_size_bond * 3 + self.params.rel_emb_dim, self.hidden_size, bias=self.bias)
            self._modules['Attention1_{}'.format(depth)] = nn.Linear(self.hidden_size + self.relation_emb, self.hidden_size, bias=self.bias)
            self._modules['Attention2_{}'.format(depth)] = nn.Linear(self.hidden_size, 1, bias=self.bias)
        
        self.W_o = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        self.gru = BatchGRU(self.hidden_size)
        
        self.communicate_mlp = nn.Linear(self.hidden_size*3, self.hidden_size, bias=self.bias)
        
        for depth in range(self.depth-1):
            self._modules['W_h_atom_{}'.format(depth)] = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)

    def forward(self, batch_inputs, subgraph):

        batch_target_relation = batch_inputs[:, 1].view(-1,)
      
        target_relation = self.final_relation_embeddings[batch_target_relation, :]

        graph_embed, source_embed, target_embed = self.batch_subgraph(batch_inputs, subgraph, target_relation) 
        
        conv_input = torch.tanh(source_embed + target_relation -target_embed)   
        out_conv = (self.linear1(conv_input))
   
        return out_conv


    def batch_subgraph(self, batch_inputs,  subgraph, target_relation):

        batch_row = batch_inputs[:, 0].view(-1,).cpu().data.numpy()
        batch_col = batch_inputs[:, 2].view(-1,).cpu().data.numpy()
    
        graph_sizes = []; node_feat = []
        list_num_nodes = np.zeros((batch_inputs.shape[0], ), dtype=np.int32)
        list_num_edges = np.zeros((batch_inputs.shape[0], ), dtype=np.int32)
        node_count = 0 ; edge_count = 0; edge_feat = []
        total_edge = []; source_node = []; target_node = [] 
        total_target_relation = []; total_edge2 = []
        total_source = []; total_target = []
        for i in range(batch_inputs.shape[0]):

            graph = subgraph[batch_row[i]][batch_col[i]]['edge'].astype(np.int64)
            if len(graph) == 0:
                graph = np.array([batch_inputs[i, 0], batch_inputs[i, 2], batch_inputs[i, 1]]).astype(np.int64)
                if len(graph.shape)==1:
                     graph = np.expand_dims(graph, axis=0)
            node = subgraph[batch_row[i]][batch_col[i]]['node'].astype(np.int64)
          
            node_embedding = node[:, 1:]
            node_feat.append(torch.FloatTensor(node_embedding.astype(np.float32)))
            
            graph_sizes.append(node.shape[0])
            list_num_nodes[i] = node.shape[0]
            list_num_edges[i] = graph.shape[0]
 

            nodes = list(node[:, 0])
            source = list(graph[:, 0])   
            target = list(graph[:, 1])
            relation = torch.LongTensor(graph[:, 2])

            relation_now = self.final_relation_embeddings[relation, :]
            # total_relation.append(relation_now) 
            target_relation_now = target_relation[i, :].unsqueeze(0).repeat(list_num_edges[i], 1)
            total_target_relation.append(target_relation_now)
          
            mapping = dict(zip(nodes, [i for i in range(node_count, node_count+list_num_nodes[i])]))

            source_map_now = np.array([mapping[v] for v in source]) - node_count
            target_map_now = np.array([mapping[v] for v in target]) - node_count
            source_embed = node_embedding[source_map_now, :].astype(np.float32)
            target_embed = node_embedding[target_map_now, :].astype(np.float32)

            if CUDA:
  
                source_embed = torch.FloatTensor(source_embed).cuda()
                target_embed = torch.FloatTensor(target_embed).cuda()

            edge_embed = torch.cat([source_embed, relation_now, target_embed], dim = 1)
            edge_feat.append(edge_embed)
            
            source_node.append(mapping[batch_row[i]])
            target_node.append(mapping[batch_col[i]])
            
            target_now = torch.LongTensor(np.expand_dims(mapping[batch_col[i]], 0)).unsqueeze(0).repeat(list_num_edges[i], 1)
            source_now = torch.LongTensor(np.expand_dims(mapping[batch_row[i]], 0)).unsqueeze(0).repeat(list_num_edges[i], 1)

            total_source.append(source_now); total_target.append(target_now)
            
            node_count += list_num_nodes[i]

            source_map = torch.LongTensor(np.array([mapping[v] for v in source])).unsqueeze(0)
            target_map = torch.LongTensor(np.array([mapping[v] for v in target])).unsqueeze(0)
          
            edge_pair = torch.cat([target_map, torch.LongTensor(np.array(range(edge_count, edge_count+list_num_edges[i]))).unsqueeze(0)], dim=0)
            
            edge_pair2 = torch.cat([source_map, torch.LongTensor(np.array(range(edge_count, edge_count+list_num_edges[i]))).unsqueeze(0)], dim=0)

            edge_count += list_num_edges[i]
            total_edge.append(edge_pair)       
            total_edge2.append(edge_pair2)      
  
        source_node = np.array(source_node); target_node = np.array(target_node)   
   
        total_edge = torch.cat(total_edge, dim = 1)
        total_edge2 = torch.cat(total_edge2, dim = 1)
        total_target_relation = torch.cat(total_target_relation, dim=0)
        total_source = torch.cat(total_source, dim=0).squeeze(1)
        total_target = torch.cat(total_target, dim=0).squeeze(1)

        total_num_nodes = np.sum(list_num_nodes)
        total_num_edges = np.sum(list_num_edges)

        e2n_value = torch.FloatTensor(torch.ones(total_edge.shape[1]))
        e2n_sp = torch.sparse.FloatTensor(total_edge, e2n_value, torch.Size([total_num_nodes, total_num_edges]))
        e2n_sp2 = torch.sparse.FloatTensor(total_edge2, e2n_value, torch.Size([total_num_nodes, total_num_edges]))
       # e2n_sp = F.normalize(e2n_sp, dim=2, p=1)
        
        node_feat = torch.cat(node_feat, dim=0)
        if CUDA:
            e2n_sp = e2n_sp.cuda()
            e2n_sp2 = e2n_sp2.cuda()
            node_feat = node_feat.cuda() 


        edge_feat = torch.cat(edge_feat, dim=0)
        graph_embed, source_embed, target_embed = self.gnn(node_feat, edge_feat, e2n_sp, e2n_sp2, graph_sizes, total_target_relation, total_source, total_target, source_node, target_node, list(list_num_edges))

        return graph_embed, source_embed, target_embed

    def gnn(self, node_feat, edge_feat, e2n_sp, e2n_sp2, graph_sizes, target_relation, total_source, total_target, source_node, target_node, edge_sizes = None, node_degs=None):
        ''' if exists edge feature, concatenate to node feature vector '''
        input_node_emb = self.W_i_node(node_feat)  # num_atoms x hidden_size
        input_node_emb = self.act_func(input_node_emb)
        message_node = input_node_emb.clone()
        relation_embed = (edge_feat[:, self.node_emb: self.node_emb + self.relation_emb])
               
        input_edge_emb = self.W_i_edge(edge_feat)  # num_bonds x hidden_size
        message_edge = self.act_func(input_edge_emb)
        input_edge_emb = self.act_func(input_edge_emb)

        graph_source_embed = message_node[total_source, :]
        graph_target_embed = message_node[total_target, :]
        graph_edge_embed = graph_source_embed + target_relation - graph_target_embed
        edge_target_message = gnn_spmm(e2n_sp.t(), message_node)
        edge_source_message = gnn_spmm(e2n_sp2.t(), message_node)
        edge_message = edge_source_message + relation_embed - edge_target_message
      #  print(total_source.shape, total_target.shape, graph_source_embed.shape)
        attention = torch.cat([graph_edge_embed, edge_message], dim=1)
        attention = torch.relu(self.input_attention1(attention))
        attention = torch.sigmoid(self.input_attention2(attention))
        
        # Message passing
        for depth in range(self.depth - 1):
            message_edge = (message_edge * attention)
            agg_message = gnn_spmm(e2n_sp, message_edge)
            message_node = message_node + agg_message
            message_node = self.act_func(self._modules['W_h_atom_{}'.format(depth)](message_node))
            
            edge_target_message = gnn_spmm(e2n_sp.t(), message_node)
            edge_source_message = gnn_spmm(e2n_sp2.t(), message_node)
           # message_edge = torch.cat([message_edge, edge_source_message, relation_embed, edge_target_message], dim=-1)
            message_edge = torch.relu(message_edge + torch.tanh( edge_source_message + relation_embed - edge_target_message))
            message_edge = self._modules['W_h_bond_{}'.format(depth)](message_edge)
            message_edge = self.act_func(input_edge_emb + message_edge)
            message_edge = self.dropout_layer(message_edge)  # num_bonds x hidden

            graph_source_embed = message_node[total_source, :]
            graph_target_embed = message_node[total_target, :]
            graph_edge_embed = graph_source_embed + target_relation - graph_target_embed
            edge_message = edge_source_message + relation_embed - edge_target_message
            attention = torch.cat([graph_edge_embed, edge_message], dim=1)            
            attention = torch.relu(self._modules['Attention1_{}'.format(depth)](attention))
            attention = torch.sigmoid(self._modules['Attention2_{}'.format(depth)](attention))

        message_edge = (message_edge * attention)
        agg_message = gnn_spmm(e2n_sp, message_edge)
        agg_message2 = self.communicate_mlp(torch.cat([agg_message, message_node, input_node_emb], 1))
# =============================================================================
#         
# =============================================================================
        a_message = torch.relu(self.gru(agg_message2, graph_sizes))        
        atom_hiddens = self.act_func(self.W_o(a_message))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
        
        # Readout
        mol_vecs = []
        a_start = 0        
        for a_size in graph_sizes:
            if a_size == 0:
                assert 0
            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
            mol_vecs.append(cur_hiddens.mean(0))
            a_start += a_size
        mol_vecs = torch.stack(mol_vecs, dim=0)
        
        source_embed = atom_hiddens[source_node, :]
        target_embed = atom_hiddens[target_node, :]

        return mol_vecs, source_embed, target_embed      

   






class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru  = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, 
                           bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size), 
                                1.0 / math.sqrt(self.hidden_size))


    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        MAX_atom_len = max(a_scope)
        # padding
        message_lst = []
        hidden_lst = []
        a_start = 0
        for i in a_scope:
            if i == 0:
                assert 0
            cur_message = message.narrow(0, a_start, i)
            cur_hidden = hidden.narrow(0, a_start, i)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))
            a_start += i
            cur_message = torch.nn.ZeroPad2d((0,0,0,MAX_atom_len-cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))
            
        message_lst = torch.cat(message_lst, 0)
        hidden_lst  = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2,1,1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)
        
        # unpadding
        cur_message_unpadding = []
        kk = 0
        for a_size in a_scope:
            cur_message_unpadding.append(cur_message[kk, :a_size].view(-1, 2*self.hidden_size))
            kk += 1
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)
        
        message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1), 
                             cur_message_unpadding], 0)
        return message











class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru  = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, 
                           bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size), 
                                1.0 / math.sqrt(self.hidden_size))


    def forward(self, node, a_scope):
        hidden = node
      #  print(hidden.shape)
        message = F.relu(node + self.bias)
        MAX_atom_len = max(a_scope)
        # padding
        message_lst = []
        hidden_lst = []
        a_start = 0
        for i in a_scope:
            if i == 0:
                assert 0
            cur_message = message.narrow(0, a_start, i)
            cur_hidden = hidden.narrow(0, a_start, i)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))
            a_start += i
            cur_message = torch.nn.ZeroPad2d((0,0,0,MAX_atom_len-cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))
            
        message_lst = torch.cat(message_lst, 0)
        hidden_lst  = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2,1,1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)
        
        # unpadding
        cur_message_unpadding = []
        kk = 0
        for a_size in a_scope:
            cur_message_unpadding.append(cur_message[kk, :a_size].view(-1, 2*self.hidden_size))
            kk += 1
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)
        
     #   message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1), 
     #                        cur_message_unpadding], 0)
     #   print(cur_message_unpadding.shape)
        return cur_message_unpadding











class OurSpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention

        """
        super(OurSpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [OurSpGraphAttentionLayer(num_nodes, nfeat,
                                                 nhid,
                                                 relation_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = OurSpGraphAttentionLayer(num_nodes, nhid * nheads,
                                             nfeat, nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False
                                             )

    def forward(self, entity_embeddings, relation_embeddings, entity_list, edge_type, target_relation):

       # x = entity_embeddings

        relation_embed = relation_embeddings[edge_type]
        target_relation_embed = relation_embeddings[target_relation].unsqueeze(0)
        x = torch.cat([att(entity_embeddings, entity_list, relation_embed, target_relation_embed)
                       for att in self.attentions], dim=1)
        x = self.dropout_layer(x)

        out_relation_1 = relation_embeddings.mm(self.W)

        relation_embed = out_relation_1[edge_type]
        target_relation_embed = out_relation_1[target_relation].unsqueeze(0)
        x = F.elu(self.out_att(x, entity_list, relation_embed, target_relation_embed))
        return x






class OurSpGAT_Simple(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention

        """
        super(OurSpGAT_Simple, self).__init__()
        self.dropout = dropout
        nheads = 1
        nhid = nfeat
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [OurSpGraphAttentionLayer(num_nodes, nfeat,
                                                 nhid,
                                                 relation_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        '''
        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = OurSpGraphAttentionLayer(num_nodes, nhid * nheads,
                                             nfeat, nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False
                                             )
         '''

    def forward(self, entity_embeddings, relation_embeddings, entity_list, edge_type, target_relation):

       # x = entity_embeddings

        relation_embed = relation_embeddings[edge_type]
        target_relation_embed = relation_embeddings[target_relation].unsqueeze(0)
        x = torch.cat([att(entity_embeddings, entity_list, relation_embed, target_relation_embed)
                       for att in self.attentions], dim=1)
      #  x = self.dropout_layer(x)

      #  out_relation_1 = relation_embeddings.mm(self.W)

       # relation_embed = out_relation_1[edge_type]
       # target_relation_embed = out_relation_1[target_relation].unsqueeze(0)
       # x = F.elu(self.out_att(x, entity_list, relation_embed, target_relation_embed))
        return x



class RGCNModel(nn.Module):
    def __init__(self, args):

        super().__init__()
        self.params = args
        self.gnn = RGCN(self.params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
       # self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)
        self.rel_emb = nn.Parameter(torch.randn(self.params.num_rels, self.params.rel_emb_dim))
        if self.params.add_ht_emb:
            self.fc_layer = nn.Linear(3 * self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)



    def forward(self, batch_inputs, subgraph):

        batch_target_relation = batch_inputs[:, 1].view(-1,)
      
        target_relation = self.rel_emb[batch_target_relation, :]

        graph_embed, source_embed, target_embed = self.batch_subgraph(batch_inputs, subgraph, target_relation) 
      

        if self.params.add_ht_emb:
            g_rep = torch.cat([graph_embed.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               source_embed.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               target_embed.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               target_relation], dim=1)
        else:
            g_rep = torch.cat([graph_embed.view(-1, self.params.num_gcn_layers * self.params.emb_dim), target_relation], dim=1)

        output = (self.fc_layer(g_rep))
        return output


    def batch_subgraph(self, batch_inputs,  subgraph, target_relation):
        batch_row = batch_inputs[:, 0].view(-1,).cpu().data.numpy()
        batch_col = batch_inputs[:, 2].view(-1,).cpu().data.numpy()
        graph_sizes = []; node_feat = []

        list_num_nodes = np.zeros((batch_inputs.shape[0], ), dtype=np.int32)
        list_num_edges = np.zeros((batch_inputs.shape[0], ), dtype=np.int32)
        node_count = 0; edge_count = 0; e2n_edge = []
        total_relation_embed = []; total_edge = []; source_node = []; target_node = []; total_target_relation = []; total_relation = []
      #  total_source = []; total_target = [] 
        for i in range(batch_inputs.shape[0]):
             if len(subgraph[batch_row[i]][batch_col[i]]['edge']):
                  graph = subgraph[batch_row[i]][batch_col[i]]['edge'].astype(np.int64)
             else:
                  graph = np.array([batch_row[i], batch_col[i], batch_inputs[i, 1]]).astype(np.int64)
                  graph = np.expand_dims(graph, axis=0)
             node = subgraph[batch_row[i]][batch_col[i]]['node'].astype(np.int64)
          
             node_embedding = node[:, 1:]
             graph_sizes.append(node.shape[0])
             node_feat.append(torch.FloatTensor(node_embedding.astype(np.float32)))

             list_num_nodes[i] = node.shape[0]
             list_num_edges[i] = graph.shape[0]
 
             nodes = list(node[:, 0])
             source = list(graph[:, 0])   
             target = list(graph[:, 1])
             relation = torch.LongTensor(graph[:, 2])
             relation_now = self.rel_emb[relation, :]
             total_relation_embed.append(relation_now) 
             total_relation.append(relation)
             current_target_relation = target_relation[i, :].unsqueeze(0).repeat(graph.shape[0], 1)
             total_target_relation.append(current_target_relation)

             mapping = dict(zip(nodes, [i for i in range(node_count, node_count+list_num_nodes[i])]))

             source_map = torch.LongTensor(np.array([mapping[v] for v in source])).unsqueeze(0)
             target_map = torch.LongTensor(np.array([mapping[v] for v in target])).unsqueeze(0)

             source_node.append(mapping[batch_row[i]])
             target_node.append(mapping[batch_col[i]])
             node_count += list_num_nodes[i]
        
             edge_pair = torch.cat([source_map, target_map], dim=0)
             total_edge.append(edge_pair)      

             edge = torch.cat([target_map, torch.LongTensor(np.array(range(edge_count, edge_count+list_num_edges[i]))).unsqueeze(0)], dim=0)
             e2n_edge.append(edge)
             edge_count += list_num_edges[i] 
   
        total_target_relation = torch.cat(total_target_relation, dim = 0)
        total_relation_embed = torch.cat(total_relation_embed, dim=0)
        total_relation = torch.cat(total_relation, dim=0)

        source_node = np.array(source_node); target_node = np.array(target_node)   
        total_edge = torch.cat(total_edge, dim = 1)

        e2n_edge = torch.cat(e2n_edge, dim = 1)      
   
        total_num_nodes = np.sum(list_num_nodes)
        total_num_edges = np.sum(list_num_edges)

        e2n_value = torch.FloatTensor(torch.ones(total_num_edges))
        e2n_sp = torch.sparse.FloatTensor(e2n_edge, e2n_value, torch.Size([total_num_nodes, total_num_edges]))

        node_feat = Variable(torch.cat(node_feat, dim=0))

        if CUDA:
            e2n_sp = e2n_sp.cuda()
            node_feat = node_feat.cuda()  
            total_edge = total_edge.cuda()
            total_relation = total_relation.cuda()
        graph_embed, source_embed, target_embed = self.gnn(node_feat, e2n_sp, graph_sizes, total_target_relation, source_node, target_node, total_edge, total_relation_embed, total_relation)

        return graph_embed, source_embed, target_embed



class RGCN(nn.Module):
    def __init__(self, params):
        super(RGCN, self).__init__()

        self.inp_dim = params.inp_dim
        self.emb_dim = params.emb_dim
        self.attn_rel_emb_dim = params.attn_rel_emb_dim
        self.num_rels = params.num_rels
        self.aug_num_rels = params.aug_num_rels
        self.num_bases = params.num_bases
        self.num_hidden_layers = params.num_gcn_layers
        self.dropout = params.dropout
        self.edge_dropout = params.edge_dropout
        # self.aggregator_type = params.gnn_agg_type
        self.has_attn = params.has_attn

        # initialize aggregators for input and hidden layers
        if params.gnn_agg_type == "sum":
            self.aggregator = SumAggregator(self.emb_dim)
        elif params.gnn_agg_type == "mlp":
            self.aggregator = MLPAggregator(self.emb_dim)
        elif params.gnn_agg_type == "gru":
            self.aggregator = GRUAggregator(self.emb_dim)

        # initialize basis weights for input and hidden layers
        # self.input_basis_weights = nn.Parameter(torch.Tensor(self.num_bases, self.inp_dim, self.emb_dim))
        # self.basis_weights = nn.Parameter(torch.Tensor(self.num_bases, self.emb_dim, self.emb_dim))

        # create rgcn layers
        self.build_model()


    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers - 1):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)

    def build_input_layer(self):
        return RGCNLayer2(self.inp_dim,
                         self.emb_dim,
                         # self.input_basis_weights,
                         self.aggregator,
                         self.attn_rel_emb_dim,
                         self.aug_num_rels,
                         self.num_bases,
                         activation=F.relu,
                         dropout=self.dropout,
                         edge_dropout=self.edge_dropout,
                         is_input_layer=True,
                         has_attn=self.has_attn)

    def build_hidden_layer(self, idx):
        return RGCNLayer2(self.emb_dim,
                         self.emb_dim,
                         # self.basis_weights,
                         self.aggregator,
                         self.attn_rel_emb_dim,
                         self.aug_num_rels,
                         self.num_bases,
                         activation=F.relu,
                         dropout=self.dropout,
                         edge_dropout=self.edge_dropout,
                         has_attn=self.has_attn)

    def forward(self, node_feat, e2n_sp, graph_sizes, total_target_relation, source_node, target_node, total_edge, total_relation_embed, total_relation):
        total_node_feat = []
        for layer in self.layers:
            node_feat = layer(node_feat, e2n_sp, total_target_relation, total_edge, total_relation_embed, total_relation)
            total_node_feat.append(node_feat)
        total_node_feat = torch.cat(total_node_feat, dim=1)
        source_embed = total_node_feat[source_node, :]
        target_embed = total_node_feat[target_node, :]
      #  print(source_embed.shape, target_embed.shape)
      #  print(len(graph_sizes))
        graph_embed = []
        a_start = 0
        for a_size in graph_sizes:
            if a_size == 0:
                assert 0
            cur_hiddens = total_node_feat[a_start: a_start+a_size, :]
            graph_embed.append(cur_hiddens.mean(0))
            a_start += a_size
        graph_embed = torch.stack(graph_embed, dim=0)
        return graph_embed, source_embed, target_embed


class RGCNLayer(nn.Module):
    def __init__(self, inp_dim, out_dim, aggregator, attn_rel_emb_dim, num_rels, num_bases=-1, bias=None,
                 activation=None, dropout=0.0, edge_dropout=0.0, is_input_layer=False, has_attn=True):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
            nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))

        self.aggregator = aggregator

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if edge_dropout:
            self.edge_dropout = nn.Dropout(edge_dropout)
        else:
            self.edge_dropout = Identity()

        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.attn_rel_emb_dim = attn_rel_emb_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        self.has_attn = has_attn

        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # add basis weights
        # self.weight = basis_weights
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.inp_dim, self.out_dim))
        self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        if self.has_attn:
            self.A = nn.Linear(2 * self.inp_dim + 2 * self.attn_rel_emb_dim, inp_dim)
            self.B = nn.Linear(inp_dim, 1)

        self.self_loop_weight = nn.Parameter(torch.Tensor(self.inp_dim, self.out_dim))

        nn.init.xavier_uniform_(self.self_loop_weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))

    def propagate(self, node_feat, e2n_sp, total_target_relation, total_edge, total_relation_embed, total_relation):
        # generate all weights from bases
        weight = self.weight.view(self.num_bases, self.inp_dim * self.out_dim)
        weight = torch.matmul(self.w_comp, weight).view(self.num_rels, self.inp_dim, self.out_dim)

        edge_dropout = self.edge_dropout(torch.ones(total_edge.shape[1], 1)).cuda()

        total_source = total_edge[0, :]
        total_source_embed = node_feat.index_select(0, total_source)
        total_target = total_edge[1, :]
        total_target_embed = node_feat.index_select(0, total_target)
      
 
        w = weight.index_select(0, total_relation)

     #   print('total_target_embed: ', total_target_embed.shape, 'w ',  w.shape) 
        msg = edge_dropout * torch.bmm(total_source_embed.unsqueeze(1), w).squeeze(1)
        curr_emb = torch.mm(total_target_embed, self.self_loop_weight)  # (B, F)


        if self.has_attn:
                e = torch.cat([total_source_embed, total_target_embed, total_relation_embed, total_target_relation], dim=1)
                a = torch.sigmoid(self.B(F.relu(self.A(e))))
        else:
                a = torch.ones((total_edge.shape[1], 1)).cuda()
       
        attn_msg = msg * a

        curr_emb = curr_emb + attn_msg
        node_feat = agg_message = gnn_spmm(e2n_sp, curr_emb)
     #   print('node feat ', node_feat.shape)
        return node_feat

    def forward(self, node_feat, e2n_sp, total_target_relation, total_edge, total_relation_embed, total_relation):

        node_repr = self.propagate(node_feat, e2n_sp, total_target_relation, total_edge, total_relation_embed, total_relation)

        # apply bias and activation
        node_repr = node_repr
        if self.bias:
            node_repr = node_repr + self.bias
        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout:
            node_repr = self.dropout(node_repr)

       # graph_embed = node_repr.mean(0)
        return node_repr



class RGCNLayer2(nn.Module):
    def __init__(self, inp_dim, out_dim, aggregator, attn_rel_emb_dim, num_rels, num_bases=-1, bias=None,
                 activation=None, dropout=0.0, edge_dropout=0.0, is_input_layer=False, has_attn=True):
        super(RGCNLayer2, self).__init__()
        self.bias = bias
        self.activation = activation

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
            nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))

        self.aggregator = aggregator

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if edge_dropout:
            self.edge_dropout = nn.Dropout(edge_dropout)
        else:
            self.edge_dropout = Identity()

        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.attn_rel_emb_dim = attn_rel_emb_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        self.has_attn = has_attn

        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # add basis weights
        # self.weight = basis_weights
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.inp_dim, self.out_dim))
        self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        if self.has_attn:
            self.A = nn.Linear(2 * self.inp_dim + 2 * self.attn_rel_emb_dim, inp_dim)
            self.B = nn.Linear(inp_dim, 1)

        self.self_loop_weight = nn.Parameter(torch.Tensor(self.inp_dim, self.out_dim))

        nn.init.xavier_uniform_(self.self_loop_weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))

    def propagate(self, node_feat, e2n_sp, total_target_relation, total_edge, total_relation_embed, total_relation):
        # generate all weights from bases
        weight = self.weight.view(self.num_bases, self.inp_dim * self.out_dim)
        weight = torch.matmul(self.w_comp, weight).view(self.num_rels, self.inp_dim, self.out_dim)

        edge_dropout = self.edge_dropout(torch.ones(total_edge.shape[1], 1)).cuda()

        total_source = total_edge[0, :]
        total_source_embed = node_feat.index_select(0, total_source)
        total_target = total_edge[1, :]
        total_target_embed = node_feat.index_select(0, total_target)
      
 
        w = weight.index_select(0, total_relation)

     #   print('total_target_embed: ', total_target_embed.shape, 'w ',  w.shape) 
        msg = edge_dropout * torch.bmm(total_source_embed.unsqueeze(1), w).squeeze(1)

       # curr_emb = torch.mm(total_target_embed, self.self_loop_weight)  # (B, F)

        curr_emb = torch.mm(node_feat, self.self_loop_weight)  # (B, F)

        if self.has_attn:
                e = torch.cat([total_source_embed, total_target_embed, total_relation_embed, total_target_relation], dim=1)
                a = torch.sigmoid(self.B(F.relu(self.A(e))))
        else:
                a = torch.ones((total_edge.shape[1], 1)).cuda()
       
        attn_msg = msg * a

     #   curr_emb = curr_emb + attn_msg
     #   node_feat = agg_message = gnn_spmm(e2n_sp, curr_emb)
        attn_msg =  gnn_spmm(e2n_sp, attn_msg)

        node_feat = F.relu(curr_emb + attn_msg)

        return node_feat

    def forward(self, node_feat, e2n_sp, total_target_relation, total_edge, total_relation_embed, total_relation):

        node_repr = self.propagate(node_feat, e2n_sp, total_target_relation, total_edge, total_relation_embed, total_relation)

        # apply bias and activation
        node_repr = node_repr
        if self.bias:
            node_repr = node_repr + self.bias
        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout:
            node_repr = self.dropout(node_repr)

       # graph_embed = node_repr.mean(0)
        return node_repr


class Aggregator(nn.Module):
    def __init__(self, emb_dim):
        super(Aggregator, self).__init__()

    def forward(self, node):
        curr_emb = node.mailbox['curr_emb'][:, 0, :]  # (B, F)
        nei_msg = torch.bmm(node.mailbox['alpha'].transpose(1, 2), node.mailbox['msg']).squeeze(1)  # (B, F)
        # nei_msg, _ = torch.max(node.mailbox['msg'], 1)  # (B, F)

        new_emb = self.update_embedding(curr_emb, nei_msg)

        return {'h': new_emb}

    def update_embedding(curr_emb, nei_msg):
        raise NotImplementedError


class SumAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(SumAggregator, self).__init__(emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = nei_msg + curr_emb

        return new_emb






class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)





