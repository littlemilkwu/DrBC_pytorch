import numpy as np
from multiprocessing import Pool, cpu_count
import itertools
import random
import networkx as nx
import torch
from tqdm import tqdm
import math
RS = np.random.RandomState(11)
test1_path = 'hw1_data/Synthetic/5000/'


def read_test1_data(file_num:int):
    global test1_path

    g = nx.Graph(nx.read_edgelist(f"{test1_path}{file_num}.txt", nodetype=int, create_using=nx.DiGraph))
    
    # y
    with open(f"{test1_path}{file_num}_score.txt", 'r') as f:
        all_bc = []
        for line in f.readlines():
            node_index, bc = line.split('\t')
            all_bc.append(float(bc))

    # edge_index
    edge_index = []
    with open(f"{test1_path}{file_num}.txt") as f:
        for line in f.readlines():
            s, t = line.split('\t')
            edge_index.append([int(s), int(t)])
    return g, all_bc, edge_index

def gen_graph(NUM_MIN, NUM_MAX):
    node_num = np.random.randint(NUM_MIN, high=NUM_MAX)
    g = nx.powerlaw_cluster_graph(node_num, m=4, p=0.05)
    return g

def prepare_synthetic(synthetic_num:int, num_range:tuple):
    num_min, num_max = num_range
    g_list = []
    dg_list = []
    bc_list = []
    for i in tqdm(range(synthetic_num), desc='[Generating new training graph]'):
        g = gen_graph(num_min, num_max)
        g_list.append(g)
        dg_list.append([g.degree[i] for i in range(g.number_of_nodes())])
        # if synthetic nodes greater than 1000, switch to parallel version would faster
        # bc_list.append(list(betweenness_centrality_parallel(g)))
        bc_ = list(dict(nx.betweenness_centrality(g)).values())
        bc_ = [x for x in bc_]
        bc_list.append(bc_)
        
    return g_list, dg_list, bc_list

def prepare_test1(test1_num:int):
    v_data = []
    for i in tqdm(range(test1_num), desc='[Reading test1 graph]'):
        g, bc, edge_index = read_test1_data(i)
        # g_list.append(g)
        # dg_list.append([g.degree[i] for i in range(g.number_of_nodes())])
        # bc = [x for x in bc]
        # bc_list.append(bc)
        X = [[float(g.degree(i)), 1., 1.] for i in range(g.number_of_nodes())]
        y = bc
        X = torch.Tensor(X)
        y = torch.Tensor(y)
        edge_index = torch.Tensor(edge_index).T.to(torch.int64)
        v_data.append([X, y, edge_index])
    return v_data

def shuffle_graph(train_g:list, train_dg:list, train_bc:list):
    temp = list(zip(train_g, train_dg, train_bc))
    random.shuffle(temp)
    train_g, train_dg, train_bc = zip(*temp)
    return train_g, train_dg, train_bc

def preprocessing_data(ls_g:list, ls_dg:list, ls_bc:list):
    X = np.zeros(shape=(0, 3))
    y = np.zeros(shape=(0, ))
    edge_index = np.zeros(shape=(0, 2))
    pre_index = 0
    for i in range(len(ls_g)):
        assert len(ls_dg[i]) == len(ls_bc[i]) == len(ls_g[i].nodes())
        # make suer is has same nodes number.
        num_node = len(ls_dg[i])
        _X = np.expand_dims(np.array(ls_dg[i]), axis=-1)
        _it = np.ones(shape=(_X.shape[0], 2))
        _X = np.hstack([_X, _it])
        X = np.append(X, _X, axis=0)

        _y = np.array(ls_bc[i])
        y = np.append(y, _y, axis=0)

        _edge = np.array([e for e in ls_g[i].edges]) + pre_index
        edge_index = np.append(edge_index, _edge, axis=0)
        pre_index += num_node
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    edge_index = torch.Tensor(edge_index).T.to(torch.int64)

    return X, y, edge_index

def get_pairwise_ids(g_list):
    s_ids = np.zeros(shape=(0, ), dtype=int)
    t_ids = np.zeros(shape=(0, ), dtype=int)
    pre_index = 0
    for g in g_list:
        num_node = len(g.nodes())
        ids_1 = np.repeat(np.arange(pre_index, pre_index+num_node), 5)
        ids_2 = np.repeat(np.arange(pre_index, pre_index+num_node), 5)

        np.random.shuffle(ids_1)
        np.random.shuffle(ids_2)

        s_ids = np.append(s_ids, ids_1, axis=0)
        t_ids = np.append(t_ids, ids_2, axis=0)
        pre_index += num_node
    return s_ids, t_ids

def top_n_acc(ar1, ar2, n=1):
    assert len(ar1) == len(ar2)
    n_num = int(len(ar1) * (n * 0.01))
    union_num = len(set(ar1[:n_num]) & set(ar2[:n_num]))
    return union_num / n_num


if __name__ == "__main__":
    # g_list, dg_list, bc_list = prepare_synthetic(2, (100, 101))
    # shuffle_graph(g_list, dg_list, bc_list)
    # get_pairwise_ids(g_list)
    # g, bc = read_test1_data(0)
    # g_list, dg_list, bc_list = prepare_test1(1)
    # preprocessing_data(g_list, dg_list, bc_list)
    pass