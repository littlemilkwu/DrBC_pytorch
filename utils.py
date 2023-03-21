import numpy as np
import random
import networkx as nx
import torch
RS = np.random.RandomState(11)
test1_path = 'hw1_data/Synthetic/5000/'


def read_test1_data(file_num:int):
    global test1_path
    # X
    with open(f"{test1_path}{file_num}.txt", 'r') as f:
        text = [l.replace('\t', ' ').replace('\n', '') for l in f.readlines()]
        pair_node = [t.split(' ') for t in text]
        pair_node = [[int(s), int(t)] for s, t in pair_node]
    
    # y
    with open(f"{test1_path}{file_num}_score.txt", 'r') as f:
        dict_bc = dict()
        for line in f.readlines():
            text = line.replace('\t', ' ').replace('\n', '')
            node_index, bc = text.split()
            dict_bc[int(node_index)] = float(bc)

    return pair_node, dict_bc

def prepare_synthetic(synthetic_num:int, num_range:tuple):
    num_min, num_max = num_range
    g_list = []
    dg_list = []
    bc_list = []
    for i in range(synthetic_num):
        g = gen_graph(num_min, num_max)
        g_list.append(g)
        dg_list.append(list(dict(nx.degree(g)).values()))
        bc_list.append(list(nx.betweenness_centrality(g)))
        
    return g_list, dg_list, bc_list


def gen_graph(NUM_MIN, NUM_MAX):
    node_num = np.random.randint(NUM_MIN, high=NUM_MAX)
    g = nx.powerlaw_cluster_graph(node_num, m=4, p=0.05)
    return g

def shuffle_graph(train_g:list, train_dg:list, train_bc:list):
    temp = list(zip(train_g, train_dg, train_bc))
    random.shuffle(temp)
    train_g, train_dg, train_bc = zip(*temp)
    return train_g, train_dg, train_bc

def preprocessing_data(train_g:list, train_dg:list, train_bc:list):
    X = np.zeros(shape=(0, 3))
    y = np.zeros(shape=(0, ))
    edge_index = np.zeros(shape=(0, 2))
    pre_index = 0
    for i in range(len(train_bc)):
        assert len(train_dg[i]) == len(train_bc[i]) == len(train_g[i].nodes())
        # make suer is has same nodes number.
        num_node = len(train_dg[i])
        _X = np.expand_dims(np.array(train_dg[i]), axis=-1)
        _it = np.ones(shape=(_X.shape[0], 2))
        _X = np.hstack([_X, _it])
        X = np.append(X, _X, axis=0)

        _y = np.array(train_bc[i])
        y = np.append(y, _y, axis=0)

        _edge = np.array(list(train_g[i].edges())) + pre_index
        edge_index = np.append(edge_index, _edge, axis=0)

        pre_index += num_node
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    edge_index = torch.Tensor(edge_index).T.to(torch.int64)
    # print(X.shape, y.shape, edge_index.shape)

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


if __name__ == "__main__":
    # print(gen_graph(10, 20).edges())
    pass
