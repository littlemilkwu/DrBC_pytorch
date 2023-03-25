import numpy as np
from multiprocessing import Pool, cpu_count
import itertools
import random
import networkx as nx
import torch
from tqdm import tqdm
import math
import time
RS = np.random.RandomState(11)
test1_path = 'hw1_data/Synthetic/5000/'
testyt_path = 'hw1_data/youtube/'


def read_test_data(test_file):
    global test1_path
    global testyt_path
    filename = f"{testyt_path}com-youtube" if str(test_file) == 'y' else f"{test1_path}{test_file}"
    g = nx.Graph(nx.read_edgelist(f"{filename}.txt", nodetype=int, create_using=nx.DiGraph))
    
    # y
    with open(f"{filename}_score.txt", 'r') as f:
        all_bc = []
        for line in f.readlines():
            line = line.replace(':', "").replace(' ', '')
            node_index, bc = line.split('\t')
            # all_bc.append(float(bc))
            all_bc.append(math.log(float(bc)+1e-8))

    # edge_index
    edge_index = []
    with open(f"{filename}.txt") as f:
        for line in f.readlines():
            line = line.replace(' ', "\t")
            s, t = line.split('\t')
            edge_index.append([int(s), int(t)])
    return g, all_bc, edge_index

def gen_graph(NUM_MIN, NUM_MAX):
    node_num = np.random.randint(NUM_MIN, high=NUM_MAX)
    g = nx.powerlaw_cluster_graph(node_num, m=4, p=0.05)
    return g

######################### from: https://networkx.org/documentation/stable/auto_examples/algorithms/plot_parallel_betweenness.html
def chunks(l, n):
    """Divide a list of nodes `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x


def betweenness_centrality_parallel(G, processes=None):
    """Parallel betweenness centrality  function"""
    p = Pool(processes=120)
    node_divisor = len(p._pool) * 4
    node_chunks = list(chunks(G.nodes(), G.order() // node_divisor))
    num_chunks = len(node_chunks)
    bt_sc = p.starmap(
        nx.betweenness_centrality_subset,
        zip(
            [G] * num_chunks,
            node_chunks,
            [list(G)] * num_chunks,
            [True] * num_chunks,
            [None] * num_chunks,
        ),
    )

    # Reduce the partial solutions
    # print(bt_sc)
    bt_c = bt_sc[0]
    for bt in bt_sc[1:]:
        for n in bt:
            bt_c[n] += bt[n]
    return bt_c


######################### from: https://networkx.org/documentation/stable/auto_examples/algorithms/plot_parallel_betweenness.html

def prepare_synthetic(synthetic_num:int, num_range:tuple, parallel=False):
    num_min, num_max = num_range
    g_list = []
    dg_list = []
    bc_list = []
    for i in tqdm(range(synthetic_num), desc='[Generating new training graph]'):
        g = gen_graph(num_min, num_max)
        g_list.append(g)
        dg_list.append([g.degree[i] for i in range(g.number_of_nodes())])
        # bc_ = list(dict(nx.betweenness_centrality(g)).values())
        if parallel == True:
            dict_bc = betweenness_centrality_parallel(g)
        else:
            dict_bc = nx.betweenness_centrality(g)
        bc_ = [math.log(dict_bc[x]+1e-8) for x in range(g.number_of_nodes())]
        # bc_ = [dict_bc[x]+1e-8 for x in range(g.number_of_nodes())]
        bc_list.append(bc_)
        
    return g_list, dg_list, bc_list

def prepare_test(test1_num):
    v_data = []
    if str(test1_num) == 'y':
        g, bc, edge_index = read_test_data('y')
        X = [[float(g.degree(i)), 1., 1.] for i in range(g.number_of_nodes())]
        y = bc
        X = torch.Tensor(X)
        y = torch.Tensor(y)
        edge_index = torch.Tensor(edge_index).T.to(torch.int64)
        v_data.append([X, y, edge_index])
        return v_data

    for i in tqdm(range(int(test1_num)), desc='[Reading test1 graph]'):
        g, bc, edge_index = read_test_data(i)
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
    # g, bc = read_test_data(0)
    # g_list, dg_list, bc_list = prepare_test1(1)
    # preprocessing_data(g_list, dg_list, bc_list)
    start_time = time.time()
    g_list, _, bc_list = prepare_synthetic(1, (1000, 1001), parallel=True)
    print(bc_list[0])
    end_time = time.time()
    print(f'used times: {round(end_time - start_time, 1)} secs.')
    pass