import numpy as np
import networkx as nx
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

def gen_graph(NUM_MIN, NUM_MAX):
    node_num = np.random.randint(NUM_MIN, high=NUM_MAX)
    g = nx.powerlaw_cluster_graph(node_num, m=4, p=0.05)
    return g


if __name__ == "__main__":
    # print(gen_graph(10, 20).edges())
    pass
