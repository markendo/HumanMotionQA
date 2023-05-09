# adopted from https://github.com/abhinanda-punnakkal/BABEL/blob/main/action_recognition/graph/ntu_rgb_d.py
# and https://github.com/abhinanda-punnakkal/BABEL/blob/main/action_recognition/graph/tools.py
import sys
import numpy as np

num_node = 22
self_link = [(i, i) for i in range(num_node)]

# zero index
inward_ori_index = [(0, 3), (3, 6), (9, 6), (12, 9), (15, 12), (13, 12), (16, 13), (18, 16),
                    (20, 18), (14, 12), (17, 14), (19, 17), (21, 19), (1, 0),
                    (4, 1), (7, 4), (10, 7), (2, 0), (5, 2), (8, 5), (11, 8)]

inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A