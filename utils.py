import numpy as np
import cPickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def getList(path):
    list = []
    for line in open(path, 'r'):
        list.append(int(line.strip()))
    return list

def getGraph(path, len, attention):
    if attention == 0:
        graph = np.zeros((len, len), dtype="float32")
        for line in open(path, 'r'):
            strs = line.strip().split('\t')
            graph[int(strs[0])][int(strs[1])] = 1.0
            graph[int(strs[1])][int(strs[0])] = 1.0
    else:
        graph = np.zeros((len, len), dtype="int32")
        for line in open(path, 'r'):
            strs = line.strip().split('\t')
            graph[int(strs[0])][int(strs[1])] = int(strs[1]) + 1
            graph[int(strs[1])][int(strs[0])] = int(strs[0]) + 1
    return graph

def normalizeGraph(graph):
    for i in range(graph.shape[0]):
        sum = np.sum(graph[i])
        if sum != 0.0:
            graph[i] = graph[i]/sum
    return graph

def load_data(dataset_str, percent, k, f, attention):
    """Load data."""
    if percent == 80:
        names = ['id', 'label', 'mask', 'graph']
    elif percent == 10:
        names = ['id', 'label', 'mask_per', 'graph']
    else:
        names = ['id', 'label', 'mask', 'graph']
    objects = []
    for i in range(len(names) - 1):
        objects.append(getList("data/" + dataset_str + "/" + f + "/" + names[i] + ".txt"))
    id, label, mask = tuple(objects)
    graph = getGraph("data/" + dataset_str + "/" + f + "/" + names[-1] + ".txt", len(id), attention)
    if attention == 0:
        graph = normalizeGraph(graph)
    inputs = []
    for i in range(k):
        inputs.append(id)
    inputs = np.array(inputs)
    inputs = np.transpose(inputs)
    train_mask = []
    val_mask = []
    test_mask = []
    for i in range(len(mask)):
        if mask[i] == 1:
            train_mask.append(1)
            val_mask.append(0)
            test_mask.append(0)
        elif mask[i] == 2:
            train_mask.append(0)
            val_mask.append(1)
            test_mask.append(0)
        elif mask[i] == 3:
            train_mask.append(0)
            val_mask.append(0)
            test_mask.append(1)
        else:
            train_mask.append(0)
            val_mask.append(0)
            test_mask.append(0)

    # features = sp.vstack((allx, tx)).tolil()
    # features[test_idx_reorder, :] = features[test_idx_range, :]
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    #
    # labels = np.vstack((ally, ty))
    # labels[test_idx_reorder, :] = labels[test_idx_range, :]
    #
    # idx_test = test_idx_range.tolist()
    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y)+500)
    #
    # train_mask = sample_mask(idx_train, labels.shape[0])
    # val_mask = sample_mask(idx_val, labels.shape[0])
    # test_mask = sample_mask(idx_test, labels.shape[0])

    # y_train = np.zeros(labels.shape)
    # y_val = np.zeros(labels.shape)
    # y_test = np.zeros(labels.shape)
    # y_train[train_mask, :] = labels[train_mask, :]
    # y_val[val_mask, :] = labels[val_mask, :]
    # y_test[test_mask, :] = labels[test_mask, :]

    return np.array(id), np.array(label), np.array(train_mask), np.array(val_mask), np.array(test_mask), graph, inputs


def getEmbedding(efile, len1, len2):
    embedding = np.zeros((len1, len2), dtype="float32")
    for line in open(efile, 'r'):
        strs = line.strip().split(' ')
        for i in range(len2):
            embedding[int(strs[0])][i] = float(strs[i + 1])
    return embedding

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
