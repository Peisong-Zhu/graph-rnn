import numpy as np
import scipy.sparse as sp

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

    return np.array(id), np.array(label), np.array(train_mask), np.array(val_mask), np.array(test_mask), graph, inputs


def getEmbedding(efile, len1, len2):
    embedding = np.zeros((len1, len2), dtype="float32")
    for line in open(efile, 'r'):
        strs = line.strip().split(' ')
        for i in range(len2):
            embedding[int(strs[0])][i] = float(strs[i + 1])
    return embedding
