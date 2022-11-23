import pickle
import numpy as np
import torch
from config import Config
import torch.nn as nn
import scipy.sparse as sp

cfg = Config()

# load train data
def train_data(path):
    with open(path, 'rb') as f:
        train_data = pickle.load(f)
        return  train_data

# load test data
def test_data(path):
    with open(path, 'rb') as f:
        test_data = pickle.load(f)
        return test_data

def SVD_decomposition(adj):
    svd_u, s, svd_v = torch.svd_lowrank(adj, q=cfg.svd_q)
    u_mul_s = svd_u @ torch.diag(s)
    v_mul_s = svd_v @ torch.diag(s)
    del adj
    del s
    return u_mul_s, v_mul_s, svd_u, svd_v

# sparse_mat to sparse_tensor
def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# adj_normalization
def adj_normalization(adj):
    rowD = np.array(adj.sum(1)).squeeze()
    colD = np.array(adj.sum(0)).squeeze()
    for i in range(len(adj.data)):
        adj.data[i] = adj.data[i] / pow(rowD[adj.row[i]] * colD[adj.col[i]], 0.5)
    adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(adj)
    adj_norm = adj_norm.coalesce()
    return adj_norm

# save model
def save_model(model, save_model_path=cfg.save_model_path,):
    torch.save(model.state_dict(), save_model_path)
    return model

# build user-item: positive and negative samples
def pos_neg(users, train_csr):
    pos, neg = [], []
    iids = set()
    for i in range(len(users)):
        u = users[i]
        u_interact = train_csr[u].toarray()[0]

        positive_items = np.random.permutation(np.where(u_interact == 1)[0])
        negative_items = np.random.permutation(np.where(u_interact == 0)[0])

        pos_item_num = min(cfg.max_samp_pos, len(positive_items))
        # neg_item_num = min(cfg.max_samp_neg, len(negative_items))

        positive_items = positive_items[:pos_item_num]
        negative_items = negative_items[:pos_item_num]
        pos.append(torch.LongTensor(positive_items))
        neg.append(torch.LongTensor(negative_items))
        iids = iids.union(set(positive_items))
        iids = iids.union(set(negative_items))
    iids = torch.LongTensor(list(iids))
    uids = torch.LongTensor(users)

    return iids, uids, pos, neg

# metrics
def metrics(uids, predictions, topk, test_labels):
    user_num = 0
    all_recall = 0
    all_ndcg = 0
    for i in range(len(uids)):
        uid = uids[i]
        prediction = list(predictions[i][:topk])
        label = test_labels[uid]
        if len(label)>0:
            hit = 0
            idcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(topk, len(label)))])
            dcg = 0
            for item in label:
                if item in prediction:
                    hit+=1
                    loc = prediction.index(item)
                    dcg = dcg + np.reciprocal(np.log2(loc+2))
            all_recall = all_recall + hit/len(label)
            all_ndcg = all_ndcg + dcg/idcg
            user_num+=1
    return all_recall/user_num, all_ndcg/user_num

def sparse_dropout(mat, dropout):
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)

def spmm(sp, emb):
    sp = sp.coalesce()
    cols = sp.indices()[1]
    rows = sp.indices()[0]
    col_segs =  emb[cols] * torch.unsqueeze(sp.values(),dim=1)
    result = torch.zeros((sp.shape[0],emb.shape[1]))
    result.index_add_(0, rows, col_segs)
    return result

from matplotlib import pyplot as plt  # 设置中文

def draw(x, y):

    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['axes.unicode_minus'] = False

    plt.bar(x, y)
    plt.title("学生人数")
    plt.show()


