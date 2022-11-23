from tqdm import tqdm
from utils import *
from model import RecomGCL
import scipy


# Config setting
cfg = Config()

# datasets
datasets = cfg.dataset
print(f'All datasets are:{[dataset for dataset in datasets]}')
dataset = input('Please input a dataset name:')

# data_path
train_path = cfg.data_path + '\\' + dataset +'\\'+ 'trnMat.pkl'
test_path = cfg.data_path + '\\' + dataset +'\\'+ 'tstMat.pkl'

# load data
train = train_data(train_path)
train_csr = (train!=0).astype(np.float32)

test = test_data(test_path)
test_labels = [[] for i in range(test.shape[0])]
for i in range(len(test.data)):
    row = test.row[i]
    col = test.col[i]
    test_labels[row].append(col)

# svd decomposition
adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce()
u_mul_s, v_mul_s, svd_u, svd_v = SVD_decomposition(adj)

# adj_norm
adj_norm = adj_normalization(train)

best = 0
# validation
def validation(model, batch_loader):
    global best
    all_recall_20 = 0
    all_ndcg_20 = 0
    all_recall_40 = 0
    all_ndcg_40 = 0

    test_uids = np.array([i for i in range(adj_norm.shape[0])])

    for batch in range(batch_loader):
        start = batch * cfg.batch_size
        end = min((batch + 1) * cfg.batch_size, len(test_uids))

        test_uids_input = torch.LongTensor(test_uids[start:end])
        predictions = model.predict(test_uids_input)
        predictions = np.array(predictions.cpu())

        # top@20
        recall_20, ndcg_20 = metrics(test_uids[start:end], predictions, 20, test_labels)
        # top@40
        recall_40, ndcg_40 = metrics(test_uids[start:end], predictions, 40, test_labels)

        all_recall_20 += recall_20
        all_ndcg_20 += ndcg_20
        all_recall_40 += recall_40
        all_ndcg_40 += ndcg_40

    if best < all_recall_20:
        best = all_recall_20
        save_model(model)

    return (all_recall_20, all_ndcg_20, all_recall_40, all_ndcg_40)

# train
def train_loop():
    model = RecomGCL(cfg, adj_norm.shape[0], adj_norm.shape[1], u_mul_s, v_mul_s, svd_u.T, svd_v.T, train_csr, adj_norm)
    for epoch in range(cfg.epochs):
        epoch_loss = 0
        epoch_loss_r = 0
        epoch_loss_n = 0
        epoch_loss_d = 0

        # user node nums
        epoch_users = adj_norm.shape[0]

        # shuffle node
        users = np.random.permutation(adj_norm.shape[0])[:epoch_users]

        batch_loader = int(np.ceil(epoch_users / cfg.batch_size))

        for batch in (range(batch_loader)):
            start = batch * cfg.batch_size
            end = min((batch + 1) * cfg.batch_size, epoch_users)
            batch_users = users[start:end]

            # build pos_neg samples
            iids, uids, pos, neg = pos_neg(batch_users, train_csr)

            # data feed model
            loss, loss_r, loss_n, loss_d = model(uids, iids, pos, neg)
            loss.backward()
            model.optimizer.step()
            model.CosineLR.step()
            model.optimizer.zero_grad()

            print(f'loss:{loss.item()}')

            epoch_loss += loss.cpu().item()
            epoch_loss_r += loss_r.cpu().item()
            epoch_loss_n += loss_n.cpu().item()
            epoch_loss_d += loss_d.cpu().item()

        epoch_loss = epoch_loss / batch_loader
        epoch_loss_r = epoch_loss_r / batch_loader
        epoch_loss_n = epoch_loss_n / batch_loader
        epoch_loss_d = epoch_loss_d / batch_loader
        print(f'Epoch:{epoch}',
              f'Loss:{epoch_loss}',
              f'Loss_r:{epoch_loss_r}',
              f'Loss_n:{epoch_loss_n}',
              f'Loss_d:{epoch_loss_d}',
              )

        # test every 10 epochs
        if (epoch + 1) % 1 == 0:
            print('-------------Start Validation-------------')
            result = validation(model, batch_loader)
            all_recall_20, all_ndcg_20, all_recall_40, all_ndcg_40 = result

            print(f'all_recall_20:{all_recall_20/batch_loader}, all_ndcg_20:{all_ndcg_20/batch_loader}\n',
                  f'all_recall_40:{all_recall_40/batch_loader}, all_ndcg_40:{all_ndcg_40/batch_loader}')


if __name__ == '__main__':
    train_loop()
