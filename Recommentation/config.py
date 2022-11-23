import torch

class Config:
    data_path = r'C:\Users\Mr.long\Desktop\研二\codes\Recommentation\data'
    dataset = ('amazon', 'gowalla', 'yelp')
    svd_q = 5 # main features number after svd
    epochs = 10 # train iter nums
    save_model_path = r'C:\Users\Mr.long\Desktop\研二\codes\Recommentation\model_weight\model.pt'
    save_optimizer_path = r'C:\Users\Mr.long\Desktop\研二\codes\Recommentation\model_weight'
    batch_size = 256
    max_samp_pos = 40 # pos samples max
    max_samp_neg = 100 # neg samples max
    dim = 32 # input dim
    layers = 2 # GCN layers
    temp = 0.5 # temperature parameters
    lambda1 = 1e-7
    lambda2 = 3e-7
    lambda3 = 1e-4
    dropout = 0.25
    eps = 1e-12
    margin = 0.4
    lr = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

