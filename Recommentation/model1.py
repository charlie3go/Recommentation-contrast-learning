import torch
import torch.nn as nn
import torch.nn.functional as F


class RecomGCL(nn.Module):
    def __init__(self, cfg, n_u, n_i, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm):
        super(RecomGCL, self).__init__()
        '''
        cfg:Config setting
        n_u:users nums
        n_i:items nums
        dim：input_dim
        u_mul_s, v_mul_s, ut, vt ---> SVD
        train_csr:sparse_mat
        adj_norm:adj
        '''
        self.temp = cfg.temp
        self.lambda_1 = cfg.lambda1
        self.lambda_2 = cfg.lambda2
        self.lambda_3 = cfg.lambda3
        self.lambda_4 = cfg.lambda4
        self.dropout = cfg.dropout
        self.batch_user = cfg.batch_size
        self.l = cfg.layers
        self.eps = cfg.eps
        self.lr = cfg.lr
        self.device = cfg.device

        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u, cfg.dim)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i, cfg.dim)))
        self.E_u_list = [None] * (self.l + 1)
        self.E_i_list = [None] * (self.l + 1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0

        self.Z_u_list = [None] * (self.l + 1)
        self.Z_i_list = [None] * (self.l + 1)

        self.Z_du_list = [None] * (self.l + 1)
        self.Z_di_list = [None] * (self.l + 1)

        self.G_u_list = [None] * (self.l + 1)
        self.G_i_list = [None] * (self.l + 1)

        self.act = nn.LeakyReLU(0.5)
        self.W_dd = W_contrastive_DD(cfg.dim)
        self.W_nn = W_contrastive_NN(cfg.dim)
        self.W_dg = nn.ModuleList([W_contrastive_DG(cfg.dim) for i in range(self.l)])
        self.W_ng = nn.ModuleList([W_contrastive_NG(cfg.dim) for i in range(self.l)])
        self.W_nd = nn.ModuleList([W_contrastive_ND(cfg.dim) for i in range(self.l)])

        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=self.lambda_4, lr=self.lr)

    def W_contrastive(self, x):
        return x @ self.W

    def predict(self, uids):
        preds = self.E_u[uids] @ self.E_i.T
        mask = self.train_csr[uids.cpu().numpy()].toarray()
        mask = torch.Tensor(mask).to(cfg.device)
        preds = preds * (1 - mask)
        predictions = preds.argsort(descending=True)
        return predictions

    def forward(self, uids, iids, pos, neg):
        for layer in range(1, self.l + 1):
            # GNN propagation
            # Dropout
            self.Z_du_list[layer] = self.act(
                spmm(sparse_dropout(self.adj_norm, self.dropout), self.E_i_list[layer - 1]))
            self.Z_di_list[layer] = self.act(
                spmm(sparse_dropout(self.adj_norm, self.dropout).transpose(0, 1), self.E_u_list[layer - 1]))

            # Add Random Noise
            self.Z_u_list[layer] = self.act(spmm(self.adj_norm, self.E_i_list[layer - 1]))
            random_noise_u = torch.empty(self.E_u_list[layer - 1].shape).uniform_().to(self.device)
            self.Z_u_list[layer] += torch.mul(torch.sign(self.Z_u_list[layer]),
                                              F.normalize(random_noise_u, p=2, dim=1)) * self.eps

            self.Z_i_list[layer] = self.act(spmm(self.adj_norm.transpose(0, 1), self.E_u_list[layer - 1]))
            random_noise_i = torch.empty(self.E_i_list[layer - 1].shape).uniform_().to(self.device)
            self.Z_i_list[layer] += torch.mul(torch.sign(self.Z_i_list[layer]),
                                              F.normalize(random_noise_i, p=2, dim=1)) * self.eps

            # svd_adj propagation
            vt_ei = self.vt @ self.E_i_list[layer - 1]
            self.G_u_list[layer] = self.act(self.u_mul_s @ vt_ei)
            ut_eu = self.ut @ self.E_u_list[layer - 1]
            self.G_i_list[layer] = self.act(self.v_mul_s @ ut_eu)

            # aggregate
            self.E_u_list[layer] = self.Z_u_list[layer] + self.Z_du_list[layer] + self.E_u_list[layer - 1]
            self.E_i_list[layer] = self.Z_i_list[layer] + self.Z_di_list[layer] + self.E_i_list[layer - 1]

        # aggregate across layers
        self.E_u = sum(self.E_u_list)
        self.E_i = sum(self.E_i_list)

        # Noise_loss
        loss_n_u = self.L_Loss(uids, self.Z_u_list, self.G_u_list, self.W_ng)
        loss_n_i = self.L_Loss(iids, self.Z_i_list, self.G_i_list, self.W_ng)

        # Noise_loss layer1 vs layer2
        loss_nn_u = self.view_InterLoss(uids, self.Z_u_list, self.W_nn)
        loss_nn_i = self.view_InterLoss(iids, self.Z_i_list, self.W_nn)

        # Dropout_loss
        loss_d_u = self.L_Loss(uids, self.Z_du_list, self.G_u_list, self.W_dg)
        loss_d_i = self.L_Loss(iids, self.Z_di_list, self.G_i_list, self.W_dg)

        # Dropout_loss layer1 vs layer2
        loss_dd_u = self.view_InterLoss(uids, self.Z_du_list, self.W_dd)
        loss_dd_i = self.view_InterLoss(iids, self.Z_di_list, self.W_dd)

        # N-D loss
        loss_nd_u = self.L_Loss(uids, self.Z_u_list, self.Z_du_list, self.W_nd)
        loss_nd_i = self.L_Loss(iids, self.Z_i_list, self.Z_di_list, self.W_nd)

        loss_n = loss_n_u + loss_n_i
        loss_d = loss_d_u + loss_d_i
        loss_nn = loss_nn_u + loss_nn_i
        loss_dd = loss_dd_u + loss_dd_i
        loss_nd = loss_nd_u + loss_nd_i
        loss_r = self.U_IConstLoss(uids, pos, neg)

        # total loss
        loss = loss_r + self.lambda_1 * (loss_n + loss_d) + self.lambda_2 * (loss_nn + loss_dd + loss_nd)
        return loss, loss_r, loss_d, loss_n

    # local_global loss
    def view_InterLoss(self, ids, Z, W):
        loss = 0
        mask = (torch.rand(len(ids)) > 0.5).float().to(self.device)
        gnn = F.normalize(Z[1][ids], p=2, dim=1)

        hyper = F.normalize(Z[2][ids], p=2, dim=1)
        hyper = W(hyper)

        pos_score = torch.exp((gnn * hyper).sum(1) / self.temp)
        neg_score = torch.exp(gnn @ hyper.T / self.temp).sum(1)
        loss_s = ((-1 * torch.log(pos_score / (neg_score + 1e-8) + 1e-8)) * mask).sum()
        loss = loss + loss_s
        return loss

    # local_global loss
    def L_Loss(self, ids, Z, G, W):
        loss = 0
        for l in range(1, self.l + 1):
            mask = (torch.rand(len(ids)) > 0.5).float().to(self.device)
            gnn = F.normalize(Z[l][ids], p=2, dim=1)

            hyper = F.normalize(G[l][ids], p=2, dim=1)
            hyper = W[l - 1](hyper)

            pos_score = torch.exp((gnn * hyper).sum(1) / self.temp)
            neg_score = torch.exp(gnn @ hyper.T / self.temp).sum(1)
            loss_s = ((-1 * torch.log(pos_score / (neg_score + 1e-8) + 1e-8)) * mask).sum()
            loss = loss + loss_s
        return loss

    def U_IConstLoss(self, uids, pos, neg):
        loss_r = 0
        for i in range(len(uids)):
            loss_d = 0.

            u = uids[i]
            u_emb = self.E_u[u]
            u_pos = pos[i]
            u_neg = neg[i]
            pos_emb = self.E_i[u_pos]
            neg_emb = self.E_i[u_neg]

            pos_scores = torch.matmul(u_emb, pos_emb.T)
            neg_scores = torch.matmul(u_emb, neg_emb.T)

            # 困难正样本
            minIndex_d_pos = pos_scores.argmin()
            pos_d = (u_emb @ pos_emb[minIndex_d_pos]).sum()

            maxIndex_d_neg = neg_scores.argmax()
            neg_d = (u_emb @ neg_emb[maxIndex_d_neg]).sum()

            if neg_d - pos_d > 0.005:
                loss_d = loss_d + 10. * (neg_d - pos_d)

            bpr = F.relu(1 - pos_scores + neg_scores)
            loss_r = loss_r + bpr.sum() + loss_d

        loss_r = loss_r / self.batch_user
        return loss_r


class W_contrastive_DG(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d, d)))

    def forward(self, x):
        return x @ self.W


class W_contrastive_NG(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d, d)))

    def forward(self, x):
        return x @ self.W


class W_contrastive_ND(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d, d)))

    def forward(self, x):
        return x @ self.W


class W_contrastive_DD(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d, d)))

    def forward(self, x):
        return x @ self.W


class W_contrastive_NN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d, d)))

    def forward(self, x):
        return x @ self.W