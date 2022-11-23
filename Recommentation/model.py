import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F

class RecomGCL(nn.Module):
    def __init__(self, cfg, n_u, n_i, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm):
        super(RecomGCL,self).__init__()
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
        self.dropout = cfg.dropout
        self.batch_user = cfg.batch_size
        self.l = cfg.layers
        self.eps = cfg.eps
        self.lr = cfg.lr
        self.device = cfg.device
        self.margin = cfg.margin
        self.weight = cfg.max_samp_pos/cfg.max_samp_neg

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
        self.Ws = nn.ModuleList([W_contrastive(cfg.dim) for i in range(self.l)])

        self.InWs = W_contrastive(cfg.dim)

        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=self.lambda_3, lr=self.lr)
        self.CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=150, eta_min=0)

    def W_contrastive(self, x):
        return x @ self.W

    def predict(self, uids):
        preds = self.E_u[uids] @ self.E_i.T
        mask = self.train_csr[uids.cpu().numpy()].toarray()
        mask = torch.Tensor(mask)
        preds = preds * (1 - mask)
        predictions = preds.argsort(descending=True)
        return predictions

    def forward(self, uids, iids, pos, neg):

        for layer in range(1, self.l + 1):
            # GNN propagation
            # Dropout
            self.Z_du_list[layer] = self.act(spmm(sparse_dropout(self.adj_norm, self.dropout), self.E_i_list[layer - 1]))
            self.Z_di_list[layer] = self.act(spmm(sparse_dropout(self.adj_norm, self.dropout).transpose(0, 1), self.E_u_list[layer - 1]))


            # # Add Random Noise
            # self.Z_u_list[layer] = self.act(spmm(self.adj_norm, self.E_i_list[layer - 1]))
            # random_noise_u = torch.empty(self.E_u_list[layer - 1].shape).uniform_()
            # self.Z_u_list[layer] += torch.mul(torch.sign(self.Z_u_list[layer]),
            #                                   F.normalize(random_noise_u, p=2, dim=1)) * self.eps
            #
            # self.Z_i_list[layer] = self.act(spmm(self.adj_norm.transpose(0, 1), self.E_u_list[layer - 1]))
            # random_noise_i = torch.empty(self.E_i_list[layer - 1].shape).uniform_()
            # self.Z_i_list[layer] += torch.mul(torch.sign(self.Z_i_list[layer]),
            #                                   F.normalize(random_noise_i, p=2, dim=1)) * self.eps

            # svd_adj propagation
            vt_ei = self.vt @ self.E_i_list[layer - 1]
            self.G_u_list[layer] = self.act(self.u_mul_s @ vt_ei)
            ut_eu = self.ut @ self.E_u_list[layer - 1]
            self.G_i_list[layer] = self.act(self.v_mul_s @ ut_eu)

            # aggregate
            # self.E_u_list[layer] = self.Z_u_list[layer] + self.Z_du_list[layer] + self.E_u_list[layer - 1]
            # self.E_i_list[layer] = self.Z_i_list[layer] + self.Z_di_list[layer] + self.E_i_list[layer - 1]
            self.E_u_list[layer] = self.Z_du_list[layer] + self.E_u_list[layer - 1]
            self.E_i_list[layer] = self.Z_di_list[layer] + self.E_i_list[layer - 1]

        # aggregate across layers
        self.E_u = sum(self.E_u_list)
        self.E_i = sum(self.E_i_list)

        # Noise_loss
        # loss_n_u = self.L_Loss(uids, self.Z_u_list, self.G_u_list)
        # loss_n_i = self.L_Loss(iids, self.Z_i_list, self.G_i_list)

        # # Noise_loss layer1 vs layer2
        # loss_nn_u = self.view_InterLoss(uids, self.Z_u_list)
        # loss_nn_i = self.view_InterLoss(uids, self.Z_i_list)

        # Dropout_loss
        # loss_d_u = self.L_Loss(uids, self.Z_du_list, self.G_u_list)
        # loss_d_i = self.L_Loss(iids, self.Z_di_list, self.G_i_list)

        # # Dropout_loss layer1 vs layer2
        # loss_dd_u = self.view_InterLoss(uids, self.Z_du_list)
        # loss_dd_i = self.view_InterLoss(uids, self.Z_di_list)
        #
        # # N-D loss
        # loss_nd_u = self.L_Loss(uids, self.Z_u_list, self.Z_du_list)
        # loss_nd_i = self.L_Loss(iids, self.Z_i_list, self.Z_di_list)

        # loss_n = loss_n_u + loss_n_i
        # loss_d = loss_d_u + loss_d_i
        # loss_nn = loss_nn_u + loss_nn_i
        # loss_dd = loss_dd_u + loss_dd_i
        # loss_nd = loss_nd_u + loss_nd_i
        loss_r = self.U_IConstLoss(uids, pos, neg)
        loss_d = 0
        # total loss
        loss = loss_r + self.lambda_1
               # + self.lambda_2 * (loss_nn + loss_dd) + self.lambda_3 * loss_nd
        # print('loss',loss.item(),'loss_r',loss_r.item(),'loss_s',loss_s.item())
        return loss, loss_r, loss_d, loss_d

    # local_global loss
    def view_InterLoss(self, ids, Z):
        loss = 0

        mask = (torch.rand(len(ids)) > 0.5).float().to(self.device)
        gnn = F.normalize(Z[1][ids], p=2, dim=1)

        hyper = F.normalize(Z[2][ids], p=2, dim=1)
        hyper = self.InWs(hyper)

        pos_score = torch.exp((gnn * hyper).sum(1) / self.temp)
        neg_score = torch.exp(gnn @ hyper.T / self.temp).sum(1)
        loss_s = ((-1 * torch.log(pos_score / (neg_score + 1e-8) + 1e-8)) * mask).sum()
        loss = loss + loss_s
        return loss

    # local_global loss
    def L_Loss(self, uids, iids):
        # cl loss
        loss_s = 0
        for l in range(1, self.l + 1):
            u_mask = (torch.rand(len(uids)) > 0.5).float().to(self.device)

            gnn_u = nn.functional.normalize(self.Z_u_list[l][uids], p=2, dim=1)
            hyper_u = nn.functional.normalize(self.G_u_list[l][uids], p=2, dim=1)
            hyper_u = self.Ws[l - 1](hyper_u)
            pos_score = torch.exp((gnn_u * hyper_u).sum(1) / self.temp)
            neg_score = torch.exp(gnn_u @ hyper_u.T / self.temp).sum(1)
            loss_s_u = ((-1 * torch.log(pos_score / (neg_score + 1e-8) + 1e-8)) * u_mask).sum()
            loss_s = loss_s + loss_s_u

            i_mask = (torch.rand(len(iids)) > 0.5).float().to(self.device)

            gnn_i = nn.functional.normalize(self.Z_i_list[l][iids], p=2, dim=1)
            hyper_i = nn.functional.normalize(self.G_i_list[l][iids], p=2, dim=1)
            hyper_i = self.Ws[l - 1](hyper_i)
            pos_score = torch.exp((gnn_i * hyper_i).sum(1) / self.temp)
            neg_score = torch.exp(gnn_i @ hyper_i.T / self.temp).sum(1)
            loss_s_i = ((-1 * torch.log(pos_score / (neg_score + 1e-8) + 1e-8)) * i_mask).sum()
            loss_s = loss_s + loss_s_i


    def U_IConstLoss(self, uids, pos, neg):
        loss_r = 0
        loss_d = 0
        for i in range(len(uids)):
            u = uids[i]
            u_emb = self.E_u[u]
            u_pos = pos[i]
            u_neg = neg[i]
            pos_emb = self.E_i[u_pos]
            neg_emb = self.E_i[u_neg]

            pos_scores = torch.matmul(u_emb, pos_emb.T)
            neg_scores = torch.matmul(u_emb, neg_emb.T)

            # a = pos_scores / (torch.sqrt(torch.sum(torch.pow(u_emb, 2))) * torch.sqrt(torch.sum(torch.pow(pos_emb, 2))))
            # s = torch.sqrt(torch.sum(torch.pow(u_emb, 2))) * torch.sqrt(torch.sum(torch.pow(pos_emb, 2)))
            # loss = torch.log(1. + torch.exp(neg_scores)/torch.exp(s * torch.cos(a + 1.)))


            minIndex_d_pos = pos_scores.argmin()
            pos_d = (u_emb @ pos_emb[minIndex_d_pos]).sum()

            maxIndex_d_neg = neg_scores.argmax()
            neg_d = (u_emb @ neg_emb[maxIndex_d_neg]).sum()

            if  neg_d - pos_d > 0.005:
                loss_d = loss_d + 10.*(pos_d - neg_d)

            pos_scores = F.relu(1 - pos_scores)
            neg_scores = F.relu(neg_scores-self.margin)

            bpr = pos_scores + neg_scores.sum(dim=-1)
            loss_r = loss_r + bpr.sum() + loss_d
        loss_r = loss_r / self.batch_user
        return loss_r

class W_contrastive(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d,d)))

    def forward(self,x):
        return x @ self.W


