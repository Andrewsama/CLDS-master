import world
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PureBPR(nn.Module):
    def __init__(self, config, dataset):
        super(PureBPR, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        print("using Normal distribution initializer")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss


class LightGCN(nn.Module):
    def __init__(self, config, dataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self._init_weight()

    def _init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['layer']
        self.embedding_user = torch.nn.Embedding(  # user embedding
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_user1 = torch.nn.Embedding(    # user embedding
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_user2 = torch.nn.Embedding(  # user embedding
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(    # item embedding
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_social = torch.nn.Embedding(  # social_user embedding
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_user1.weight, std=0.1)
        nn.init.normal_(self.embedding_user2.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        nn.init.normal_(self.embedding_social.weight, std=0.1)

        # self.embedding_user.weight = F.normalize(self.embedding_user.weight,p=2,dim=1)
        # self.embedding_item.weight = F.normalize(self.embedding_item.weight,p=2,dim=1)

        self.f = nn.Sigmoid()
        self.interactionGraph, self.interactionGraph2 = self.dataset.getInteractionGraph()
        print(f"{world.model_name} is already to go")


    def getUsersRating(self, users):
        all_users, all_items = self.final_user, self.final_item
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating


class CLDS(LightGCN):
    def _init_weight(self):
        super(CLDS, self)._init_weight()
        self.socialGraph1 = self.dataset.getSocialGraph1()
        self.socialGraph2 = self.dataset.getSocialGraph2()
        # self.socialGraph3 = self.dataset.getSocialGraph3()
        self.mapping = nn.Linear(2 * self.latent_dim, self.latent_dim, bias=False)
        self.f_k = nn.Bilinear(self.latent_dim, self.latent_dim, 1)
        self.f_k2 = nn.Bilinear(self.latent_dim, self.latent_dim, 1)
        self.social_s = nn.Linear(2 * self.latent_dim, self.latent_dim, bias=False)
        self.social_u = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.social_c = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.social_i = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.Graph_Comb = Graph_Comb(self.latent_dim)

        # a,b = list(range(1892)),list(range(1892))
        # a,b = list(range(7375)),list(range(7375))
        a,b = list(range(world.n_node)),list(range(world.n_node))
        np.random.shuffle(a)
        np.random.shuffle(b)
        self.shuffle_index1, self.shuffle_index2 = torch.tensor(a),torch.tensor(b)
        # self.item_comb = item_comb(self.latent_dim)

    def computer(self, epoch):
        A = self.interactionGraph
        A2 = self.interactionGraph2
        if epoch < 2000:
            users_emb = self.embedding_user.weight
            items_emb = self.embedding_item.weight
            all_emb = torch.cat([users_emb, items_emb])

            S1 = self.socialGraph1  # user * user
            embs = [all_emb]
            for layer in range(self.n_layers):
                if layer == 0:

                    all_emb_interaction = torch.sparse.mm(A, all_emb)  # 19524 * 64
                else:
                    all_emb_interaction = torch.sparse.mm(A2, all_emb)  # 19524 * 64
                users_emb_interaction, items_emb_next = torch.split(all_emb_interaction,
                                                                    [self.num_users, self.num_items])
                users_emb_next = torch.tanh(self.social_i(users_emb_interaction))  # 1892 * 64
                all_emb = torch.cat([users_emb_next, items_emb_next])  # 19524 * 64


                users_emb_social = torch.sparse.mm(S1, users_emb)  # 1892 * 32
                users_emb = torch.tanh(self.social_c(users_emb_social))


                users = self.social_s(torch.cat([users_emb_next, users_emb], dim=1))
                users = users / users.norm(2)
                embs.append(torch.cat([users, items_emb_next]))
            embs = torch.stack(embs, dim=1)
            final_embs = torch.mean(embs, dim=1)
            # final_embs = all_emb
            users, items = torch.split(final_embs, [self.num_users, self.num_items])
            # users, items = users_emb_next, items_emb_next
            self.final_user, self.final_item = users, items

            self.embedding_user1.weight.data.copy_(users_emb.detach())
            # self.embedding_user2.weight.data.copy_(users_emb.detach())
            # self.embedding_user1.weight = users_emb.detach()
            # self.embedding_user2.weight = users_emb.detach()
            return users, items, 0, 0

        items_emb = self.embedding_item.weight
        users1_emb = self.embedding_user1.weight
        users2_emb = self.embedding_user2.weight
        users_emb = (users1_emb + users2_emb) / 2
        all_emb = torch.cat([users_emb, items_emb])

        S1 = self.socialGraph2  # user * user
        S2 = self.socialGraph2  # user * user

        # a, b = list(range(1892)), list(range(1892))
        # a, b = list(range(7375)), list(range(7375))
        a, b = list(range(world.n_node)), list(range(world.n_node))
        np.random.shuffle(a)
        np.random.shuffle(b)
        shuffle_index1, shuffle_index2 = torch.tensor(a), torch.tensor(b)
        users1_neg = users1_emb[shuffle_index1]
        users2_neg = users2_emb[shuffle_index2]

        embs = [all_emb]
        logits_true = []
        logits_false = []
        # all_social = [social_emb]
        for layer in range(self.n_layers):
            if layer == 0:

                all_emb_interaction = torch.sparse.mm(A, all_emb)  # 19524 * 64
            else:
                all_emb_interaction = torch.sparse.mm(A2, all_emb)  # 19524 * 64

            # all_emb_interaction = torch.sparse.mm(A, all_emb)  # 19524 * 64
            users_emb_interaction, items_emb_next = torch.split(all_emb_interaction, [self.num_users, self.num_items])
            users_emb_next = torch.tanh(self.social_i(users_emb_interaction))  # 1892 * 64
            all_emb = torch.cat([users_emb_next, items_emb_next])  # 19524 * 64


            users1_emb_social = torch.sparse.mm(S1, users1_emb)  # 1892 * 32
            users1_emb_social = torch.tanh(self.social_c(users1_emb_social))  # 1892 * 64

            users2_emb_social = torch.sparse.mm(S2, users2_emb)  # 1892 * 64
            users2_emb_social = torch.tanh(self.social_c(users2_emb_social))  # 1892 * 64

            users1_neg = torch.sparse.mm(S1, users1_neg)  # 1892 * 32
            users1_neg = torch.tanh(self.social_c(users1_neg))  # 1892 * 64
            users2_neg = torch.sparse.mm(S2, users2_neg)  # 1892 * 64
            users2_neg = torch.tanh(self.social_c(users2_neg))  # 1892 * 64

            c_1 = torch.mean(users1_emb_social, dim=0)        # 1892 * 1
            c_1 = c_1.expand_as(users1_emb_social)            # 1892 * 64
            c_2 = torch.mean(users2_emb_social, dim=0)        # 1892 * 1
            c_2 = c_2.expand_as(users2_emb_social)            # 1892 * 64
            sc_1 = self.f_k(users2_emb_social, c_1).T         # 1 * 1892
            sc_2 = self.f_k(users1_emb_social, c_2).T         # 1 * 1892
            sc_3 = self.f_k(users2_neg, c_1).T                # 1 * 1892
            sc_4 = self.f_k(users1_neg, c_2).T                # 1 * 1892
            logits_true.append(torch.cat((sc_1, sc_2), dim = 1))
            logits_false.append(torch.cat((sc_3, sc_4), dim = 1))


            users_emb = (users1_emb_social + users2_emb_social) / 2
            users = self.social_s(torch.cat([users_emb_next, users_emb], dim=1))
            users = users / users.norm(2)
            embs.append(torch.cat([users, items_emb_next]))
        embs = torch.stack(embs, dim=1)
        logits_true.extend(logits_false)
        logits = torch.stack(logits_true, dim = 2)
        logits = logits.view(1,-1,1).squeeze(2)

        final_embs = torch.mean(embs, dim=1)
        # final_embs = all_emb
        users, items = torch.split(final_embs, [self.num_users, self.num_items])
        # users, items = users_emb_next, items_emb_next
        self.final_user, self.final_item = users, items
        return users, items, logits, 1

    def getEmbedding(self, users, pos_items, neg_items, epoch):
        all_users, all_items, logits, tag = self.computer(epoch)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        # users2_emb_ego = self.embedding_user2(users)
        # users_emb_ego = self.mapping(torch.cat([users1_emb_ego, users2_emb_ego], dim=1))
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, logits, tag


    def bpr_loss(self, users, pos, neg, epoch):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0, logits, tag) = self.getEmbedding(users.long(), pos.long(), neg.long(), epoch)
        # print('pos_emb_shape:',pos_emb.shape)
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        '''GOR?'''

        M_ = torch.mul(users_emb, neg_emb)
        M1 = torch.pow(torch.mean(M_), 2)
        M2 = torch.mean(torch.pow(M_, 2)) - torch.tensor(1/float(self.latent_dim))
        attr_loss = M1 + F.softplus(M2)


        loss = torch.mean(F.softplus(neg_scores - pos_scores))

        # print('loss',loss)
        # print('reg_loss',reg_loss)

        # return loss, reg_loss

        loss_lbl = 0
        if tag == 1:
            b_xent = nn.BCEWithLogitsLoss()
            lbl_1 = torch.ones((1, logits.detach().shape[1] // 2))
            lbl_2 = torch.zeros(lbl_1.shape)
            lbl = torch.cat((lbl_1, lbl_2), 1).cuda()
            loss_lbl = b_xent(logits, lbl)
        return loss, reg_loss, attr_loss, loss_lbl, tag

class Graph_Comb(nn.Module):
    def __init__(self, embed_dim):
        super(Graph_Comb, self).__init__()
        self.att_x = nn.Linear(embed_dim, embed_dim, bias=False)
        self.att_y = nn.Linear(embed_dim, embed_dim, bias=False)
        self.comb = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x, y):
        h1 = torch.tanh(self.att_x(x))
        h2 = torch.tanh(self.att_y(y))
        output = self.comb(torch.cat((h1, h2), dim=1))
        output = output / output.norm(2)
        # output = F.normalize(output, p=2, dim=1)
        return output

class item_comb(nn.Module):
    def __init__(self, embed_dim):
        super(item_comb, self).__init__()
        self.att_x = nn.Linear(embed_dim, embed_dim, bias=False)
        self.att_y = nn.Linear(embed_dim, embed_dim, bias=False)
        self.comb = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x, y):
        h1 = torch.tanh(self.att_x(x))
        h2 = torch.tanh(self.att_y(y))
        output = self.comb(torch.cat((h1, h2), dim=1))
        output = output / output.norm(2)
        return output
