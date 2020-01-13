import torch.nn as nn
import numpy as np
import copy
import random
from channel_importance import channel_importance
from result_save_visualization import *

# パラメータ利用
net = parameter_use('./result/pkl/original_train_epoch50.pkl')
# 畳み込み層のリスト
conv_list = [module for module in net.modules() if isinstance(module, nn.Conv2d)]


class PfgaCnn:
    def __init__(self, gene_len1, gene_len2, conv_num, evaluate_func=None, better_high=True, mutate_rate=0.05):
        self.family = []
        self.gene_len1 = gene_len1
        self.gene_len2 = gene_len2
        self.evaluate_func = evaluate_func
        self.better_high = better_high
        self.mutate_rate = mutate_rate
        self.generation_num = 0
        self.conv_num = conv_num

    def add_new_population(self):
        new_gene = []
        # -1から+1の範囲に正規化した初期個体をランダムに決定
        # a = np.random.rand(self.gene_len1, self.gene_len2[0], self.gene_len2[1])
        # a_max = np.max(a)
        # a_min = np.min(a)
        # y = 2 * (a - a_min) / (a_max - a_min) - 1
        # new_gene.append(y)
        # チャネル重要度が上位10%の個体を初期個体にする
        ch_high, ch_low = channel_importance(self.conv_num)
        # 選択されるチャネルのindex
        ch_index = random.choice(np.concatenate([ch_high, ch_low]))
        # ch_index = random.choice(ch_high10)
        new_gene.append(conv_list[self.conv_num].weight.data.clone().cpu().detach().numpy()[ch_index, :, :, :])
        new_gene.append(None)
        self.family.append(new_gene)

    def pop_num(self):
        return len(self.family)

    def copy_gene(self, g):
        return copy.deepcopy(g)

    def best_gene(self):
        idx = self.family[0]
        for i in self.family:
            if i[1] is not None:
                if idx[1] is not None:
                    if (self.better_high is True and idx[1] < i[1]) or (self.better_high is False and idx[1] > i[1]):
                        idx = i
                else:
                    idx = i
        return idx

    def select_and_delete_gene(self):
        return self.copy_gene(self.family.pop(np.random.randint(0, len(self.family))))

    def crossover(self, p1, p2):
        c1 = self.copy_gene(p1)
        c2 = self.copy_gene(p2)
        ch_seed = [i for i in range(len(c1[0]))]
        val_seed = [i for i in range(len(c1[0][0]))]
        # 一点交叉(チャネルごと交換)
        for i in range(len(c1[0])):
            ch_cross_point1, ch_cross_point2 = random.choice(ch_seed), random.choice(ch_seed)
            if np.random.rand() < 0.5:
                c1[0][ch_cross_point1], c2[0][ch_cross_point2] = c2[0][ch_cross_point2], c1[0][ch_cross_point1]
        # 二点交叉(チャネルの一部を交換)
        # for i in range(len(c1[0])):
        #     ch_cross_point1, ch_cross_point2 = random.choice(ch_seed), random.choice(ch_seed)
        #     val_cross_point1, val_cross_point2 = random.choice(val_seed), random.choice(val_seed)
        #     if val_cross_point1 > val_cross_point2:
        #         val_cross_point1, val_cross_point2 = val_cross_point2, val_cross_point1
        #     if np.random.rand() < 0.5:
        #         c1[0][ch_cross_point1][val_cross_point1:val_cross_point2], c2[0][ch_cross_point2][val_cross_point1:val_cross_point2]\
        #             = c2[0][ch_cross_point2][val_cross_point1:val_cross_point2], c1[0][ch_cross_point1][val_cross_point1:val_cross_point2]
        # 一様交叉
        # for i in range(len(c1[0])):
        #     uniform_mask1 = np.ones(c1[0][i].shape)
        #     uniform_mask1[:int(len(c1[0][i][0]) / 2), :int(len(c1[0][i][0]) / 2)] = 0
        #     np.random.shuffle(uniform_mask1)
        #     np.random.shuffle(uniform_mask1.T)
        #     uniform_mask2 = np.where(uniform_mask1 == 0, 1, 0)
        #     if np.random.rand() < 0.5:
        #         c1[0][i], c2[0][i] = c1[0][i] * uniform_mask1 + c2[0][i] * uniform_mask2, c1[0][i] * uniform_mask2 + c2[0][i] * uniform_mask1
        c1[1], c2[1] = None, None
        return c1, c2

    def mutate(self, g):
        # 摂動
        if np.random.rand() < self.mutate_rate:
            for i in range(len(g[0])):
                g[0][i] *= 1.05
        # 反転
        # if np.random.rand() < self.mutate_rate:
        #     for i in range(len(g[0])):
        #         g[0][i] = -g[0][i]
        # 逆位
        # if np.random.rand() < self.mutate_rate:
        #     for i in range(len(g[0])):
        #         g[0][i] = g[0][i, ::-1, ::-1]
        # 撹拌
        # if np.random.rand() < self.mutate_rate:
        #     for i in range(len(g[0])):
        #         g[0][i] = np.random.permutation(g[0][i])
        #         g[0][i] = np.random.permutation(g[0][i].T)
        # 欠失
        # if np.random.rand() < self.mutate_rate:
        #     for i in range(len(g[0])):
        #         deletion_mask = np.ones(g[0][i].shape)
        #         deletion_mask[:int(len(g[0][i][0]) / 2), :int(len(g[0][i][0]) / 2)] = 0
        #         np.random.shuffle(deletion_mask)
        #         np.random.shuffle(deletion_mask.T)
        #         g[0][i] = g[0][i] * deletion_mask
        return g

    def next_generation(self):
        while len(self.family) < 2:
            self.add_new_population()

        p1, p2 = self.select_and_delete_gene(), self.select_and_delete_gene()

        c1, c2 = self.crossover(p1, p2)

        if np.random.rand() < 0.5:
            c1 = self.mutate(c1)
        else:
            c2 = self.mutate(c2)

        self.generation_num += 1

        genelist = p1, p2, c1, c2
        count = 0
        for i in genelist:
            # if i[1] is None:
            i[1] = self.evaluate_func(i[0], count, self.conv_num)
            count += 1

        # rule-1:both child is better than both parent, remain both child and better 1 parent
        if (self.better_high is True and min(c1[1], c2[1]) > max(p1[1], p2[1])) or (
                self.better_high is False and max(c1[1], c2[1]) < min(p1[1], p2[1])):
            self.family.append(c1)
            self.family.append(c2)
            if (self.better_high is True and p1[1] > p2[1]) or (self.better_high is False and p1[1] < p2[1]):
                self.family.append(p1)
            else:
                self.family.append(p2)

        # rule-2:both parent is better than both child, remain better 1 parent
        elif (self.better_high is True and max(c1[1], c2[1]) < min(p1[1], p2[1])) or (
                self.better_high is False and min(c1[1], c2[1]) > max(p1[1], p2[1])):
            if (self.better_high is True and p1[1] > p2[1]) or (self.better_high is False and p1[1] < p2[1]):
                self.family.append(p1)
            else:
                self.family.append(p2)

        # rule-3:better 1 parent is better than both child, remain better 1 parent and better 1 child
        elif (self.better_high is True and max(c1[1], c2[1]) < max(p1[1], p2[1])) or (
                self.better_high is False and min(c1[1], c2[1]) > min(p1[1], p2[1])):
            if (self.better_high is True and p1[1] > p2[1]) or (self.better_high is False and p1[1] < p2[1]):
                self.family.append(p1)
            else:
                self.family.append(p2)

            if (self.better_high is True and c1[1] > c2[1]) or (self.better_high is False and c1[1] < c2[1]):
                self.family.append(c1)
            else:
                self.family.append(c2)

        # rule-4:better 1 child is better than both parent, remain better 1 child and append 1 new gene
        else:
            if (self.better_high is True and c1[1] > c2[1]) or (self.better_high is False and c1[1] < c2[1]):
                self.family.append(c1)
            else:
                self.family.append(c2)
            self.add_new_population()
