from channel_mask_generator import ChannelMaskGenerator
from dataset import *
from result_save_visualization import *
from channel_importance import channel_euclidean_distance, cos_sim
import torch
import numpy as np

data_dict = {'attribute': [], 'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
dense_per = 0
conv_per = 60
csv = 1
pkl = 1
coef = 2500

# 枝刈り前パラメータ利用
original_net = parameter_use(f'./result/pkl{pkl}/original_train_epoch50.pkl')
# 畳み込み層のリスト
original_conv_list = [module for module in original_net.modules() if isinstance(module, nn.Conv2d)]


class CnnEvaluatePrune:
    def __init__(self, count):
        self.network = None
        self.count = count

    def evaluate(self, gene, count, conv_num):
        return self.train(gene, count, conv_num)

    def train(self, gene, g_count, conv_num):
        # 枝刈り後パラメータ利用
        self.network = parameter_use(f'./result3/pkl{pkl}_/dense_conv_prune_dense{dense_per}per_conv{conv_per}per.pkl')

        # 畳み込み層のリスト
        conv_list = [module for module in self.network.modules() if isinstance(module, nn.Conv2d)]
        # 全結合層のリスト
        dense_list = [module for module in self.network.modules() if isinstance(module, nn.Linear)]

        # 畳み込み層マスクのオブジェクト
        ch_mask = [ChannelMaskGenerator() for _ in range(len(conv_list))]
        for i, conv in enumerate(conv_list):
            ch_mask[i].mask = np.where(np.abs(conv.weight.data.clone().cpu().detach().numpy()) == 0, 0, 1)

        # 追加
        with torch.no_grad():
            if conv_num == len(conv_list) - 1:
                gene_fil = gene.reshape(conv_list[conv_num].weight.data[0, :, :, :].cpu().numpy().shape)
                for j in range(len(conv_list[conv_num].weight.data.cpu().numpy())):
                    if np.sum(np.abs(ch_mask[conv_num].mask[j])) < 25 * (self.count + 1) + 1:
                        ch_mask[conv_num].mask[j] = 1
                        conv_list[conv_num].weight.data[j] = torch.tensor(gene_fil, device=device, dtype=dtype)
                        print(f'add_filter_conv{conv_num + 1}')
                        break
            else:
                gene_fil_vec = gene[:len(conv_list[conv_num].weight.data[0, :, :, :].cpu().numpy().flatten())]
                gene_ker_vec = gene[len(conv_list[conv_num].weight.data[0, :, :, :].cpu().numpy().flatten()):]
                gene_fil = gene_fil_vec.reshape(conv_list[conv_num].weight.data[0, :, :, :].cpu().numpy().shape)
                gene_ker = gene_ker_vec.reshape(conv_list[conv_num+1].weight.data[:, 0, :, :].cpu().numpy().shape)
                for j in range(len(conv_list[conv_num].weight.data.cpu().numpy())):
                    if (conv_num == 0 and np.sum(np.abs(ch_mask[conv_num].mask[j])) < 1) \
                            or (conv_num > 0 and np.sum(np.abs(ch_mask[conv_num].mask[j])) < 25 * (self.count + 1) + 1):
                        ch_mask[conv_num].mask[j] = 1
                        conv_list[conv_num].weight.data[j] = torch.tensor(gene_fil, device=device, dtype=dtype)
                        ch_mask[conv_num + 1].mask[j, :] = 1
                        conv_list[conv_num + 1].weight.data[:, j] = torch.tensor(gene_ker, device=device, dtype=dtype)
                        print(f'add_filter_conv{conv_num + 1}')
                        break

        # パラメータの割合
        weight_ratio = [np.count_nonzero(conv.weight.cpu().detach().numpy()) /
                        np.size(conv.weight.cpu().detach().numpy()) for conv in conv_list]

        # 枝刈り後のチャネル数
        channel_num_new = [conv_list[i].out_channels - ch_mask[i].channel_number(conv.weight) for i, conv in
                           enumerate(conv_list)]

        print(f'parent{g_count + 1}: ') if g_count < 2 else print(f'children{g_count - 1}: ')
        for i in range(len(conv_list)):
            print(f'conv{i + 1}_param: {weight_ratio[i]:.4f}', end=", " if i != len(conv_list) - 1 else "\n")
        for i in range(len(conv_list)):
            print(f'channel_number{i + 1}: {channel_num_new[i]}', end=", " if i != len(conv_list) - 1 else "\n")

        similarity = 0
        # チャネル間の類似度
        for i in range(conv_list[conv_num].out_channels):
            if conv_num == len(conv_list) - 1:
                gene_fil = gene.reshape(conv_list[conv_num].weight.data[0, :, :, :].cpu().numpy().shape)
            else:
                gene_fil_vec = gene[:len(conv_list[conv_num].weight.data[0, :, :, :].cpu().numpy().flatten())]
                gene_fil = gene_fil_vec.reshape(conv_list[conv_num].weight.data[0, :, :, :].cpu().numpy().shape)
            similarity += channel_euclidean_distance(gene_fil, conv_list[conv_num].weight.data.cpu().detach().numpy()[i])
            # similarity += cos_sim(gene, conv_list[conv_num].weight.data.cpu().detach().numpy()[i])

        f_num_epochs = 1
        eva = 0
        avg_train_loss, avg_train_acc = 0, 0

        for epoch in range(f_num_epochs):
            # val
            self.network.eval()
            val_loss, val_acc = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    labels = labels.to(device)
                    outputs = self.network(images.to(device), False)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_acc += (outputs.max(1)[1] == labels).sum().item()
            avg_val_loss, avg_val_acc = val_loss / len(test_loader.dataset), val_acc / len(test_loader.dataset)
            eva = avg_val_loss

            print(f'epoch [{epoch + 1}/{f_num_epochs}], train_loss: {avg_train_loss:.4f}'
                  f', train_acc: {avg_train_acc:.4f}, val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')
            exit()

            # 結果の保存
            input_data = [g_count, epoch + 1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc]
            result_save(f'./result3/csv{csv}/add_channels_train_dense{dense_per}per_conv{conv_per}per.csv', data_dict, input_data)

        return coef * eva + similarity
        # return eva
