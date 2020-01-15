import os
import sys
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from channel_mask_generator import ChannelMaskGenerator
from dense_mask_generator import DenseMaskGenerator
from dataset import *
from result_save_visualization import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

data_dict = {'epoch': [], 'time': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

# パラメータ利用, 全結合パラメータの凍結
new_net = parameter_use('./result2/pkl1/dense_prune_90per.pkl')
# 全結合層、畳み込み層のリスト
dense_list = [module for module in new_net.modules() if isinstance(module, nn.Linear)]
conv_list = [module for module in new_net.modules() if isinstance(module, nn.Conv2d)]
conv_count = len(conv_list)
# for dense in dense_list:
#     dense.weight.requires_grad = False

optimizer = optim.SGD(new_net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# マスクのオブジェクト
ch_mask = [ChannelMaskGenerator() for _ in range(conv_count)]
de_mask = [DenseMaskGenerator() for _ in dense_list]
for i, dense in enumerate(dense_list):
    de_mask[i].mask = np.where(np.abs(dense.weight.data.clone().cpu().detach().numpy()) == 0, 0, 1)
inv_prune_ratio = 10

# channel_pruning
for count in range(1, inv_prune_ratio + 9):
    print(f'\nchannel pruning: {count}')
    # ノルムの合計を保持
    channel_l1norm_for_each_layer = [list() for _ in range(conv_count)]

    # ノルムの取得, 昇順にソート
    for i, conv in enumerate(conv_list):
        channel_l1norm_for_each_layer[i] = [np.sum(torch.abs(param).cpu().detach().numpy()) for param in conv.weight]
        channel_l1norm_for_each_layer[i].sort()

    # 枝刈り本体
    with torch.no_grad():
        for i in range(len(conv_list)):
            threshold = channel_l1norm_for_each_layer[i][int(conv_list[i].out_channels / inv_prune_ratio * count)]\
                if count <= 9 else channel_l1norm_for_each_layer[i][int(conv_list[i].out_channels *
                                                        (9 / inv_prune_ratio + (count - 9) / inv_prune_ratio ** 2))]
            save_mask = ch_mask[i].generate_mask(conv_list[i].weight.data.clone(),
                                                 None if i == 0 else conv_list[i - 1].weight.data.clone(), threshold)
            conv_list[i].weight.data *= torch.tensor(save_mask, device=device, dtype=dtype)

    # パラメータの割合
    weight_ratio = [np.count_nonzero(conv.weight.cpu().detach().numpy()) / np.size(conv.weight.cpu().detach().numpy())
                    for conv in conv_list]

    # 枝刈り後のチャネル数
    channel_num_new = [conv.out_channels - ch_mask[i].channel_number(conv.weight) for i, conv in enumerate(conv_list)]

    for i in range(conv_count):
        print(f'conv{i + 1}_param: {weight_ratio[i]:.4f}', end=", " if i != conv_count - 1 else "\n")
    for i in range(conv_count):
        print(f'channel_number{i + 1}: {channel_num_new[i]}', end=", " if i != conv_count - 1 else "\n")

    accuracy = 0
    f_num_epochs = 10
    # finetune
    start = time.time()
    for epoch in range(f_num_epochs):
        # train
        new_net.train()
        train_loss, train_acc = 0, 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = new_net(images, False)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            train_acc += (outputs.max(1)[1] == labels).sum().item()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for j, conv in enumerate(conv_list):
                    conv.weight.data *= torch.tensor(ch_mask[j].mask, device=device, dtype=dtype)
                for k, dense in enumerate(dense_list):
                    dense.weight.data *= torch.tensor(de_mask[k].mask, device=device, dtype=dtype)
        avg_train_loss, avg_train_acc = train_loss / len(train_loader.dataset), train_acc / len(train_loader.dataset)

        # val
        new_net.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                labels = labels.to(device)
                outputs = new_net(images.to(device), False)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.max(1)[1] == labels).sum().item()
        avg_val_loss, avg_val_acc = val_loss / len(test_loader.dataset), val_acc / len(test_loader.dataset)
        process_time = time.time() - start
        accuracy = avg_val_acc

        print(f'epoch [{epoch + 1}/{f_num_epochs}], time: {process_time:.4f}, train_loss: {avg_train_loss:.4f}'
              f', train_acc: {avg_train_acc:.4f}, 'f'val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')

        # 結果の保存
        input_data = [epoch + 1, process_time, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc]
        result_save('./result2/csv1/dense_conv_prune_parameter_dense90per.csv', data_dict, input_data)

    # パラメータの保存
    if count == 8:
        parameter_save('./result2/pkl1/dense_conv_prune_dense90per_conv80per.pkl', new_net)
        parameter_save('./result2/pkl1/dense_conv_prune_dense90per_conv80per_copy.pkl', new_net)
    elif count == 9:
        parameter_save('./result2/pkl1/dense_conv_prune_dense90per_conv90per.pkl', new_net)
        parameter_save('./result2/pkl1/dense_conv_prune_dense90per_conv90per_copy.pkl', new_net)
    elif count == 14:
        parameter_save('./result2/pkl1/dense_conv_prune_dense90per_conv95per.pkl', new_net)
        parameter_save('./result2/pkl1/dense_conv_prune_dense90per_conv95per_copy.pkl', new_net)
        break
    else:
        pass
