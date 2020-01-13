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

data_dict = {'val_loss': [], 'val_acc': []}
# 枝刈り前パラメータ利用
original_net = parameter_use('./result/pkl/original_train_epoch50.pkl')
# 枝刈り前畳み込み層のリスト
original_conv_list = [module for module in original_net.modules() if isinstance(module, nn.Conv2d)]
# 枝刈り後パラメータ利用
new_net = parameter_use('./result/pkl/dense_conv_prune_dense90per_conv80per.pkl')
# 枝刈り後畳み込み層・全結合層・係数パラメータのリスト
conv_list = [module for module in new_net.modules() if isinstance(module, nn.Conv2d)]
dense_list = [module for module in new_net.modules() if isinstance(module, nn.Linear)]
param_list = [module for module in new_net.modules() if isinstance(module, nn.Parameter)]
# マスクのオブジェクト
ch_mask = [ChannelMaskGenerator() for _ in range(len(conv_list))]
for i, conv in enumerate(conv_list):
    ch_mask[i].mask = np.where(np.abs(conv.weight.data.clone().cpu().detach().numpy()) == 0, 0, 1)
de_mask = [DenseMaskGenerator() for _ in range(len(dense_list))]
for i, dense in enumerate(dense_list):
    de_mask[i].mask = np.where(np.abs(dense.weight.data.clone().cpu().detach().numpy()) == 0, 0, 1)

optimizer = optim.SGD(new_net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

for param in new_net.parameters():
    param.requires_grad = False
for dense in dense_list:
    dense.weight.requires_grad = True
for param in param_list:
    param.requires_grad = True

f_num_epochs = 5
for epoch in range(f_num_epochs):
    # train
    new_net.train()
    train_loss, train_acc = 0, 0
    for _, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = new_net(images, True)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        train_acc += (outputs.max(1)[1] == labels).sum().item()
        loss.backward()
        optimizer.step()
        for param in param_list:
            param_max, param_min = torch.max(param), torch.min(param)
            param = 2 * (param - param_min) / (param_max - param_min)
        with torch.no_grad():
            for j, dense in enumerate(dense_list):
                if de_mask[j].mask is None:
                    break
                dense.weight.data *= torch.tensor(de_mask[j].mask, device=device, dtype=dtype)
    avg_train_loss, avg_train_acc = train_loss / len(train_loader.dataset), train_acc / len(train_loader.dataset)

    # val
    new_net.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.to(device)
            outputs = new_net(images.to(device), True)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += (outputs.max(1)[1] == labels).sum().item()
    avg_val_loss, avg_val_acc = val_loss / len(test_loader.dataset), val_acc / len(test_loader.dataset)

    print(f'finetune, epoch [{epoch + 1}/{f_num_epochs}], train_loss: {avg_train_loss:.4f}'
          f', train_acc: {avg_train_acc:.4f}, val_loss: {avg_val_loss:.4f}, val_acc: {avg_val_acc:.4f}')
    print()

    # 結果の保存
    input_data = [avg_val_loss, avg_val_acc]
    result_save('./result/pkl/result_dense_retrain_dense90per_conv80per.csv', data_dict, input_data)
