import time
import pickle as pkl
import warnings
import copy
import os
import json
import utils
import model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import random

import numpy as np
import pandas as pd

from sklearn import metrics

# def train(dataloader):
#     net.train()
#     for batch, data in enumerate(dataloader):
#         x, y, z = data
#         batch_input1 = x[:, 0].to(device)
#         batch_input2 = x[:, 1].to(device)
#         batch_other = y.to(device)
#         batch_label = z.to(device)

#         # 前向传播，使用共享的 G
#         output = net(batch_input1, batch_input2, G, batch_other)
#         loss = criterion(output, batch_label.long())

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     torch.cuda.empty_cache()
def train(data_loader_generator):
    """
    模型训练：处理数据生成器
    """
    net.train()
    for dataloader in data_loader_generator:  # 遍历 loader 生成的 DataLoader
        for batch, data in enumerate(dataloader):  # 遍历 DataLoader 中的批次
            x, y, z, kgpt_embeddings = data
            batch_input1 = x[:, 0].to(device)
            batch_input2 = x[:, 1].to(device)
            batch_other = y.to(device)
            batch_label = z.to(device)
            batch_kgpt_embedding = kgpt_embeddings.to(device)  # KPGT 嵌入 [batch_size, 2304]

            # 前向传播
            output = net(batch_input1, batch_input2, G, batch_other, batch_kgpt_embedding)
            loss = criterion(output, batch_label.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.cuda.empty_cache()


@torch.no_grad()
def evaluate(data_loader_generator):
    net.eval()
    batch_loss, pairs_num = 0.0, 0
    label_list, class_list, score_list = [], [], []

    for dataloader in data_loader_generator:
        for batch, data in enumerate(dataloader):
            x, y, z, kgpt_embeddings = data
            batch_input1 = x[:, 0].to(device)
            batch_input2 = x[:, 1].to(device)
            batch_other = y.to(device)
            batch_label = z.to(device)
            batch_kgpt_embedding = kgpt_embeddings.to(device)  # KPGT 嵌入 [batch_size, 2304]
            label_list += list(batch_label.cpu().numpy())

            output = net(batch_input1, batch_input2, G, batch_other, batch_kgpt_embedding)
            loss = criterion(output, batch_label.long())
            batch_loss += loss.item() * len(batch_label)
            pairs_num += len(batch_label)

            out_score = F.softmax(output, dim=1)
            out_class = torch.argmax(out_score, 1)
            cpi_score = out_score[:, -1]
            class_list += list(out_class.cpu().numpy())
            score_list += list(cpi_score.cpu().numpy())

    indicator_list = [label_list, class_list, score_list]
    epoch_loss = batch_loss / pairs_num
    torch.cuda.empty_cache()
    return epoch_loss, indicator_list

@torch.no_grad()
def topk_acc(cp_df, kd_df, cell, label_dict1, label_dict2,  kgpt_embeddings_dict):
    net.eval()
    cp_num = cp_df.shape[1]
    kd_num = kd_df.shape[1]
    
    brd_list = [cp.split(':')[1] for cp in cp_df for j in range(kd_num)]
    gene_list = [kd.split(':')[1] for kd in kd_df] * cp_num
    
    cell_others = torch.tensor([[cell_lines_dict[cell]]]*cp_num*kd_num)
    #others = torch.tensor([[24,10,96]]*cp_num*kd_num)
    others = utils.get_extest_others(cp_df, kd_df)
    others = torch.cat((cell_others, others),dim=1)

    cp_data = torch.tensor(cp_df.values).T.to(device)
    kd_data = torch.tensor(kd_df.values).T.to(device)

    # 获取化合物的 KPGT 嵌入
    compound_ids = [cp.split(':')[1] for cp in cp_df.columns]
    kgpt_embeddings = torch.tensor([
        kgpt_embeddings_dict.get(cid, np.zeros(2304, dtype=np.float32)) for cid in compound_ids
    ], dtype=torch.float32).to(device)  # [cp_num, 2304]

    feature1 = net.inference1(cp_data.float(), G, kgpt_embeddings, 256)
    feature2 = net.inference11(kd_data.float(), G, 256) #什么意思？
    
    feature1 = feature1.repeat_interleave(kd_num, 0)
    feature2 = feature2.repeat(cp_num, 1)
    inputs = [feature1.float(), feature2.float(), others.float()]
    output = net.inference2(inputs, 2048)
    
    out_score = F.softmax(output, dim=1)
    cpi_class = list(torch.argmax(out_score,1).numpy())
    cpi_score = list(out_score[:,-1].numpy())
    score_matrix = (out_score[:,-1].view(cp_num, kd_num))    
    
    cp_label = torch.stack([label_dict1[cp.split(':')[1]] for cp in cp_df])
    kd_label = torch.stack([label_dict2[kd.split(':')[1]] for kd in kd_df])
    mask = torch.matmul(cp_label, kd_label.T)
    target = torch.where(mask > 0, torch.ones_like(mask), torch.zeros_like(mask))
    cpi_label = list(target.view(-1).numpy())
    
    b, q = target.size()
    pred = torch.argsort(score_matrix, 1, descending = True)
    pred = F.one_hot(pred[:, 0:top_k], num_classes = q).sum(-2)
    pred = (pred * target).sum(-1)
    preds = torch.where(pred > 0, torch.ones_like(pred), torch.zeros_like(pred))
    acck = torch.sum(preds, dtype = float).item() * 100.0 / b
    
    lists = [cpi_label, cpi_class, cpi_score, brd_list, gene_list]
    return acck, lists

# def loader(all_cp_df):
#     exps, labels, others = utils.get_pairs(neg_num, all_cp_df, all_kd_df, dti_dict1, cell_lines_dict)
#     print(exps.size(), others.size(), labels.size())
#     print('exps memory: {:04f} Gb'.format(exps.element_size() * exps.nelement() / 1024 / 1024 / 1024))
#     data_set = TensorDataset(exps.float(), labels, others.float())
#     data_loader = DataLoader(dataset=data_set, num_workers=3, batch_size=batch_size, shuffle=True, pin_memory=True)
#     return data_loader

# def loader(cp_df, batch_size):
#     """
#     使用数据生成器加载数据
#     """
#     generator = utils.data_generator(neg_num, cp_df, all_kd_df, dti_dict1, cell_lines_dict, batch_size)
#     for exps, others, labels in generator:
#         data_set = TensorDataset(exps, others, labels)
#         data_loader = DataLoader(dataset=data_set, num_workers=3, batch_size=batch_size, shuffle=True, pin_memory=True)
#         yield data_loader

def loader(save_dir):
    """
    从硬盘加载已保存的批次数据
    """
    batch_files = sorted([os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.startswith("batch_")])
    for batch_file in batch_files:
        exp_pairs, others, labels, kgpt_embeddings = torch.load(batch_file)
        dataset = TensorDataset(exp_pairs, others, labels, kgpt_embeddings)
        data_loader = DataLoader(dataset=dataset, num_workers=3, batch_size=batch_size, shuffle=True, pin_memory=True)
        yield data_loader


def indicator(indicator_list): #计算各类评价指标
    label_list, class_list, score_list = indicator_list
    acc = metrics.accuracy_score(label_list, class_list)
    precision = metrics.precision_score(label_list, class_list)
    recall = metrics.recall_score(label_list, class_list)
    f1 = metrics.f1_score(label_list, class_list)
    AUROC = metrics.roc_auc_score(label_list, score_list)
    AUPRC = metrics.average_precision_score(label_list, score_list)
    indicator_dict = {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'AUROC': AUROC, 'AUPRC': AUPRC}
    return acc, precision, recall, f1, AUROC, AUPRC

def weight_initialize(net):
    for layer in net.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode='fan_in')


config = {'seed': 888, 'neg_num': 3, 'batch_size': 1024, 'kgpt_embedding_dim':2304, 'combined_embedding_dim':100,
          'drop_out': 0.5, 'learning_rate': 1e-3, 'weight_decay': 1e-3, 'resume_training': False}
print(config)
seed = config['seed']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
utils.seed_torch(seed, device)
utils.setup_cpu(1)

neg_num = config['neg_num']
batch_size = config['batch_size']
kgpt_embedding_dim = config['kgpt_embedding_dim']
combined_embedding_dim = config['combined_embedding_dim']
drop_out = config['drop_out']
lr = config['learning_rate']
weight_decay = config['weight_decay']
resume_training = config['resume_training']
warnings.filterwarnings('ignore')
# all_available_cell_lines = ['A375', 'A549', 'HA1E', 'HCC515', 'HT29', 'MCF7', 'PC3', 'VCAP']
#cell_lines = ['PC3']
cell_lines = ['A375', 'A549', 'HA1E', 'HT29', 'MCF7', 'PC3']
cell_lines_dict = {'A375': 8, 'A549': 7, 'HA1E': 6, 'HCC515': 5, 'HT29': 4, 'MCF7': 3, 'PC3': 2, 'VCAP': 1}


#association_file = 'dataset/filter_godata/go_term_to_gene_ids.txt'
association_file = 'dataset2/filter_godata/go_term_to_gene_ids.txt'
H, catids, geneids = model.read_association_file(association_file)
G = model.generate_G_from_H(catids, H)
G = torch.Tensor(G).to(device)

net = model.Mymodel(device, drop_out=drop_out).to(device)
weight_initialize(net)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

train_id_file = 'dataset/train_id.txt' #cp数据
valid_id_file = 'dataset/valid_id.txt'
test_id_file = 'dataset/test_id.txt'
train_id_list = pd.read_csv(train_id_file, sep='\t', header=None)[0].tolist()
valid_id_list = pd.read_csv(valid_id_file, sep='\t', header=None)[0].tolist()
test_id_list = pd.read_csv(test_id_file, sep='\t', header=None)[0].tolist()

dti_file1 = 'dataset/phase1.dti.txt'
dti_file2 = 'dataset/phase2.dti.txt'
#dti_file3 = 'dataset/pabons.dti.txt'
dti_dict1 = utils.get_dti(dti_file1)
dti_dict2 = utils.get_dti(dti_file2)
#dti_dict3 = utils.get_dti(dti_file3)
kd_gene_file = 'dataset/kd_genes.txt'
kd_gene_list = pd.read_csv(kd_gene_file, sep='\t', header=None)[0].tolist()
cp_label_dict, kd_label_dict = utils.get_multi_hot_label(dti_dict1, dti_dict2, kd_gene_list)

all_kd_df, train_cp_df, valid_cp_df, test_cp_df = utils.get_state_df(cell_lines, train_id_list, valid_id_list,
                                                                        test_id_list)

embedding_file = 'dataset/kgpt_embeddings.tsv'
kgpt_embeddings_dict = utils.load_kpgt_embeddings(embedding_file)
embedding_file2 = 'dataset/kgpt_embeddings2.tsv'
kgpt_embeddings_dict2 = utils.load_kpgt_embeddings(embedding_file2)

##data loading
print('data generating...................')
start_time = time.time()
train_save_dir = "train_batches"
valid_save_dir = "valid_batches"
test_save_dir = "test_batches"
# utils.data_generator(neg_num, train_cp_df, all_kd_df, dti_dict1, cell_lines_dict, train_save_dir, kgpt_embeddings_dict, batch_size=1024)
# utils.data_generator(neg_num, valid_cp_df, all_kd_df, dti_dict1, cell_lines_dict, valid_save_dir, kgpt_embeddings_dict, batch_size=1024)
# utils.data_generator(neg_num, test_cp_df, all_kd_df, dti_dict1, cell_lines_dict, test_save_dir, kgpt_embeddings_dict, batch_size=1024)
end_time = time.time()
print('time for data generating: {:.5f}'.format(end_time - start_time))

# train_dataloader = loader(train_cp_df, batch_size)
# valid_dataloader = loader(valid_cp_df, batch_size)
# test_dataloader = loader(test_cp_df, batch_size)
# with open('pickle/train_dataloader.pickle', 'wb') as train_data_file:
#     pkl.dump(train_dataloader, train_data_file)
# with open('pickle/valid_dataloader.pickle', 'wb') as valid_data_file:
#     pkl.dump(valid_dataloader, valid_data_file)
# with open('pickle/test_dataloader.pickle', 'wb') as test_data_file:
#     pkl.dump(test_dataloader, test_data_file)

#resume_training = resume_training  # 设置为 True 表示从断点恢复
save_path = "latest_checkpoint.pt"
if resume_training and os.path.exists(save_path):
    # 从检查点加载模型和训练状态
    start_epoch, checkpoint = utils.load_checkpoint(net, optimizer, save_path)
    
    # 恢复历史记录
    train_loss_history = checkpoint.get('train_loss_history', [])
    valid_loss_history = checkpoint.get('valid_loss_history', [])
    test_loss_history = checkpoint.get('test_loss_history', [])
    train_F1_history = checkpoint.get('train_F1_history', [])
    valid_F1_history = checkpoint.get('valid_F1_history', [])
    test_F1_history = checkpoint.get('test_F1_history', [])
    train_AUROC_history = checkpoint.get('train_AUROC_history', [])
    valid_AUROC_history = checkpoint.get('valid_AUROC_history', [])
    test_AUROC_history = checkpoint.get('test_AUROC_history', [])
    train_AUPRC_history = checkpoint.get('train_AUPRC_history', [])
    valid_AUPRC_history = checkpoint.get('valid_AUPRC_history', [])
    test_AUPRC_history = checkpoint.get('test_AUPRC_history', [])
    best_AUPRC = checkpoint.get('best_AUPRC', 0.0)
    all_time = checkpoint.get('all_time', 0.0)
    
    print(f"Resuming training from epoch {start_epoch}")
else:
    start_epoch = 1
    train_loss_history, valid_loss_history, test_loss_history = [], [], []
    train_F1_history, valid_F1_history, test_F1_history = [], [], []
    train_AUROC_history, valid_AUROC_history, test_AUROC_history =[], [], []
    train_AUPRC_history, valid_AUPRC_history, test_AUPRC_history = [], [], []
    all_time, best_AUPRC = 0.0, 0.0
    print("Starting new training session from scratch.")


for epoch in range(start_epoch, 300):
    start_time = time.time()
    print('#########', '\n', 'Epoch {:04d}'.format(epoch))
    train_dataloader = loader(train_save_dir)
    train(train_dataloader)

    train_dataloader = loader(train_save_dir)
    loss, indicator_list = evaluate(train_dataloader)
    train_loss_history.append(loss)
    acc, precision, recall, f1, AUROC, AUPRC = indicator(indicator_list)
    train_F1_history.append(f1)
    train_AUROC_history.append(AUROC)
    train_AUPRC_history.append(AUPRC)
    print(
        'Tra: loss: {:.4f} acc: {:.4f} precision: {:.4f} recall: {:.4f} f1: {:.4f} AUROC: {:.4f} AUPRC: {:.4f}'.format(
            loss, acc, precision, recall, f1, AUROC, AUPRC))

    valid_dataloader = loader(valid_save_dir)
    loss, indicator_list = evaluate(valid_dataloader)    
    valid_loss_history.append(loss)
    acc, precision, recall, f1, AUROC, AUPRC = indicator(indicator_list)
    valid_F1_history.append(f1)
    valid_AUROC_history.append(AUROC)
    valid_AUPRC_history.append(AUPRC)
    print(
        'Val: loss: {:.4f} acc: {:.4f} precision: {:.4f} recall: {:.4f} f1: {:.4f} AUROC: {:.4f} AUPRC: {:.4f}'.format(
            loss, acc, precision, recall, f1, AUROC, AUPRC))

    # copy best valid_acck model
    if AUPRC > best_AUPRC:
        best_epoch = epoch
        best_AUPRC = AUPRC
        best_model = copy.deepcopy(net)
        best_optimizer = copy.deepcopy(optimizer)

    test_dataloader = loader(test_save_dir)
    loss, indicator_list = evaluate(test_dataloader)
    test_loss_history.append(loss)
    acc, precision, recall, f1, AUROC, AUPRC = indicator(indicator_list)
    test_F1_history.append(f1)
    test_AUROC_history.append(AUROC)
    test_AUPRC_history.append(AUPRC)
    print(
        'Tes: loss: {:.4f} acc: {:.4f} precision: {:.4f} recall: {:.4f} f1: {:.4f} AUROC: {:.4f} AUPRC: {:.4f}'.format(
            loss, acc, precision, recall, f1, AUROC, AUPRC))
    
    # 保存检查点
    utils.save_checkpoint(
        epoch=epoch,
        model=net,
        optimizer=optimizer,
        save_path=save_path,
        train_loss_history=train_loss_history,
        valid_loss_history=valid_loss_history,
        test_loss_history=test_loss_history,
        train_F1_history=train_F1_history,
        valid_F1_history=valid_F1_history,
        test_F1_history=test_F1_history,
        train_AUROC_history=train_AUROC_history,
        valid_AUROC_history=valid_AUROC_history,
        test_AUROC_history=test_AUROC_history,
        train_AUPRC_history=train_AUPRC_history,
        valid_AUPRC_history=valid_AUPRC_history,
        test_AUPRC_history=test_AUPRC_history,
        best_AUPRC=best_AUPRC
    )

    # Early stop
    if epoch >= 200 and valid_AUPRC_history[-1] <= np.mean(valid_AUPRC_history[-21:-1]): #300个epoch后，若验证集AUPRC低于前20个epoch的平均值，则触发早停
        print('#########', '\n', 'Early stopping...')
        break

    end_time = time.time()
    all_time += end_time - start_time
    print('epoch_time = {:.5f}'.format(end_time - start_time))
print('Optimization Finished! Stop at epoch:', epoch, 'time= {:.5f}'.format(all_time))

print('###save_model###')
state = {'net': best_model.state_dict(), 'optimizer': best_optimizer.state_dict(), 'epoch': best_epoch}
file_name = 'best_epoch_{}_bs_{}_lr_{}_wd_{}.mymodel.pth'.format(best_epoch, batch_size, lr, weight_decay)
torch.save(state, f'./saved_model/{file_name}')
print(file_name)

net = model.Mymodel(device, drop_out=drop_out).to(device) #重新初始化模型
file = f'saved_model/{file_name}' #加载保存的最佳模型
net.load_state_dict(torch.load(file)['net'],strict=False)
#net.load_state_dict(torch.load(file, map_location=device)['net'],strict=False)

# print('#########Topk=30#########')
# top_k = 30
# print('#########Topk acc test in phase1') #phase1测试集上的效果
# for cell in ['A375','A549','HA1E','HT29','MCF7','PC3']:
#     cell_kd_df, cell_cp_df = utils.get_acck_df2(cell)
#     test_acck1, lists = topk_acc(cell_cp_df, cell_kd_df, cell, cp_label_dict, kd_label_dict, kgpt_embeddings_dict)
#     print(cell, test_acck1)

# print('#########Topk acc test in phase2') #phase2外部测试集上的效果
# for cell in ['A375','A549','HA1E','HT29','MCF7','PC3']: #没有'VCAP'数据
#     cell_kd_df, cell_cp_df = utils.get_acck_df(cell)
#     test_acck1, lists = topk_acc(cell_cp_df, cell_kd_df, cell, cp_label_dict, kd_label_dict, kgpt_embeddings_dict2)
#     print(cell, test_acck1)

# print('#########Topk=50#########')
# top_k = 50
# print('#########Topk acc test in phase1') #phase1测试集上的效果
# for cell in ['A375','A549','HA1E','HT29','MCF7','PC3']:
#     cell_kd_df, cell_cp_df = utils.get_acck_df2(cell)
#     test_acck1, lists = topk_acc(cell_cp_df, cell_kd_df, cell, cp_label_dict, kd_label_dict, kgpt_embeddings_dict)
#     print(cell, test_acck1)

# print('#########Topk acc test in phase2') #phase2外部测试集上的效果
# for cell in ['A375','A549','HA1E','HT29','MCF7','PC3']: #没有'VCAP'数据
#     cell_kd_df, cell_cp_df = utils.get_acck_df(cell)
#     test_acck1, lists = topk_acc(cell_cp_df, cell_kd_df, cell, cp_label_dict, kd_label_dict, kgpt_embeddings_dict2)
#     print(cell, test_acck1)

print('#########Topk=100#########')
top_k = 100
print('#########Topk acc test in phase1') #phase1测试集上的效果
for cell in ['A375','A549','HA1E','HT29','MCF7','PC3']:
    cell_kd_df, cell_cp_df = utils.get_acck_df2(cell)
    test_acck1, lists = topk_acc(cell_cp_df, cell_kd_df, cell, cp_label_dict, kd_label_dict, kgpt_embeddings_dict)
    print(cell, test_acck1)


print('#########Topk acc test in phase2') #phase2外部测试集上的效果
for cell in ['A375','A549','HA1E','HT29','MCF7','PC3']: #没有'VCAP'数据
    cell_kd_df, cell_cp_df = utils.get_acck_df(cell)
    test_acck1, lists = topk_acc(cell_cp_df, cell_kd_df, cell, cp_label_dict, kd_label_dict, kgpt_embeddings_dict2)
    print(cell, test_acck1)