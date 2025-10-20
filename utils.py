import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules.module import Module

# set random seed
def seed_torch(seed, device=None):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 仅为当前设备设置种子
    if device:
        torch.cuda.set_device(device)  # 确保使用指定的设备
        torch.cuda.manual_seed(seed)  # 为指定的 GPU 设备设置种子
    else:
        torch.cuda.manual_seed_all(seed)  # 仅在需要多 GPU 时使用
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def setup_cpu(cpu_num):
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

def save_checkpoint(epoch, model, optimizer, save_path, **kwargs):
    """
    保存模型检查点
    Args:
        epoch: 当前的 epoch 编号
        model: 模型对象
        optimizer: 优化器对象
        save_path: 保存路径
        **kwargs: 其他需要保存的内容
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparameters': {  # 保存超参数
            'lr': optimizer.param_groups[0]['lr'],  # 学习率
            'weight_decay': optimizer.param_groups[0].get('weight_decay', 0.0),
        },
        **kwargs  # 包括传入的其他内容，如历史记录
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at epoch {epoch} to {save_path}")

def load_checkpoint(model, optimizer, load_path, expected_hyperparameters=None):
    """
    加载模型检查点
    Args:
        model: 模型对象
        optimizer: 优化器对象
        load_path: 加载路径
        expected_hyperparameters: 预期的超参数字典，用于检查一致性
    Returns:
        start_epoch: 下一次训练的起始 epoch
        checkpoint: 加载的检查点字典，包含额外的历史记录
    """
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
        # 检查超参数是否一致
        if expected_hyperparameters:
            saved_hyperparameters = checkpoint.get('hyperparameters', {})
            for key, value in expected_hyperparameters.items():
                if saved_hyperparameters.get(key) != value:
                    print(f"Warning: Hyperparameter {key} mismatch! "
                          f"Saved: {saved_hyperparameters.get(key)}, Expected: {value}")
        
        print(f"Checkpoint loaded, resuming from epoch {start_epoch}")
        return start_epoch, checkpoint
    else:
        print("No checkpoint found, starting from scratch.")
        return 1, {}


def get_dti(dti_file):
    cp_gene_dict = {}
    with open(dti_file, 'r') as f:
        for line in f.readlines():
            hang = line.strip().split('\t')
            genelist = hang[1].split(';')
            cp_gene_dict[hang[0]] = genelist
    return cp_gene_dict


def get_multi_hot_label(dti_dict1, dti_dict2, kd_list):
    dti_dict1.update(dti_dict2)
    #dti_dict1.update(dti_dict3)
    cp_label_dict, kd_label_dict = {}, {}
    for kd in kd_list:
        kd_label_dict[kd] = F.one_hot(torch.tensor(kd_list.index(kd)), num_classes=len(kd_list))
    for cp in dti_dict1:
        target_list = [n for n in dti_dict1[cp] if n in kd_list]
        if target_list == []:
            cp_label_dict[cp] = torch.tensor([0] * len(kd_list))
        else:
            index_list = [kd_list.index(n) for n in target_list]
            cp_label_dict[cp] = F.one_hot(torch.tensor(index_list), num_classes=len(kd_list)).sum(-2)
    return cp_label_dict, kd_label_dict

def get_acck_df2(cell): #phase1测试集
    kd_file = 'dataset2/kd_data/' + 'trt_sh_'+ cell + '_core_signatures.tsv'
    kd_df = pd.read_csv(kd_file, sep='\t').drop(columns='rid')

    cp_file = 'dataset2/test_cpdata/' + 'trt_cp_'+ cell + '_filtered_test_data.tsv'
    cp_df = pd.read_csv(cp_file, sep='\t').drop(columns='rid')
    return kd_df, cp_df

def get_acck_df_targetcold(cell): #phase1测试集
    kd_file = 'dataset2/test_kddata_targetcold/' + 'trt_sh_'+ cell + '_core_signatures.tsv'
    kd_df = pd.read_csv(kd_file, sep='\t').drop(columns='rid')

    cp_file = 'dataset2/test_cpdata_targetcold/' + 'trt_cp_'+ cell + '_filtered_test_data.tsv'
    cp_df = pd.read_csv(cp_file, sep='\t').drop(columns='rid')
    return kd_df, cp_df

def get_acck_df_warm(cell): #phase1测试集
    kd_file = 'dataset2/test_kddata_warm/' + 'trt_sh_'+ cell + '_core_signatures.tsv'
    kd_df = pd.read_csv(kd_file, sep='\t').drop(columns='rid')

    cp_file = 'dataset2/test_cpdata_warm/' + 'trt_cp_'+ cell + '_filtered_test_data.tsv'
    cp_df = pd.read_csv(cp_file, sep='\t').drop(columns='rid')
    return kd_df, cp_df

def get_acck_df(cell): #phase2测试集
    kd_file = 'dataset2/kd_data/' + 'trt_sh_'+ cell + '_core_signatures.tsv'
    kd_df = pd.read_csv(kd_file, sep='\t').drop(columns='rid')

    cp_file = 'dataset2/external_data/' + 'trt_cp_'+ cell + '_test_set.tsv'
    cp_df = pd.read_csv(cp_file, sep='\t').drop(columns='rid')
    return kd_df, cp_df

def get_acck_experiment(cell): #根据靶标预测药物实验
    #kd_file = 'experiment/kd_data/' + 'trt_sh_'+ cell + '_TGFBR1.tsv'
    #kd_file = 'experiment/kd_data/' + 'trt_sh_'+ cell + '_EGFR.tsv'
    kd_file = 'experiment_ac/' + 'trt_oe_'+ cell + '_ESR1.tsv'
    kd_df = pd.read_csv(kd_file, sep='\t').drop(columns='rid')

    cp_file = 'experiment/cp_data/' + 'trt_cp_'+ cell + '_expression_data_cleaned.tsv'
    cp_df = pd.read_csv(cp_file, sep='\t').drop(columns='rid')
    return kd_df, cp_df

def get_acck_df_ac2(cell): #激活实验-phase1测试集
    kd_file = 'dataset2/oe_data/' + 'trt_oe_'+ cell + '_core_signatures.tsv'
    kd_df = pd.read_csv(kd_file, sep='\t').drop(columns='rid')

    cp_file = 'dataset2/test_cpdata_ac/' + 'trt_cp_'+ cell + '_filtered_test_data.tsv'
    cp_df = pd.read_csv(cp_file, sep='\t').drop(columns='rid')
    return kd_df, cp_df

# def get_acck_df_ac(cell): #激活实验-phase2测试集
#     kd_file = 'dataset2/oe_data/' + 'trt_oe_'+ cell + '_core_signatures.tsv'
#     kd_df = pd.read_csv(kd_file, sep='\t').drop(columns='rid')

#     cp_file = 'dataset2/external_data/' + 'trt_cp_'+ cell + '_test_set.tsv'
#     cp_df = pd.read_csv(cp_file, sep='\t').drop(columns='rid')
#     return kd_df, cp_df


def get_state_df(cell_lines, train_id_list, valid_id_list, test_id_list):
    all_kd_df = pd.DataFrame({})
    train_cp_df = pd.DataFrame({})
    valid_cp_df = pd.DataFrame({})
    test_cp_df = pd.DataFrame({})
    for cell in cell_lines:
        kd_file = 'dataset2/kd_data/' + 'trt_sh_'+cell + '_core_signatures.tsv'
        cell_kd_df = pd.read_csv(kd_file, sep='\t').drop(columns='rid')
        print(cell, 'kd_num:', cell_kd_df.shape[1])
        all_kd_df = pd.concat([all_kd_df, cell_kd_df], axis=1)

        cp_file = 'dataset2/cp_data/' + 'trt_cp_'+cell + '_filtered_expression_data.tsv'
        cell_cp_df = pd.read_csv(cp_file, sep='\t').drop(columns='rid')

        for brdid in train_id_list:
            brd_df = cell_cp_df.filter(like=brdid, axis=1)
            train_cp_df = pd.concat([train_cp_df, brd_df], axis=1)
        for brdid in valid_id_list:
            brd_df = cell_cp_df.filter(like=brdid, axis=1)
            valid_cp_df = pd.concat([valid_cp_df, brd_df], axis=1)
        for brdid in test_id_list:
            brd_df = cell_cp_df.filter(like=brdid, axis=1)
            test_cp_df = pd.concat([test_cp_df, brd_df], axis=1)
        print(cell, 'train_cp_num:', train_cp_df.shape[1])
        print(cell, 'valid_cp_num:', valid_cp_df.shape[1])
        print(cell, 'test_cp_num:', test_cp_df.shape[1])
    return all_kd_df, train_cp_df, valid_cp_df, test_cp_df

def get_state_df_targetcold(cell_lines, train_id_list, valid_id_list, test_id_list): # cp 和 kd 交换
    all_cp_df = pd.DataFrame({})
    train_kd_df = pd.DataFrame({})
    valid_kd_df = pd.DataFrame({})
    test_kd_df = pd.DataFrame({})

    for cell in cell_lines:
        cp_file = 'dataset2/cp_data/' + 'trt_cp_'+cell + '_filtered_expression_data.tsv'
        cell_cp_df = pd.read_csv(cp_file, sep='\t').drop(columns='rid')
        print(cell, 'cp_num:', cell_cp_df.shape[1])
        all_cp_df = pd.concat([all_cp_df, cell_cp_df], axis=1)

        kd_file = 'dataset2/kd_data/' + 'trt_sh_'+cell + '_core_signatures.tsv'
        cell_kd_df = pd.read_csv(kd_file, sep='\t').drop(columns='rid')

        # 合并数据时去除重复列
        for brdid in train_id_list:
            brd_df = cell_kd_df.filter(like=brdid, axis=1)
            brd_df = brd_df.loc[:, ~brd_df.columns.duplicated()]  # 去除重复列
            train_kd_df = pd.concat([train_kd_df, brd_df], axis=1)

        for brdid in valid_id_list:
            brd_df = cell_kd_df.filter(like=brdid, axis=1)
            brd_df = brd_df.loc[:, ~brd_df.columns.duplicated()]  # 去除重复列
            valid_kd_df = pd.concat([valid_kd_df, brd_df], axis=1)

        for brdid in test_id_list:
            brd_df = cell_kd_df.filter(like=brdid, axis=1)
            brd_df = brd_df.loc[:, ~brd_df.columns.duplicated()]  # 去除重复列
            test_kd_df = pd.concat([test_kd_df, brd_df], axis=1)

        print(cell, 'train_kd_num:', train_kd_df.shape[1])
        print(cell, 'valid_kd_num:', valid_kd_df.shape[1])
        print(cell, 'test_kd_num:', test_kd_df.shape[1])

    # 在拼接后的数据中去除重复列
    train_kd_df = train_kd_df.loc[:, ~train_kd_df.columns.duplicated()]
    valid_kd_df = valid_kd_df.loc[:, ~valid_kd_df.columns.duplicated()]
    test_kd_df = test_kd_df.loc[:, ~test_kd_df.columns.duplicated()]

    return all_cp_df, train_kd_df, valid_kd_df, test_kd_df

def get_state_df_warm(cell_lines, drug_train_id_list, drug_valid_id_list, drug_test_id_list, target_train_id_list, target_valid_id_list, target_test_id_list): # cp 和 kd 交换
    train_cp_df = pd.DataFrame({})
    valid_cp_df = pd.DataFrame({})
    test_cp_df = pd.DataFrame({})
    train_kd_df = pd.DataFrame({})
    valid_kd_df = pd.DataFrame({})
    test_kd_df = pd.DataFrame({})

    for cell in cell_lines:
        cp_file = 'dataset2/cp_data/' + 'trt_cp_'+cell + '_filtered_expression_data.tsv'
        cell_cp_df = pd.read_csv(cp_file, sep='\t').drop(columns='rid')
        for brdid in drug_train_id_list:
            brd_df = cell_cp_df.filter(like=brdid, axis=1)
            train_cp_df = pd.concat([train_cp_df, brd_df], axis=1)
        for brdid in drug_valid_id_list:
            brd_df = cell_cp_df.filter(like=brdid, axis=1)
            valid_cp_df = pd.concat([valid_cp_df, brd_df], axis=1)
        for brdid in drug_test_id_list:
            brd_df = cell_cp_df.filter(like=brdid, axis=1)
            test_cp_df = pd.concat([test_cp_df, brd_df], axis=1)
        print(cell, 'train_cp_num:', train_cp_df.shape[1])
        print(cell, 'valid_cp_num:', valid_cp_df.shape[1])
        print(cell, 'test_cp_num:', test_cp_df.shape[1])
        

        kd_file = 'dataset2/kd_data/' + 'trt_sh_'+cell + '_core_signatures.tsv'
        cell_kd_df = pd.read_csv(kd_file, sep='\t').drop(columns='rid')

        # 合并数据时去除重复列
        for kdid in target_train_id_list:
            kd_df = cell_kd_df.filter(like=kdid, axis=1)
            train_kd_df = pd.concat([train_kd_df, kd_df], axis=1)

        for kdid in target_valid_id_list:
            kd_df = cell_kd_df.filter(like=kdid, axis=1)
            valid_kd_df = pd.concat([valid_kd_df, kd_df], axis=1)

        for kdid in target_test_id_list:
            kd_df = cell_kd_df.filter(like=kdid, axis=1)
            test_kd_df = pd.concat([test_kd_df, kd_df], axis=1)

        print(cell, 'train_kd_num:', train_kd_df.shape[1])
        print(cell, 'valid_kd_num:', valid_kd_df.shape[1])
        print(cell, 'test_kd_num:', test_kd_df.shape[1])

    # 在拼接后的数据中去除重复列
    train_cp_df = train_cp_df.loc[:, ~train_cp_df.columns.duplicated()]
    valid_cp_df = valid_cp_df.loc[:, ~valid_cp_df.columns.duplicated()]
    test_cp_df = test_cp_df.loc[:, ~test_cp_df.columns.duplicated()]
    train_kd_df = train_kd_df.loc[:, ~train_kd_df.columns.duplicated()]
    valid_kd_df = valid_kd_df.loc[:, ~valid_kd_df.columns.duplicated()]
    test_kd_df = test_kd_df.loc[:, ~test_kd_df.columns.duplicated()]

    return train_cp_df, valid_cp_df, test_cp_df, train_kd_df, valid_kd_df, test_kd_df

def get_state_df2(cell_lines, train_id_list, valid_id_list, test_id_list): #激活实验-过表达数据
    all_oe_df = pd.DataFrame({})
    train_cp_df = pd.DataFrame({})
    valid_cp_df = pd.DataFrame({})
    test_cp_df = pd.DataFrame({})
    for cell in cell_lines:
        oe_file = 'dataset2/oe_data/' + 'trt_oe_'+cell + '_core_signatures.tsv'
        cell_oe_df = pd.read_csv(oe_file, sep='\t').drop(columns='rid')
        print(cell, 'oe_num:', cell_oe_df.shape[1])
        all_oe_df = pd.concat([all_oe_df, cell_oe_df], axis=1)

        cp_file = 'dataset2/cp_data_ac/' + 'trt_cp_'+cell + '_filtered_expression_data.tsv'
        cell_cp_df = pd.read_csv(cp_file, sep='\t').drop(columns='rid')

        for brdid in train_id_list:
            brd_df = cell_cp_df.filter(like=brdid, axis=1)
            train_cp_df = pd.concat([train_cp_df, brd_df], axis=1)
        for brdid in valid_id_list:
            brd_df = cell_cp_df.filter(like=brdid, axis=1)
            valid_cp_df = pd.concat([valid_cp_df, brd_df], axis=1)
        for brdid in test_id_list:
            brd_df = cell_cp_df.filter(like=brdid, axis=1)
            test_cp_df = pd.concat([test_cp_df, brd_df], axis=1)
        print(cell, 'train_cp_num:', train_cp_df.shape[1])
        print(cell, 'valid_cp_num:', valid_cp_df.shape[1])
        print(cell, 'test_cp_num:', test_cp_df.shape[1])
    return all_oe_df, train_cp_df, valid_cp_df, test_cp_df

def get_pairs(neg_num, cp_df, kd_df, cp_gene_dict, cell_lines_dict):
    cp_list = cp_df.columns.tolist()
    kd_list = kd_df.columns.tolist()
    exp_pair_list, pair_other_list, lable_list = [], [], []
    for cp in cp_list:
        # BRAF001_A375_24H:BRD-K63675182-003-18-6:0.15625:BRD-K63675182:0.1:24
        cell1 = cp.split(':')[0]
        brdid = cp.split(':')[1]
        dose = cp.split(':')[2]
        time = cp.split(':')[3]
        cp_exp = torch.tensor(np.array([cp_df[cp]]))
        cp_cell = torch.tensor([cell_lines_dict[cell1]])
        cp_dose = torch.tensor([float(dose)])
        cp_time = torch.tensor([float(time)])

        targets_list = [kd for kd in kd_list if kd.split(':')[1] in cp_gene_dict[brdid]]
        targets_kd_num = len(targets_list)
        if targets_kd_num == 0:
            # print('The targets of ' + brdid + ' is not in kd_df !')
            continue
        random_id_list = random.sample(list(set(kd_list).difference(set(targets_list))), neg_num * targets_kd_num)
        # KDC009_A375_96H:TRCN0000004498:-666:ZFAND6:96
        for kd in (targets_list + random_id_list):
            kd_exp = torch.tensor(np.array([kd_df[kd]]))
            kd_time = torch.tensor([float(kd.split(':')[2])])
            exp_pair = torch.cat((cp_exp, kd_exp), dim=0)
            pair_other = torch.cat((cp_cell, cp_time, cp_dose, kd_time), dim=0)
            exp_pair_list.append(exp_pair)
            pair_other_list.append(pair_other)
        lable_list += [1] * targets_kd_num + [0] * targets_kd_num * neg_num
    return torch.stack(exp_pair_list), torch.stack(pair_other_list), torch.tensor(lable_list)

#get_pairs批次版
def data_generator(neg_num, cp_df, kd_df, cp_gene_dict, cell_lines_dict, save_dir, kgpt_embeddings_dict, batch_size):
    """
    数据生成器，分批生成数据以降低内存占用
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cp_list = cp_df.columns.tolist()
    kd_list = kd_df.columns.tolist()
    batch_counter = 0
    for i in range(0, len(cp_list), batch_size):
        exp_pair_list, pair_other_list, kgpt_embedding_list, label_list = [], [], [], []
        for cp in cp_list[i:i + batch_size]:
            cell1 = cp.split(':')[0]
            brdid = cp.split(':')[1]
            dose = cp.split(':')[2]
            time = cp.split(':')[3]
            cp_exp = torch.tensor(cp_df[cp].values, dtype=torch.float32).unsqueeze(0)
            cp_cell = torch.tensor([cell_lines_dict[cell1]], dtype=torch.float32)
            cp_dose = torch.tensor([float(dose)], dtype=torch.float32)
            cp_time = torch.tensor([float(time)], dtype=torch.float32)
            
            targets_list = [kd for kd in kd_list if kd.split(':')[1] in cp_gene_dict[brdid]]
            targets_kd_num = len(targets_list)
            if targets_kd_num == 0:
                continue
            
            random_id_list = random.sample(list(set(kd_list).difference(set(targets_list))), neg_num * targets_kd_num)
            
            # 获取 KPGT 嵌入
            kgpt_embedding = kgpt_embeddings_dict.get(brdid, np.zeros(2304, dtype=np.float32))  # 2304维
            kgpt_embedding_tensor = torch.tensor(kgpt_embedding, dtype=torch.float32)  # [2304]

            for kd in (targets_list + random_id_list):
                kd_exp = torch.tensor(kd_df[kd].values, dtype=torch.float32).unsqueeze(0)
                kd_time = torch.tensor([float(kd.split(':')[2])], dtype=torch.float32)
                exp_pair = torch.cat((cp_exp, kd_exp), dim=0)
                pair_other = torch.cat((cp_cell, cp_time, cp_dose, kd_time), dim=0)
                exp_pair_list.append(exp_pair)
                pair_other_list.append(pair_other)
                kgpt_embedding_list.append(kgpt_embedding_tensor)
            label_list += [1] * targets_kd_num + [0] * targets_kd_num * neg_num
        if exp_pair_list:  # 检查是否有生成的数据
            batch_file = os.path.join(save_dir, f"batch_{batch_counter}.pt")
            torch.save((torch.stack(exp_pair_list), torch.stack(pair_other_list), torch.tensor(label_list, dtype=torch.float32), torch.stack(kgpt_embedding_list)), batch_file)
            batch_counter += 1


#get_pairs批次版
def data_generator_ablation(neg_num, cp_df, kd_df, cp_gene_dict, cell_lines_dict, save_dir, batch_size):
    """
    数据生成器，分批生成数据以降低内存占用
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cp_list = cp_df.columns.tolist()
    kd_list = kd_df.columns.tolist()
    batch_counter = 0
    for i in range(0, len(cp_list), batch_size):
        exp_pair_list, pair_other_list, label_list = [], [], []
        for cp in cp_list[i:i + batch_size]:
            cell1 = cp.split(':')[0]
            brdid = cp.split(':')[1]
            dose = cp.split(':')[2]
            time = cp.split(':')[3]
            cp_exp = torch.tensor(cp_df[cp].values, dtype=torch.float32).unsqueeze(0)
            cp_cell = torch.tensor([cell_lines_dict[cell1]], dtype=torch.float32)
            cp_dose = torch.tensor([float(dose)], dtype=torch.float32)
            cp_time = torch.tensor([float(time)], dtype=torch.float32)
            
            targets_list = [kd for kd in kd_list if kd.split(':')[1] in cp_gene_dict[brdid]]
            targets_kd_num = len(targets_list)
            if targets_kd_num == 0:
                continue
            
            random_id_list = random.sample(list(set(kd_list).difference(set(targets_list))), neg_num * targets_kd_num)
            

            for kd in (targets_list + random_id_list):
                kd_exp = torch.tensor(kd_df[kd].values, dtype=torch.float32).unsqueeze(0)
                kd_time = torch.tensor([float(kd.split(':')[2])], dtype=torch.float32)
                exp_pair = torch.cat((cp_exp, kd_exp), dim=0)
                pair_other = torch.cat((cp_cell, cp_time, cp_dose, kd_time), dim=0)
                exp_pair_list.append(exp_pair)
                pair_other_list.append(pair_other)
            label_list += [1] * targets_kd_num + [0] * targets_kd_num * neg_num
        if exp_pair_list:  # 检查是否有生成的数据
            batch_file = os.path.join(save_dir, f"batch_{batch_counter}.pt")
            torch.save((torch.stack(exp_pair_list), torch.stack(pair_other_list), torch.tensor(label_list, dtype=torch.float32)), batch_file)
            batch_counter += 1

def load_kpgt_embeddings(embedding_file):
    """
    加载 KPGT 嵌入并创建 compound_id 到嵌入向量的映射字典。
    
    参数:
        embedding_file (str): KPGT 嵌入文件路径（kgpt_embeddings.tsv）。
    
    返回:
        dict: compound_id 到嵌入向量的映射。
    """
    kgpt_df = pd.read_csv(embedding_file, sep='\t')
    compound_ids = kgpt_df['compound_id'].tolist()
    embeddings = kgpt_df.drop(columns=['compound_id']).values  # 假设其他列为嵌入向量
    compound_to_embedding = {cid: emb for cid, emb in zip(compound_ids, embeddings)}
    return compound_to_embedding

def get_extest_others(cp_df, kd_df):
    cp_list = cp_df.columns.tolist()
    kd_list = kd_df.columns.tolist()
    pair_other_list = []
    for cp in cp_list:
        # BRAF001_A375_24H:BRD-K63675182-003-18-6:0.15625:BRD-K63675182:0.1:24
        #cell1 = cp.split(':')[0]
        #brdid = cp.split(':')[1]
        dose = cp.split(':')[2]
        time = cp.split(':')[3]
        #cp_exp = torch.tensor(np.array([cp_df[cp]]))
        #cp_cell = torch.tensor([cell_lines_dict[cell1]])
        cp_dose = torch.tensor([float(dose)])
        cp_time = torch.tensor([float(time)])

        # KDC009_A375_96H:TRCN0000004498:-666:ZFAND6:96
        for kd in kd_list:
            #kd_exp = torch.tensor(np.array([kd_df[kd]]))
            kd_time = torch.tensor([float(kd.split(':')[2])])
            #exp_pair = torch.cat((cp_exp, kd_exp), dim=0)
            pair_other = torch.cat((cp_time, cp_dose, kd_time), dim=0)
            #exp_pair_list.append(exp_pair)
            pair_other_list.append(pair_other)
        
    return torch.stack(pair_other_list)

#code from HyperGT
def ExtractV2E(edge_index,num_nodes,num_hyperedges):
    # Assume edge_index = [V|E;E|V]
#     First, ensure the sorting is correct (increasing along edge_index[0])
    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
    if not ((num_nodes+num_hyperedges-1) == edge_index[0].max().item()):
        print('num_hyperedges does not match! 1')
        return
    cidx = torch.where(edge_index[0] == num_nodes)[0].min()  # cidx: [V...|cidx E...]
    V2E = edge_index[:, :cidx].type(torch.LongTensor)
    return V2E

def ConstructH(edge_index_0,num_nodes):
    """
    Construct incidence matrix H of size (num_nodes,num_hyperedges) from edge_index = [V;E]
    """
#     ipdb.set_trace()
    edge_index = torch.zeros_like(edge_index_0,dtype=edge_index_0.dtype)
    edge_index[0]=edge_index_0[0]-edge_index_0[0].min()
    edge_index[1]=edge_index_0[1]-edge_index_0[1].min()
    v=torch.ones(edge_index.shape[1])
    # Don't use edge_index[0].max()+1, as some nodes maybe isolated
    num_hyperedges = edge_index[1].max()+1
    H=torch.sparse.FloatTensor(edge_index, v, torch.Size([num_nodes, num_hyperedges]))
    return H

def add_self_loops(edge_index, edge_weight: Optional[torch.Tensor] = None,
                   fill_value: float = 1., num_nodes: Optional[int] = None):
    
    N = num_nodes

    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    # if edge_index.min() > 0:
    #     loop_index = loop_index + edge_index.min()

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((N, ), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight

class SparseLinear(Module):
    r"""
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # init.ones_(self.weight)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        # wb=torch.sparse.mm(input,self.weight.T).to_dense()+self.bias
        wb=torch.sparse.mm(input,self.weight.T)
        if self.bias is not None:
            out = wb + self.bias
        else:
            out = wb
        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )