import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

#读取GO数据，转换为关联矩阵H
import numpy as np
def read_association_file(genesetsf):  
    geneidx = {}  
    genecnt = 0  
    geneids = []  
    cat2gene = {}  
    with open(genesetsf) as fr:  
        for line in fr:  
            catid = line.split('\t')[0]  
            cat2gene[catid] = []  
            for gene in line.split('\n')[0].split('\t')[2].split(','):  
                cat2gene[catid].append(gene)  
                if gene not in geneidx:  
                    geneidx[gene] = genecnt  
                    geneids.append(gene)  
                    genecnt += 1  
  
    H = np.zeros([len(geneids), len(cat2gene)])  
    catids = list(cat2gene.keys())  
    for i,cat in enumerate(catids):  
        for gene in cat2gene[cat]:  
            H[geneidx[gene], i] = 1  
    return H, catids, geneids

#计算超边权重
def go_ic(C_GO, association):  #association 即 H
    goidx = {go:i for i,go in enumerate(C_GO)}  
  
    from goatools.base import get_godag  
    from goatools.gosubdag.gosubdag import GoSubDag  
    godag = get_godag("dataset/go.obo")  
  
    freq = np.sum(association, axis=0)  
    freq_copy = np.sum(association, axis=0)  
    go_ = list(C_GO)  
    gosubdag_r0 = GoSubDag(go_, godag, prt=None)  
    for go in C_GO:
        if go in gosubdag_r0.rcntobj.go2parents:
            for pgo in gosubdag_r0.rcntobj.go2parents[go]:
                if pgo in C_GO:
                    freq_copy[goidx[pgo]] += freq[goidx[go]] 
    print(freq_copy / np.sum(freq))  
    ic = -np.log2(freq_copy / np.sum(freq))  
    ic = np.power(ic, 2)  
    print(freq)  
    print(freq_copy)  
    ic[np.where(freq_copy == 0)] = 0  
    return ic

#用关联矩阵H计算邻接矩阵G
def generate_G_from_H(catids, H, variable_weight=False):  #超边权重可变
    """  
    calculate G from hypgraph incidence matrix H    :param H: hypergraph incidence matrix H    :param variable_weight: whether the weight of hyperedge is variable    :return: G    """    
    if type(H) != list:  
        return _generate_G_from_H(catids, H, variable_weight)  
    else:  
        G = []  
        for sub_H in H:  
            G.append(generate_G_from_H(sub_H, variable_weight))  
        return G

def _generate_G_from_H(catids, H, variable_weight=False):  
    """  
    calculate G from hypgraph incidence matrix H    :param H: hypergraph incidence matrix H    :param variable_weight: whether the weight of hyperedge is variable    :return: G    """    
    H = np.array(H)  
    #n_edge = H.shape[1]  
    # the weight of the hyperedge  
    #W = np.ones(n_edge)  #超边权重为1
    W = go_ic(catids, H) #超边权重为ic值
    # the degree of the node  
    DV = np.sum(H * W, axis=1)  
    # the degree of the hyperedge  
    DE = np.sum(H, axis=0)  
  
    invDE = np.mat(np.diag(np.power(DE, -1)))  
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))  
    W = np.mat(np.diag(W))  
    H = np.mat(H)  
    HT = H.T  
  
    if variable_weight:  
        DV2_H = DV2 * H  
        invDE_HT_DV2 = invDE * HT * DV2  
        return DV2_H, W, invDE_HT_DV2  
    else:  
        G = DV2 * H * W * invDE * HT * DV2  
        return G


#卷积层
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        # x 的形状为 [num_nodes, in_ft]（在 batch 处理中为 [batch_size, num_nodes, in_ft]）
        # G 的形状为 [num_nodes, num_nodes]

        x = torch.matmul(x, self.weight)  # [num_nodes, out_ft]，out_ft=1 所以保持为 [num_nodes, 1]
        
        if self.bias is not None:
            x = x + self.bias

        # 确保 G 的形状为 [num_nodes, num_nodes]，使其与 x 形状兼容
        x = torch.matmul(G, x)  # [num_nodes, out_ft]
        return x

class HGNN_encoder(nn.Module):
    def __init__(self, drop_out):
        super(HGNN_encoder, self).__init__()
        self.hgnn = HGNN_conv(1, 1)  # 输入和输出维度均为 1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_out)
        
        # 两层全连接网络，将输出降维到 (1, 100)
        # self.fc = nn.Sequential(
        #     nn.Linear(11677, 4096),  # 假设 num_nodes 为 11677
        #     nn.ReLU(),
        #     nn.Dropout(p=drop_out),
        #     nn.Linear(4096, 1024),  # 假设 num_nodes 为 11677
        #     nn.ReLU(),
        #     nn.Dropout(p=drop_out),
        #     nn.Linear(1024, 100)
        # )
        self.fc = nn.Sequential(
            nn.Linear(5355, 1024),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            nn.Linear(1024, 100)
        )

    def forward(self, inputs, G):
        
        inputs=inputs.unsqueeze(-1) # inputs 形状为 [batch_size, num_nodes, 1]
        output = self.hgnn(inputs, G)  # [batch_size, num_nodes, 1]
        
        output = output.view(output.size(0), -1)  # 将 x reshape 为 [batch_size, 1, num_nodes]
        
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc(output)  # 降维到 [batch_size, 100]
        return output

class HGNN_encoder_ablation(nn.Module):
    def __init__(self, drop_out):
        super(HGNN_encoder_ablation, self).__init__()
        self.hgnn = HGNN_conv(1, 1)  # 输入和输出维度均为 1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_out)
        
        self.fc = nn.Sequential(
            nn.Linear(5355, 2048),   #正常是2048
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            nn.Linear(2048, 100)
        )

    def forward(self, inputs, G):
        
        inputs=inputs.unsqueeze(-1) # inputs 形状为 [batch_size, num_nodes, 1]
        output = self.hgnn(inputs, G)  # [batch_size, num_nodes, 1]
        
        output = output.view(output.size(0), -1)  # 将 x reshape 为 [batch_size, 1, num_nodes]
        
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc(output)  # 降维到 [batch_size, 100]
        return output


#合并KPGT嵌入
class HGNN_encoder2(nn.Module):
    def __init__(self, drop_out, kgpt_embedding_dim=2304, combined_embedding_dim=100):
        super(HGNN_encoder2, self).__init__()
        self.hgnn = HGNN_conv(1, 1)  # 输入和输出维度均为1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_out)
        
        # 全连接层，将 HGNN 输出降维到 100
        self.fc = nn.Sequential(
            nn.Linear(5355, 2048),   # 根据您的具体数据调整
            nn.ReLU(),
            nn.Dropout(p=drop_out)
        )
        
        
        # 新增线性层用于合并 HGNN 输出与降维后的 KPGT 嵌入，并通过中间层降维到 100 维
        self.combine_fc = nn.Sequential(
            nn.Linear(2048+2304, 2048),  # 输入维度 500，输出维度 256
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            nn.Linear(2048, combined_embedding_dim)  # 输入维度 256，输出维度 100
        )
  

    def forward(self, inputs, G, kgpt_embedding):
        """
        前向传播函数

        参数:
            inputs (torch.Tensor): 化合物或基因的表达数据 [batch_size, num_nodes]
            G (torch.Tensor): 图结构 [batch_size, num_nodes, num_nodes]
            kgpt_embedding (torch.Tensor): KPGT 嵌入 [batch_size, 2304]
        
        返回:
            torch.Tensor: 编码后的嵌入 [batch_size, 100]
        """
        # HGNN 编码
        inputs = inputs.unsqueeze(-1)  # [batch_size, num_nodes, 1]
        output = self.hgnn(inputs, G)  # [batch_size, num_nodes, 1]
        output = output.view(output.size(0), -1)  # [batch_size, num_nodes]
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc(output)  # [batch_size, 2048]
        
        # 合并 HGNN 输出与降维后的 KPGT 嵌入
        combined = torch.cat((output, kgpt_embedding), dim=1)  # [batch_size, 2048+2304]
        combined = self.combine_fc(combined)  # [batch_size, 100]
        
        return combined

class predictor(nn.Module):
    def __init__(self):
        super(predictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )

    def get_r2(self, input1, input2):
        pred1 = input1 - torch.mean(input1, dim=1, keepdim=True)
        pred2 = input2 - torch.mean(input2, dim=1, keepdim=True)
        pred1_norm = torch.sqrt(torch.sum(torch.pow(pred1, 2), dim=1, keepdim=True))
        pred2_norm = torch.sqrt(torch.sum(torch.pow(pred2, 2), dim=1, keepdim=True))
        pred1_pred2 = torch.sum(torch.mul(pred1, pred2), dim=1, keepdim=True)
        r2 = torch.pow(pred1_pred2 / (pred1_norm * pred2_norm), 2)
        return r2

    def forward(self, input1, input2, input_others):
        r2 = self.get_r2(input1, input2)
        if input_others.dim() == 1:
            input_others = input_others.unsqueeze(0)
        output = torch.cat([r2, input_others], dim=1)
        output = self.mlp(output)
        return output


class Mymodel(nn.Module):
    def __init__(self, device, drop_out, kgpt_embedding_dim=2304, combined_embedding_dim=100):
        super(Mymodel, self).__init__()
        self.device = device
        self.encoder2 = HGNN_encoder2(drop_out, kgpt_embedding_dim, combined_embedding_dim)
        self.encoder = HGNN_encoder(drop_out)
        self.predictor = predictor()

    def forward(self, input1, input2, G, input_others, kgpt_embedding):
        output1 = self.encoder2(input1, G, kgpt_embedding)  # 输出形状为 [1, 100]
        output2 = self.encoder(input2, G)
        
        output = self.predictor(output1, output2, input_others)
        return output
    
    @torch.no_grad()
    def inference1(self, inputs, G, kgpt_embeddings, chunk_size):
        num = inputs.shape[0]  # 样本数量
        out = torch.zeros(num, 100)  # 假设最终输出维度为 100
        
        # 逐小批次处理输入数据，避免构建大矩阵
        for start in range(0, num, chunk_size):
            end = min(start + chunk_size, num)
            
            # 获取当前小批次的数据
            x = inputs[start:end].to(self.device)
            kgpt_embedding = kgpt_embeddings[start:end].to(self.device)  # [batch_size, 2304]

            # 使用共享的 G 进行推理
            x = self.encoder2(x, G, kgpt_embedding)
            
            # 存入输出张量
            out[start:end] = x.cpu()
        
        return out
    
    @torch.no_grad()
    def inference11(self, inputs, G, chunk_size):
        num = inputs.shape[0]  # 样本数量
        out = torch.zeros(num, 100)  # 假设最终输出维度为 100
        
        # 逐小批次处理输入数据，避免构建大矩阵
        for start in range(0, num, chunk_size):
            end = min(start + chunk_size, num)
            
            # 获取当前小批次的数据
            x = inputs[start:end].to(self.device)

            # 使用共享的 G 进行推理
            x = self.encoder(x, G)
            
            # 存入输出张量
            out[start:end] = x.cpu()
        
        return out

    @torch.no_grad()
    def inference2(self, inputs, chunk_size):
        num = inputs[2].shape[0]
        out = torch.zeros(num, 2)  # 假设最终输出为二分类（维度为 2）
        
        # 逐小批次处理
        for start in range(0, num, chunk_size):
            end = min(start + chunk_size, num)
            x1 = inputs[0][start:end].to(self.device)
            x2 = inputs[1][start:end].to(self.device)
            z = inputs[2][start:end].to(self.device)
            
            # 直接在 predictor 中进行推理
            x = self.predictor(x1, x2, z)
            out[start:end] = x.cpu()
        
        return out

    @torch.no_grad()
    def inference(self, inputs, G, kgpt_embeddings, chunk_size):
        pair_num = inputs[2].shape[0]  # 样本对的数量
        y = torch.zeros(pair_num, 2)  # 假设输出为二分类结果（维度为 2）

        # 逐小批次处理，避免构建过大矩阵
        for start in range(0, pair_num, chunk_size):
            end = min(start + chunk_size, pair_num)
            
            # 获取当前小批次的数据
            x1 = inputs[0][start:end].to(self.device)
            x2 = inputs[1][start:end].to(self.device)
            z = inputs[2][start:end].to(self.device)

            kgpt_embedding = kgpt_embeddings[start:end].to(self.device)  # [batch_size, 2304]

            # 使用共享的 G 进行编码
            x1 = self.encoder2(x1, G, kgpt_embedding)
            x2 = self.encoder(x2, G)
            
            # 将编码结果传入 predictor 进行推理
            x = self.predictor(x1, x2, z)
            y[start:end] = x.cpu()
        
        return y

class Mymodel_ablation(nn.Module):
    def __init__(self, device, drop_out):
        super(Mymodel_ablation, self).__init__()
        self.device = device
        self.encoder = HGNN_encoder_ablation(drop_out)
        self.predictor = predictor()

    def forward(self, input1, input2, G, input_others):
        output1 = self.encoder(input1, G)  # 输出形状为 [1, 100]
        output2 = self.encoder(input2, G)
        
        output = self.predictor(output1, output2, input_others)
        return output
    
    @torch.no_grad()
    def inference1(self, inputs, G, chunk_size):
        num = inputs.shape[0]  # 样本数量
        out = torch.zeros(num, 100)  # 假设最终输出维度为 100
        
        # 逐小批次处理输入数据，避免构建大矩阵
        for start in range(0, num, chunk_size):
            end = min(start + chunk_size, num)
            
            # 获取当前小批次的数据
            x = inputs[start:end].to(self.device)

            # 使用共享的 G 进行推理
            x = self.encoder(x, G)
            
            # 存入输出张量
            out[start:end] = x.cpu()
        
        return out


    @torch.no_grad()
    def inference2(self, inputs, chunk_size):
        num = inputs[2].shape[0]
        out = torch.zeros(num, 2)  # 假设最终输出为二分类（维度为 2）
        
        # 逐小批次处理
        for start in range(0, num, chunk_size):
            end = min(start + chunk_size, num)
            x1 = inputs[0][start:end].to(self.device)
            x2 = inputs[1][start:end].to(self.device)
            z = inputs[2][start:end].to(self.device)
            
            # 直接在 predictor 中进行推理
            x = self.predictor(x1, x2, z)
            out[start:end] = x.cpu()
        
        return out

    @torch.no_grad()
    def inference(self, inputs, G, chunk_size):
        pair_num = inputs[2].shape[0]  # 样本对的数量
        y = torch.zeros(pair_num, 2)  # 假设输出为二分类结果（维度为 2）

        # 逐小批次处理，避免构建过大矩阵
        for start in range(0, pair_num, chunk_size):
            end = min(start + chunk_size, pair_num)
            
            # 获取当前小批次的数据
            x1 = inputs[0][start:end].to(self.device)
            x2 = inputs[1][start:end].to(self.device)
            z = inputs[2][start:end].to(self.device)

            # 使用共享的 G 进行编码
            x1 = self.encoder(x1, G)
            x2 = self.encoder(x2, G)
            
            # 将编码结果传入 predictor 进行推理
            x = self.predictor(x1, x2, z)
            y[start:end] = x.cpu()
        
        return y
