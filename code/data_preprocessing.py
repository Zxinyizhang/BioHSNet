
#-------------筛选L1000 phase1 trt_cp数据
from cmapPy.pandasGEXpress import parse
import pandas as pd
from sklearn.cluster import KMeans
import re  # 用于提取数字

# 从 phase1.dti.txt 加载化合物ID
compounds_df = pd.read_csv("dataset/phase1.dti.txt", sep="\t", header=None)
compounds_list = compounds_df[0].tolist()  # 将化合物ID存入列表，不带 "BRD-" 前缀

# 从 matched_gene_ids.txt 加载需要的基因 ID
matched_genes_df = pd.read_csv("dataset2/filter_godata/matched_genes_with_ids.txt", sep="\t")
matched_gene_ids = set(matched_genes_df["ID"].astype(str))  # 将需要的基因 ID 存入集合


# 读取元数据文件，假设元数据文件为 `sig_info.txt`
sig_info = pd.read_csv("dataset/GSE92742_Broad_LINCS_sig_info.txt", sep="\t")

# 定义目标的 pert_type 和细胞系列表
pert_types = ["trt_cp"]
#cell_lines = ["A375"]
cell_lines = ["A375", "A549", "HA1E", "HCC515", "HT29", "MCF7", "PC3", "VCAP"]

# 定义 .gctx 文件路径
gctx_file_path = "dataset/GSE92742_level5_data.gctx"

# 对每个 pert_type 和细胞系分别进行筛选和处理
for pert_type in pert_types:
    # 筛选当前 pert_type 且在 phase1.dti.txt 中存在的化合物
    pert_type_data = sig_info[(sig_info["pert_type"] == pert_type) &
                              (sig_info["pert_id"].str.replace("BRD-", "").isin(compounds_list))]

    for cell_line in cell_lines:
        # 筛选特定细胞系的数据
        cell_line_data = pert_type_data[pert_type_data["cell_id"] == cell_line]
        cell_line_cids = cell_line_data["sig_id"].tolist()
        
        if cell_line_cids:
            # 读取特定样本ID的表达数据
            cell_line_gctoo = parse.parse(gctx_file_path, cid=cell_line_cids)
            cell_line_data_df = cell_line_gctoo.data_df

            # 仅保留需要的基因 ID
            filtered_data_df = cell_line_data_df[cell_line_data_df.index.isin(matched_gene_ids)]

            # 生成新的列名，去除 "BRD-" 前缀
            new_column_names = []
            for cid in filtered_data_df.columns:
                sample_info = cell_line_data[cell_line_data["sig_id"] == cid].iloc[0]
                pert_id_no_prefix = sample_info["pert_id"].replace("BRD-", "")
                
                if pert_type == "trt_cp":
                    new_column_name = f"{sample_info['cell_id']}:{pert_id_no_prefix}:{sample_info['pert_dose']}:{sample_info['pert_time']}"
                elif pert_type == "trt_sh":
                    new_column_name = f"{sample_info['cell_id']}:{sample_info['pert_id']}:{sample_info['pert_iname']}:{sample_info['pert_time']}"
                
                new_column_names.append(new_column_name)
            
            # 设置新的列名
            filtered_data_df.columns = new_column_names

            # 保存为 TSV 文件
            output_file = f"dataset2/cp_data/{pert_type}_{cell_line}_filtered_expression_data.tsv"
            filtered_data_df.to_csv(output_file, sep="\t")
            print(f"已保存 {pert_type} - {cell_line} 数据至 {output_file}")
        else:
            print(f"没有找到 {pert_type} - {cell_line} 的数据")

#--------------sh data
from cmapPy.pandasGEXpress import parse
import pandas as pd
from sklearn.cluster import KMeans
import re  # 用于提取数字

# 从 matched_gene_ids.txt 加载需要的基因 ID
matched_genes_df = pd.read_csv("dataset2/filter_godata/matched_genes_with_ids.txt", sep="\t")
matched_gene_ids = set(matched_genes_df["ID"].astype(str))  # 将需要的基因 ID 存入集合


# 读取元数据文件，假设元数据文件为 `sig_info.txt`
sig_info = pd.read_csv("dataset/GSE92742_Broad_LINCS_sig_info.txt", sep="\t")

# 定义目标的细胞系列表
cell_lines = ["A375", "A549", "HA1E", "HCC515", "HT29", "MCF7", "PC3", "VCAP"]

# 定义 .gctx 文件路径
gctx_file_path = "dataset/GSE92742_level5_data.gctx"

trt_sh_data = sig_info[sig_info["pert_type"] == "trt_sh"]
# 对每个细胞系进行处理
for cell_line in cell_lines:
    # 筛选出当前细胞系的 trt_sh 样本
    cell_line_data = trt_sh_data[trt_sh_data["cell_id"] == cell_line]
    
    # 筛选出当前细胞系的 trt_sh 样本，并只筛选需要的基因
    cell_line_cids = cell_line_data["sig_id"].tolist()
    if cell_line_cids:
        # 读取特定样本ID的表达数据
        cell_line_gctoo = parse.parse(gctx_file_path, cid=cell_line_cids)
        cell_line_data_df = cell_line_gctoo.data_df
        # 提前过滤不需要的基因
        cell_line_data_df = cell_line_data_df[cell_line_data_df.index.isin(matched_gene_ids)]

        # 获取该细胞系下所有不同的干扰名称（pert_iname）
        pert_inames = cell_line_data["pert_iname"].unique()
        
        core_signatures = []  # 用于存储每个扰动的核心签名
        column_names = []  # 用于存储每个扰动的列名
        
        for pert_iname in pert_inames:
            # 筛选出特定 `pert_iname` 的样本
            iname_data = cell_line_data[cell_line_data["pert_iname"] == pert_iname]
            
            # 获取该干扰名称下所有不同的 KD 时间点（pert_itime）
            pert_itimes = iname_data["pert_itime"].unique()
            
            for pert_itime in pert_itimes:
                # 提取 `pert_itime` 中的数字部分
                time_numeric = re.search(r'\d+', str(pert_itime)).group()
                
                # 筛选出当前 KD 时间的样本 ID
                time_specific_cids = iname_data["sig_id"][iname_data["pert_itime"] == pert_itime].tolist()
                
                if time_specific_cids:  # 确保有数据
                    # 提取时间点和干扰名称的样本数据
                    time_specific_data = cell_line_data_df[time_specific_cids].T  # 转置，使样本为行
                    
                    # 使用 KMeans 聚类，k=1
                    kmeans = KMeans(n_clusters=1, n_init=10, random_state=0)
                    kmeans.fit(time_specific_data)
                    
                    # 获取聚类中心
                    core_signature = kmeans.cluster_centers_[0]
                    core_signatures.append(core_signature)  # 将核心签名添加到列表中
                    
                    # 生成列名格式为：细胞系:pert_iname:时间数字
                    column_name = f"{cell_line}:{pert_iname}:{time_numeric}"
                    column_names.append(column_name)  # 将列名添加到列表中
        
        if core_signatures:
            # 将所有核心签名拼接为一个 DataFrame，恢复为基因在行，扰动样本在列
            core_signatures_df = pd.DataFrame(core_signatures).T
            core_signatures_df.columns = column_names  # 设置列名
            core_signatures_df.index = cell_line_data_df.index  # 设置行名为基因名称
            
            # 保存为 .tsv 文件
            output_file = f"dataset2/kd_data/trt_sh_{cell_line}_core_signatures.tsv"
            core_signatures_df.to_csv(output_file, sep="\t")
            
            print(f"已保存 {cell_line} 的核心签名数据至 {output_file}")
        else:
            print(f"{cell_line} 没有有效的核心签名数据保存")
    else:
        print(f"没有找到 {cell_line} 的数据")

#--------------cmapPy筛选L1000 phase2数据
#----------------cp data
from cmapPy.pandasGEXpress import parse 
import pandas as pd
from sklearn.cluster import KMeans
import re  # 用于提取数字

# 从 phase2.dti.txt 加载化合物ID
compounds_df = pd.read_csv("dataset/phase2.dti.txt", sep="\t", header=None)
compounds_list = compounds_df[0].tolist()  # 将化合物ID存入列表，不带 "BRD-" 前缀

# 从 matched_gene_ids.txt 加载需要的基因 ID
matched_genes_df = pd.read_csv("dataset2/filter_godata/matched_genes_with_ids.txt", sep="\t")
matched_gene_ids = set(matched_genes_df["ID"].astype(str))  # 将需要的基因 ID 存入集合

# 读取元数据文件，假设元数据文件为 `sig_info.txt`
sig_info = pd.read_csv("dataset/GSE70138_Broad_LINCS_sig_info.txt", sep="\t")

# 定义目标的 pert_type 和细胞系列表
pert_types = ["trt_cp"]
#cell_lines = ["A375"]
cell_lines = ["A375", "A549", "HA1E", "HCC515", "HT29", "MCF7", "PC3", "VCAP"]

# 定义 .gctx 文件路径
gctx_file_path = "dataset/GSE70138_level5_data.gctx"

# 对每个 pert_type 和细胞系分别进行筛选和处理
for pert_type in pert_types:
    # 筛选当前 pert_type 且在 phase1.dti.txt 中存在的化合物
    pert_type_data = sig_info[(sig_info["pert_type"] == pert_type) &
                              (sig_info["pert_id"].str.replace("BRD-", "").isin(compounds_list))]

    for cell_line in cell_lines:
        # 筛选特定细胞系的数据
        cell_line_data = pert_type_data[pert_type_data["cell_id"] == cell_line]
        cell_line_cids = cell_line_data["sig_id"].tolist()
        
        if cell_line_cids:
            # 读取特定样本ID的表达数据
            cell_line_gctoo = parse.parse(gctx_file_path, cid=cell_line_cids)
            cell_line_data_df = cell_line_gctoo.data_df

            # 过滤掉不需要的基因 ID
            filtered_data_df = cell_line_data_df[cell_line_data_df.index.isin(matched_gene_ids)]

            # 生成新的列名，去除 "BRD-" 前缀
            new_column_names = []
            for cid in filtered_data_df.columns:
                sample_info = cell_line_data[cell_line_data["sig_id"] == cid].iloc[0]
                pert_id_no_prefix = sample_info["pert_id"].replace("BRD-", "")
                
                # 提取 `pert_idose` 和 `pert_itime` 的数字部分
                pert_idose_numeric = re.search(r'\d+', str(sample_info["pert_idose"])).group() if re.search(r'\d+', str(sample_info["pert_idose"])) else ""
                pert_itime_numeric = re.search(r'\d+', str(sample_info["pert_itime"])).group() if re.search(r'\d+', str(sample_info["pert_itime"])) else ""

                if pert_type == "trt_cp":
                    new_column_name = f"{sample_info['cell_id']}:{pert_id_no_prefix}:{pert_idose_numeric}:{pert_itime_numeric}"
                elif pert_type == "trt_sh":
                    new_column_name = f"{sample_info['cell_id']}:{sample_info['pert_id']}:{sample_info['pert_iname']}:{sample_info['pert_time']}"
                
                new_column_names.append(new_column_name)
            
            # 设置新的列名
            filtered_data_df.columns = new_column_names

            # 保存为 TSV 文件
            output_file = f"dataset2/external_data/{pert_type}_{cell_line}_test_set.tsv"
            filtered_data_df.to_csv(output_file, sep="\t")
            print(f"已保存 {pert_type} - {cell_line} 数据至 {output_file}")
        else:
            print(f"没有找到 {pert_type} - {cell_line} 的数据")
