#-------------直接在gene_info文件中找id和symbol的对应，用这个更合适
import pandas as pd

# Load the gene_info.txt file
gene_info_path = "dataset/GSE92742_Broad_LINCS_gene_info.txt"
gene_info = pd.read_csv(gene_info_path, sep="\t", dtype=str)

# Extract only the relevant columns for gene ID and gene symbol
gene_id_symbol_mapping = gene_info[["pr_gene_id", "pr_gene_symbol"]]

# Save to a new TSV file with specified columns
output_path = "dataset/filter_godata/gene_id_to_symbol_mapping.tsv"
gene_id_symbol_mapping.to_csv(output_path, sep="\t", index=False)

#--------------gene_id_to_symbol_mapping.tsv 文件在goa_human里面筛选，651个基因没找到
import pandas as pd

# 读取 gene_id_to_symbol_mapping.tsv 文件，获取基因 ID 和基因符号
genes_data = pd.read_csv("dataset/filter_godata/gene_id_to_symbol_mapping.tsv", sep="\t", header=0, names=["ID", "Gene_Symbol"])

# 创建基因符号到基因 ID 的映射字典
gene_symbol_to_id = dict(zip(genes_data["Gene_Symbol"], genes_data["ID"]))

# 读取 goa_human.gaf 文件，选择需要的列
goa_data = pd.read_csv("dataset/goa_human.gaf", sep="\t", comment="!", header=None, usecols=[2, 4], names=["Gene_Symbol", "GO_term"])

# 筛选出在 gene_id_to_symbol_mapping.tsv 里的基因符号的行
target_genes = set(genes_data["Gene_Symbol"].dropna())
filtered_goa_data = goa_data[goa_data["Gene_Symbol"].isin(target_genes)]

# 创建基因 ID -> GO 术语的映射
gene_to_go = filtered_goa_data.merge(genes_data, on="Gene_Symbol")
gene_to_go = gene_to_go.groupby("ID")["GO_term"].apply(lambda x: ",".join(set(x))).reset_index()
gene_to_go["GO_Count"] = gene_to_go["GO_term"].apply(lambda x: len(x.split(",")))
gene_to_go = gene_to_go[["ID", "GO_Count", "GO_term"]]
gene_to_go.to_csv("dataset/filter_godata/id_to_go_terms.txt", sep="\t", index=False)

# 创建 GO 术语 -> 基因符号的映射
go_to_gene_symbols = filtered_goa_data.groupby("GO_term")["Gene_Symbol"].apply(lambda x: ",".join(set(x))).reset_index()
go_to_gene_symbols["Gene_Count"] = go_to_gene_symbols["Gene_Symbol"].apply(lambda x: len(x.split(",")))
go_to_gene_symbols = go_to_gene_symbols[["GO_term", "Gene_Count", "Gene_Symbol"]]
go_to_gene_symbols.columns = ["GO_term", "Gene_Count", "Gene_Symbols"]
go_to_gene_symbols.to_csv("dataset/filter_godata/go_term_to_genes.txt", sep="\t", index=False)

# 创建 GO 术语 -> 基因 ID 的映射
go_to_gene_ids = []
for go_term, symbols in go_to_gene_symbols[["GO_term", "Gene_Symbols"]].itertuples(index=False):
    gene_ids = [gene_symbol_to_id[symbol] for symbol in symbols.split(",") if symbol in gene_symbol_to_id]
    go_to_gene_ids.append((go_term, len(gene_ids), ",".join(map(str, gene_ids))))

# 转换为 DataFrame 并保存
go_to_gene_ids_df = pd.DataFrame(go_to_gene_ids, columns=["GO_term", "Gene_Count", "Gene_IDs"])
go_to_gene_ids_df.to_csv("dataset/filter_godata/go_term_to_gene_ids.txt", sep="\t", index=False)

# 检查 gene_id_to_symbol_mapping.tsv 里的基因是否都找到对应的 GO 术语
found_genes = set(filtered_goa_data["Gene_Symbol"].unique())
unmatched_genes = target_genes - found_genes

# 将未匹配到的基因符号和对应的基因 ID 保存到文件
if unmatched_genes:
    with open("dataset/filter_godata/unmatched_genes_with_ids.txt", "w") as f:
        f.write("Gene_Symbol\tID\n")
        for gene in unmatched_genes:
            gene_id = gene_symbol_to_id.get(gene, "N/A")
            f.write(f"{gene}\t{gene_id}\n")
    print("未匹配到的基因符号和对应的基因 ID 已保存到 unmatched_genes_with_ids.txt")
else:
    print("所有基因符号都找到对应的 GO 术语")

#--------------cmapPy筛选L1000 phase1数据,把未找到GO数据的基因删掉
#----------------cp data
from cmapPy.pandasGEXpress import parse 
import pandas as pd
from sklearn.cluster import KMeans
import re  # 用于提取数字

# 从 phase1.dti.txt 加载化合物ID
compounds_df = pd.read_csv("dataset/phase1.dti.txt", sep="\t", header=None)
compounds_list = compounds_df[0].tolist()  # 将化合物ID存入列表，不带 "BRD-" 前缀

# 从 unmatched_genes_with_ids.txt 加载不需要的基因 ID
unmatched_genes_df = pd.read_csv("dataset/filter_godata/unmatched_genes_with_ids.txt", sep="\t")
unmatched_gene_ids = set(unmatched_genes_df["ID"].astype(str))  # 将不需要的基因 ID 存入集合

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

            # 过滤掉不需要的基因 ID
            filtered_data_df = cell_line_data_df[~cell_line_data_df.index.isin(unmatched_gene_ids)]

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
            output_file = f"dataset/cp_data/{pert_type}_{cell_line}_filtered_expression_data.tsv"
            filtered_data_df.to_csv(output_file, sep="\t")
            print(f"已保存 {pert_type} - {cell_line} 数据至 {output_file}")
        else:
            print(f"没有找到 {pert_type} - {cell_line} 的数据")

#--------------sh data
trt_sh_data = sig_info[sig_info["pert_type"] == "trt_sh"]
# 对每个细胞系进行处理
for cell_line in cell_lines:
    # 筛选出当前细胞系的 trt_sh 样本
    cell_line_data = trt_sh_data[trt_sh_data["cell_id"] == cell_line]
    
    # 筛选出当前细胞系的 trt_sh 样本，并过滤掉不需要的基因
    cell_line_cids = cell_line_data["sig_id"].tolist()
    if cell_line_cids:
        # 读取特定样本ID的表达数据
        cell_line_gctoo = parse.parse(gctx_file_path, cid=cell_line_cids)
        cell_line_data_df = cell_line_gctoo.data_df
        # 提前过滤不需要的基因
        cell_line_data_df = cell_line_data_df[~cell_line_data_df.index.isin(unmatched_gene_ids)]

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
            output_file = f"dataset/kd_data/trt_sh_{cell_line}_core_signatures.tsv"
            core_signatures_df.to_csv(output_file, sep="\t")
            
            print(f"已保存 {cell_line} 的核心签名数据至 {output_file}")
        else:
            print(f"{cell_line} 没有有效的核心签名数据保存")
    else:
        print(f"没有找到 {cell_line} 的数据")
#---------提取sh data中的敲除基因都有哪些
import pandas as pd
import glob
import re

# 定义数据文件路径
data_files_path = "dataset/kd_data/trt_sh_*_core_signatures.tsv"

# 用于存储所有细胞系中的唯一 pert_iname
all_pert_inames = set()

# 遍历每个数据文件
for file_path in glob.glob(data_files_path):
    # 提取列名
    data_df = pd.read_csv(file_path, sep="\t", index_col=0)
    column_names = data_df.columns.tolist()
    
    # 从列名中提取 pert_iname
    for col in column_names:
        # 假设列名格式为 "细胞系:pert_iname:时间数字"，提取 pert_iname
        parts = col.split(":")
        if len(parts) > 1:
            pert_iname = parts[1]
            all_pert_inames.add(pert_iname)

# 将唯一的 pert_iname 保存到 txt 文件中
with open("unique_pert_inames.txt", "w") as f:
    for pert_iname in sorted(all_pert_inames):
        f.write(f"{pert_iname}\n")

#-----比较unique_pert_inames.txt和kd_genes.txt的差别，完全一致
import pandas as pd

# 读取 unique_pert_inames.txt 文件的基因
with open("unique_pert_inames.txt", "r") as f:
    unique_pert_inames = set(line.strip() for line in f if line.strip())

# 读取 kd_genes.txt 文件的基因
with open("kd_genes.txt", "r") as f:
    kd_genes = set(line.strip() for line in f if line.strip())

# 找出相同和不同的基因
common_genes = unique_pert_inames & kd_genes
unique_to_pert_inames = unique_pert_inames - kd_genes
unique_to_kd_genes = kd_genes - unique_pert_inames

# 统计相同基因的数量
common_genes_count = len(common_genes)

# 创建 DataFrame 并保存到文件
comparison_df = pd.DataFrame({
    "Common_Genes": list(common_genes) + [""] * (max(len(unique_to_pert_inames), len(unique_to_kd_genes)) - len(common_genes)),
    "Unique_to_Pert_Inames": list(unique_to_pert_inames) + [""] * (max(len(common_genes), len(unique_to_kd_genes)) - len(unique_to_pert_inames)),
    "Unique_to_KD_Genes": list(unique_to_kd_genes) + [""] * (max(len(common_genes), len(unique_to_pert_inames)) - len(unique_to_kd_genes))
})

output_file = "gene_comparison_results.txt"
comparison_df.to_csv(output_file, sep="\t", index=False)

# 输出结果
print(f"相同的基因种类数: {common_genes_count}")


#--------------cmapPy筛选L1000 phase2数据,把未找到GO数据的基因删掉,11677行
#----------------cp data
from cmapPy.pandasGEXpress import parse 
import pandas as pd
from sklearn.cluster import KMeans
import re  # 用于提取数字

# 从 phase2.dti.txt 加载化合物ID
compounds_df = pd.read_csv("dataset/phase2.dti.txt", sep="\t", header=None)
compounds_list = compounds_df[0].tolist()  # 将化合物ID存入列表，不带 "BRD-" 前缀

# 从 unmatched_genes_with_ids.txt 加载不需要的基因 ID
unmatched_genes_df = pd.read_csv("dataset/filter_godata/unmatched_genes_with_ids.txt", sep="\t")
unmatched_gene_ids = set(unmatched_genes_df["ID"].astype(str))  # 将不需要的基因 ID 存入集合

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
            filtered_data_df = cell_line_data_df[~cell_line_data_df.index.isin(unmatched_gene_ids)]

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
            output_file = f"dataset/external_data/{pert_type}_{cell_line}_test_set.tsv"
            filtered_data_df.to_csv(output_file, sep="\t")
            print(f"已保存 {pert_type} - {cell_line} 数据至 {output_file}")
        else:
            print(f"没有找到 {pert_type} - {cell_line} 的数据")

#--------------cmapPy筛选L1000 phase1数据(用dti2的化合物),把未找到GO数据的基因删掉,不对
#----------------cp data
from cmapPy.pandasGEXpress import parse 
import pandas as pd
from sklearn.cluster import KMeans
import re  # 用于提取数字

# 从 phase2.dti.txt 加载化合物ID
compounds_df = pd.read_csv("dataset/phase2.dti.txt", sep="\t", header=None)
compounds_list = compounds_df[0].tolist()  # 将化合物ID存入列表，不带 "BRD-" 前缀

# 从 unmatched_genes_with_ids.txt 加载不需要的基因 ID
unmatched_genes_df = pd.read_csv("dataset/filter_godata/unmatched_genes_with_ids.txt", sep="\t")
unmatched_gene_ids = set(unmatched_genes_df["ID"].astype(str))  # 将不需要的基因 ID 存入集合

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

            # 过滤掉不需要的基因 ID
            filtered_data_df = cell_line_data_df[~cell_line_data_df.index.isin(unmatched_gene_ids)]

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
            output_file = f"dataset/external_data/phase1{pert_type}_{cell_line}_test_set.tsv"
            filtered_data_df.to_csv(output_file, sep="\t")
            print(f"已保存 {pert_type} - {cell_line} 数据至 {output_file}")
        else:
            print(f"没有找到 {pert_type} - {cell_line} 的数据")




#------------------------------------------------------筛选高变基因得到的数据集--------------------------------
import pandas as pd
import scanpy as sc
from cmapPy.pandasGEXpress import parse

# 文件路径
gctx_file_path = "dataset/GSE92742_level5_data.gctx"
sig_info_file_path = "dataset/GSE92742_Broad_LINCS_sig_info.txt"
trt_cp_h5ad_path = "trt_cp_data.h5ad"
trt_sh_h5ad_path = "trt_sh_data.h5ad"
trt_cp_gene_ids_path = "trt_cp_top_genes.txt"
trt_sh_gene_ids_path = "trt_sh_top_genes.txt"
intersection_gene_ids_path = "intersection_top_genes.txt"

# 读取元数据文件
sig_info = pd.read_csv(sig_info_file_path, sep="\t")

# 定义函数筛选高变异基因
def get_top_genes(pert_type, output_h5ad_path, output_gene_ids_path, n_top_genes=3000):
    """
    筛选指定扰动类型的高变异基因
    参数：
    - pert_type: str,扰动类型(如 trt_cp 或 trt_sh)
    - output_h5ad_path: str,.h5ad 文件输出路径
    - output_gene_ids_path: str,基因 ID 输出路径
    - n_top_genes: int,高变异基因数量,默认3000
    返回：
    - set,高变异基因的集合
    """
    # 筛选pert_type的样本
    samples = sig_info[sig_info["pert_type"] == pert_type]["sig_id"].tolist()

    # 读取GCTX数据，仅提取pert_type的样本
    gctx_data = parse.parse(gctx_file_path, cid=samples)

    # 将数据转换为AnnData对象
    adata = sc.AnnData(X=gctx_data.data_df.T)  # 转置，因为AnnData要求行是样本，列是基因
    adata.var_names = gctx_data.data_df.index  # 基因ID作为变量名
    adata.obs_names = gctx_data.data_df.columns  # 样本ID作为观察名

    # 转换为.h5ad格式保存
    adata.write(output_h5ad_path)
    print(f"转换后的 {pert_type} 数据已保存为 {output_h5ad_path}")

    # 确认数据形式，行是样本，列是基因
    print(f"{pert_type} 数据维度：{adata.shape}（样本数, 基因数）")

    # 筛选前n_top_genes个高变异基因
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

    # 提取高变异基因的ID
    highly_variable_genes = adata.var[adata.var["highly_variable"]].index.tolist()

    # 保存高变异基因ID到txt文件
    with open(output_gene_ids_path, "w") as f:
        for gene_id in highly_variable_genes:
            f.write(f"{gene_id}\n")

    print(f"{pert_type} 前 {n_top_genes} 个高变异基因ID已保存到 {output_gene_ids_path}")

    return set(highly_variable_genes)

# 筛选 trt_cp 和 trt_sh 数据的高变异基因
trt_cp_genes = get_top_genes("trt_cp", trt_cp_h5ad_path, trt_cp_gene_ids_path, n_top_genes=8000)
trt_sh_genes = get_top_genes("trt_sh", trt_sh_h5ad_path, trt_sh_gene_ids_path, n_top_genes=8000)

# 计算交集
intersection_genes = trt_cp_genes & trt_sh_genes

# 保存交集结果到txt文件
with open(intersection_gene_ids_path, "w") as f:
    for gene_id in intersection_genes:
        f.write(f"{gene_id}\n")

print(f"交集基因数：{len(intersection_genes)}")
print(f"交集基因ID已保存到 {intersection_gene_ids_path}")

#-------------gene_id to symbol----
import pandas as pd

# 文件路径
gene_info_path = "dataset/GSE92742_Broad_LINCS_gene_info.txt"
intersection_genes_path = "intersection_top_genes.txt"
output_path = "dataset2/filter_godata/filtered_gene_id_to_symbol_mapping.tsv"

# 加载 gene_info.txt 文件
gene_info = pd.read_csv(gene_info_path, sep="\t", dtype=str)

# 加载 intersection_top_genes.txt 文件
with open(intersection_genes_path, "r") as f:
    intersection_genes = set(f.read().splitlines())  # 加载基因ID列表

# 筛选 intersection_top_genes 中的基因 ID 及其对应的 symbol
filtered_gene_info = gene_info[gene_info["pr_gene_id"].isin(intersection_genes)]

# 仅保留基因 ID 和基因符号列
filtered_gene_info = filtered_gene_info[["pr_gene_id", "pr_gene_symbol"]]

# 保存到新的 TSV 文件
filtered_gene_info.to_csv(output_path, sep="\t", index=False)

print(f"筛选结果已保存到 {output_path}")

#--------------filtered_gene_id_to_symbol_mapping.tsv 文件在goa_human里面筛选，275个基因没找到
import pandas as pd

# 读取 gene_id_to_symbol_mapping.tsv 文件，获取基因 ID 和基因符号
genes_data = pd.read_csv("dataset2/filter_godata/filtered_gene_id_to_symbol_mapping.tsv", sep="\t", header=0, names=["ID", "Gene_Symbol"])

# 创建基因符号到基因 ID 的映射字典
gene_symbol_to_id = dict(zip(genes_data["Gene_Symbol"], genes_data["ID"]))

# 读取 goa_human.gaf 文件，选择需要的列
goa_data = pd.read_csv("dataset/goa_human.gaf", sep="\t", comment="!", header=None, usecols=[2, 4], names=["Gene_Symbol", "GO_term"])

# 筛选出在 gene_id_to_symbol_mapping.tsv 里的基因符号的行
target_genes = set(genes_data["Gene_Symbol"].dropna())
filtered_goa_data = goa_data[goa_data["Gene_Symbol"].isin(target_genes)]

# 创建 GO 术语 -> 基因 ID 的映射
go_to_gene_ids = []
for go_term, symbols in filtered_goa_data.groupby("GO_term")["Gene_Symbol"].apply(lambda x: set(x)).items():
    gene_ids = [gene_symbol_to_id[symbol] for symbol in symbols if symbol in gene_symbol_to_id]
    go_to_gene_ids.append((go_term, len(gene_ids), ",".join(map(str, gene_ids))))

# 转换为 DataFrame 并保存
go_to_gene_ids_df = pd.DataFrame(go_to_gene_ids, columns=["GO_term", "Gene_Count", "Gene_IDs"])
go_to_gene_ids_df.to_csv("dataset2/filter_godata/go_term_to_gene_ids.txt", sep="\t", index=False)

# 检查 gene_id_to_symbol_mapping.tsv 里的基因是否都找到对应的 GO 术语
found_genes = set(filtered_goa_data["Gene_Symbol"].unique())
unmatched_genes = target_genes - found_genes
matched_genes = target_genes & found_genes  # 找到匹配的基因符号

# 将未匹配到的基因符号和对应的基因 ID 保存到文件
if unmatched_genes:
    with open("dataset2/filter_godata/unmatched_genes_with_ids.txt", "w") as f:
        f.write("Gene_Symbol\tID\n")
        for gene in unmatched_genes:
            gene_id = gene_symbol_to_id.get(gene, "N/A")
            f.write(f"{gene}\t{gene_id}\n")
    print("未匹配到的基因符号和对应的基因 ID 已保存到 unmatched_genes_with_ids.txt")
else:
    print("所有基因符号都找到对应的 GO 术语")

# 保存匹配到的基因及对应的ID
with open("dataset2/filter_godata/matched_genes_with_ids.txt", "w") as f:
    f.write("Gene_Symbol\tID\n")
    for gene in matched_genes:
        gene_id = gene_symbol_to_id.get(gene, "N/A")
        f.write(f"{gene}\t{gene_id}\n")
print("匹配到的基因符号和对应的基因 ID 已保存到 matched_genes_with_ids.txt")

#-------------筛选L1000 phase1 trt_cp数据，matched_genes_with_ids
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

#--------------cmapPy筛选L1000 phase2数据,筛选需要的基因
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



# 找到GO数据中第二列数量为1的基因
import pandas as pd

# 读取go.txt文件
data = pd.read_csv("dataset2/filter_godata/go_term_to_gene_ids_yuanshi.txt", header=None, sep="\t", names=["GO_term", "Gene_count", "Genes"])

# 分割基因列表
data["Genes"] = data["Genes"].str.split(",")

# (1) 找出第二列数量为1的行对应的GO术语
count_1_genes = data[data["Gene_count"] == 1]

# (2) 找出第二列数量为1的行对应的基因
single_gene_rows = []
for _, row in count_1_genes.iterrows():
    for gene in row["Genes"]:
        single_gene_rows.append({"GO_term": row["GO_term"], "Gene": gene})

single_gene_df = pd.DataFrame(single_gene_rows)

# (3) 找出第二列数量大于1的行
count_gt1_genes = data[data["Gene_count"] > 1]
multiple_genes = set(g for genes in count_gt1_genes["Genes"] for g in genes)

# 将第二列数量为1的基因进行分类
single_gene_df["Shared"] = single_gene_df["Gene"].apply(lambda g: g in multiple_genes)

# 分别提取在多个GO术语中出现的基因和没有出现的基因
shared_genes = single_gene_df[single_gene_df["Shared"]]["Gene"].unique()
unique_genes = single_gene_df[~single_gene_df["Shared"]]["Gene"].unique()

# (4) 将所有结果合并
result_df = pd.DataFrame({
    "GO_term": list(single_gene_df["GO_term"]),
    "Gene": list(single_gene_df["Gene"]),
    "Shared_Genes": [g if g in shared_genes else None for g in single_gene_df["Gene"]],
    "Unique_Genes": [g if g in unique_genes else None for g in single_gene_df["Gene"]],
})

# 保存结果到txt文件
result_df.to_csv("processed_go_genes.txt", sep="\t", index=False, header=True)

# (5) 找出Unique基因所在的GO术语
unique_go_terms = set(result_df[result_df["Unique_Genes"].notnull()]["GO_term"])

# 从原数据中删除result_df中的GO术语，但保留Unique基因所在的GO术语的行
filtered_data = data[~data["GO_term"].isin(result_df["GO_term"]) | data["GO_term"].isin(unique_go_terms)]

# 将第三列的基因列表转换为逗号分隔的字符串
filtered_data["Genes"] = filtered_data["Genes"].apply(lambda genes: ",".join(genes))

# 保存过滤后的数据到新文件
filtered_data.to_csv("filtered_go_data.txt", sep="\t", index=False, header=False)




#--------------------------------药物结构embedding------------------------------------------
# 找到phase1化合物对应的canonical_smiles
import pandas as pd
from rdkit import Chem
import os

# 文件路径
pert_info_file = 'dataset/GSE92742_Broad_LINCS_pert_info.txt'
phase1_file = 'dataset/phase1.dti.txt'
output_file = 'dataset/phase1_kpgt.csv'
mapping_file = 'dataset/phase1_id_to_label_mapping.csv'

# # phase2
# # 文件路径
# pert_info_file = 'dataset/GSE70138_Broad_LINCS_pert_info.txt'
# phase1_file = 'dataset/phase2.dti.txt'
# output_file = 'dataset/phase2_kpgt.csv'
# mapping_file = 'dataset/phase2_id_to_label_mapping.csv'


# Step 1: 读取pert_info文件
# 假设文件是制表符分隔，并且包含表头
df_info = pd.read_csv(pert_info_file, sep='\t')

# 检查必要的列是否存在
required_columns = ['pert_id', 'canonical_smiles']
for col in required_columns:
    if col not in df_info.columns:
        raise ValueError(f"文件 {pert_info_file} 中缺少必要的列: {col}")

# Step 2: 读取phase1.dti.txt文件
# 假设文件是制表符分隔且无表头
df_phase1 = pd.read_csv(phase1_file, sep='\t', header=None)

# 提取第一列作为化合物ID，并确保它们是字符串类型
phase1_ids = df_phase1[0].astype(str).tolist()

# Step 3: 为phase1的化合物ID添加"BRD-"前缀以匹配pert_id格式
phase1_ids_brd = ['BRD-' + cid for cid in phase1_ids]

# Step 4: 从df_info中筛选出在phase1_ids_brd中的pert_id
df_filtered = df_info[df_info['pert_id'].isin(phase1_ids_brd)].copy()

# 检查是否有未匹配的化合物ID
matched_ids = df_filtered['pert_id'].str.replace('BRD-', '', regex=False).tolist()
unmatched_ids = set(phase1_ids) - set(matched_ids)
if unmatched_ids:
    print(f"警告: 以下化合物ID在 {pert_info_file} 中未找到对应的pert_id:")
    for uid in unmatched_ids:
        print(uid)

# Step 5: 选择需要的列，并去除"BRD-"前缀以恢复原始ID
df_filtered['compound_id'] = df_filtered['pert_id'].str.replace('BRD-', '', regex=False)
df_result = df_filtered[['compound_id', 'canonical_smiles']]

# Step 6: 定义generate_canonical_smiles函数

def generate_canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # 生成canonical SMILES
        return Chem.MolToSmiles(mol, canonical=True)
    else:
        return None

# Step 7: 生成规范化的canonical SMILES
df_result['standard_canonical_smiles'] = df_result['canonical_smiles'].apply(generate_canonical_smiles)

# Step 8: 验证SMILES的合法性
def validate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

# 应用验证函数
df_result['is_valid_smiles'] = df_result['standard_canonical_smiles'].apply(validate_smiles)

# Step 9: 过滤出合法的SMILES
df_valid = df_result[df_result['is_valid_smiles']].copy()

# Step 10: 为每个化合物分配一个唯一的标签
# 创建一个映射字典：compound_id -> label
df_valid = df_valid.reset_index(drop=True)
df_valid['label'] = df_valid.index  # 标签从0开始递增

# 保存映射关系到CSV文件
df_mapping = df_valid[['compound_id', 'label']].copy()
df_mapping.to_csv(mapping_file, index=False)
print(f"化合物ID与标签的映射已保存到 {mapping_file}")

# Step 11: 创建KPGT需要的格式：smiles,label
df_kpgt = df_valid[['standard_canonical_smiles', 'label']].rename(columns={'standard_canonical_smiles': 'smiles'})

# Step 12: 保存为KPGT预处理脚本需要的CSV格式
df_kpgt.to_csv(output_file, index=False)
print(f"已生成符合KPGT要求的CSV文件：{output_file}")

#把phase1_kpgt.csv/phase2_kpgt.csv上传到服务器上KPGT相应目录(datasets文件夹)下，改名为phase2.csv


#----------KPGT获得嵌入后转换文件格式
import numpy as np
import pandas as pd

# 文件路径
npz_file = 'kpgt_base.npz'  # 嵌入文件
mapping_file = 'dataset/phase1_id_to_label_mapping.csv'  # compound_id 到 label 的映射文件
output_tsv = 'kgpt_embeddings.tsv'  # 输出的 TSV 文件

# # 文件路径
# npz_file = 'kpgt_base2.npz'  # 嵌入文件 （激活实验数据）
# mapping_file = 'dataset/phase2_id_to_label_mapping.csv'  # compound_id 到 label 的映射文件
# output_tsv = 'kgpt_embeddings2.tsv'  # 输出的 TSV 文件


# Step 1: 加载 npz 文件并查看内容
data = np.load(npz_file)
print("NPZ 文件中包含的数组:", data.files)

# 假设嵌入数组名称为 'embeddings'，根据实际情况修改
if 'fps' in data.files:
    embeddings = data['fps']
elif len(data.files) == 1:
    embeddings = data[data.files[0]]
else:
    raise ValueError("NPZ 文件中包含多个数组，请指定要使用的数组名称。")
print("嵌入的形状:", data['fps'].shape)

# Step 2: 加载映射文件
mapping_df = pd.read_csv(mapping_file)
print("映射文件预览:")
print(mapping_df.head())

# Step 3: 确保标签是从0开始且连续的
if not set(mapping_df['label']) == set(range(len(mapping_df))):
    raise ValueError("标签不连续或不从0开始。请检查映射文件。")

# Step 4: 创建嵌入的 DataFrame，并添加标签
embeddings_df = pd.DataFrame(embeddings)
embeddings_df['label'] = range(len(embeddings_df))

# Step 5: 合并映射关系
merged_df = pd.merge(mapping_df, embeddings_df, on='label')

# 移除 'label' 列，只保留 'compound_id' 和嵌入向量
merged_df = merged_df.drop(columns=['label'])

# Step 6: 保存为 TSV 文件
merged_df.to_csv(output_tsv, sep='\t', index=False)
print(f"嵌入已成功保存为 {output_tsv}")



#-------------------------筛选phase1数据中各个细胞系中测试集里的数据，以便计算测试集上的top_acc-----------
import pandas as pd
import os

# 定义细胞系列表
cell_lines = ['A375', 'A549', 'HA1E', 'HCC515', 'HT29', 'MCF7', 'PC3', 'VCAP']

# 定义表达数据文件的模板
input_file_template = 'dataset2/cp_data/trt_cp_{}_filtered_expression_data.tsv'

# 定义输出文件的模板
output_file_template = 'dataset2/test_cpdata/trt_cp_{}_filtered_test_data.tsv'

# 读取test_id.txt中的药物名称
with open('dataset/test_id.txt', 'r') as f:
    test_drugs = set(line.strip() for line in f if line.strip())

print(f"共有 {len(test_drugs)} 个测试药物名称。")

# 遍历每个细胞系
for cell in cell_lines:
    input_file = input_file_template.format(cell)
    
    if not os.path.exists(input_file):
        print(f"文件 {input_file} 不存在，跳过。")
        continue
    
    print(f"处理细胞系: {cell}，文件: {input_file}")
    
    # 读取表达数据
    df = pd.read_csv(input_file, sep='\t')

    
    # 获取所有列名（除了第一列基因ID）
    all_columns = df.columns.tolist()
    
    # 筛选出需要的列
    # 首先保留基因ID列
    filtered_columns = [all_columns[0]]
    
    # 遍历其余列，检查药物名称是否在test_drugs中
    for col in all_columns[1:]:
        try:
            # 列名格式: 细胞系:药物名称:剂量:用药时间
            parts = col.split(':')
            if len(parts) < 2:
                print(f"列名 {col} 格式不正确，跳过。")
                continue
            drug_name = parts[1]
            if drug_name in test_drugs:
                filtered_columns.append(col)
        except Exception as e:
            print(f"解析列名 {col} 时出错: {e}")
            continue
    
    print(f"细胞系 {cell} 找到 {len(filtered_columns)-1} 个匹配的药物列。")
    
    # 筛选数据
    filtered_df = df[filtered_columns]
    
    # 保存到新文件
    output_file = output_file_template.format(cell)
    filtered_df.to_csv(output_file, sep='\t', index=False)
    print(f"筛选后的数据已保存到 {output_file}\n")









#------------------------后续实验-根据靶标筛选药物------------

# ENPP1既存在于kd_genes.txt又存在于phase1.dti.txt，是K73947551的真实靶标
# 打开文件并检查是否包含 ENPP1，结果存在，根据靶标找药物时，靶标从kd_genes里找
def check_gene_in_file(file_path, gene_name):
    try:
        with open(file_path, 'r') as file:
            genes = file.readlines()
            genes = [gene.strip() for gene in genes]  # 去掉每行的换行符
            if gene_name in genes:
                return f"{gene_name} exists in the file."
            else:
                return f"{gene_name} does not exist in the file."
    except FileNotFoundError:
        return "File not found."

# 文件路径为 'kd_genes.txt'，需要查找的基因为 ENPP1
file_path = 'dataset/kd_genes.txt'
gene_name = 'ENPP1'
# 查找 ENPP1 是否存在
result = check_gene_in_file(file_path, gene_name)
result

def check_target_in_dti_file(file_path, target_name):
    drugs_with_target = []  # 用于存储包含靶标的药物
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                columns = line.strip().split('\t')  # 以制表符分割列
                if len(columns) > 1:  # 确保有至少两列
                    targets = columns[1].split(';')  # 第二列包含靶标，多个靶标以分号隔开
                    if target_name in targets:
                        drugs_with_target.append(columns[0])  # 将药物名称添加到列表中
        if drugs_with_target:
            return f"Drugs containing {target_name}: " + ", ".join(drugs_with_target)
        else:
            return f"{target_name} not found in the second column."
    except FileNotFoundError:
        return "File not found."

# 文件路径为 'phase1.dti.txt'，需要查找的靶标为 ENPP1
file_path = 'dataset/phase1.dti.txt'
target_name = 'LDHA'
# 查找 ENPP1 是否存在于第二列
result = check_target_in_dti_file(file_path, target_name)
print(result)

# 筛选phase1.dti.txt里对应药物少于5个，且与癌症非常相关的靶标
from collections import defaultdict
import pandas as pd
# 重新加载文件
file_path = "dataset/phase1.dti.txt"
target_drug_map = defaultdict(set)

with open(file_path, 'r') as file:
    for line in file:
        columns = line.strip().split('\t')  # 以制表符分割列
        if len(columns) > 1:  # 确保有至少两列
            drug = columns[0]  # 第一列是药物
            targets = columns[1].split(';')  # 第二列包含靶标，多个靶标以分号隔开
            for target in targets:
                target_drug_map[target].add(drug)

# 筛选出出现药物数小于5的靶标
filtered_targets = {target: drugs for target, drugs in target_drug_map.items() if len(drugs) < 5}

# 预定义与肿瘤和癌症相关的关键靶标（基于常见文献和数据库）
cancer_related_targets = {
    "EGFR", "HER2", "ERBB2", "PIK3CA", "PTEN", "BRAF", "KRAS", "NRAS", "HRAS",
    "ALK", "ROS1", "MET", "RET", "FGFR1", "FGFR2", "FGFR3", "FGFR4", "VEGFA",
    "VEGFR1", "VEGFR2", "VEGFR3", "PDGFRB", "PDGFRA", "KIT", "ABL1", "JAK2",
    "STAT3", "MTOR", "CDK4", "CDK6", "CDKN2A", "MDM2", "TP53", "RB1", "MYC",
    "WNT", "APC", "TGFBR1", "SMAD4", "NOTCH1", "HIF1A", "PARP1", "BRCA1",
    "BRCA2", "CHK1", "ATR", "ATM", "TOP1", "TOP2A", "PRKDC"
}

# 筛选与癌症及细胞系相关的靶标
filtered_cancer_targets = {target: drugs for target, drugs in filtered_targets.items() if target in cancer_related_targets}

# 转换为DataFrame以便展示
df_cancer_targets = pd.DataFrame([(target, ', '.join(drugs)) for target, drugs in filtered_cancer_targets.items()],
                                  columns=["Target", "Drugs"])



#筛选PC3细胞系2万＋药物扰动数据
from cmapPy.pandasGEXpress import parse
import pandas as pd
from sklearn.cluster import KMeans
import re  # 用于提取数字

# 从 matched_gene_ids.txt 加载需要的基因 ID
matched_genes_df = pd.read_csv("dataset2/filter_godata/matched_genes_with_ids.txt", sep="\t")
matched_gene_ids = set(matched_genes_df["ID"].astype(str))  # 将需要的基因 ID 存入集合

# 读取元数据文件，假设元数据文件为 `sig_info.txt`
sig_info = pd.read_csv("dataset/GSE92742_Broad_LINCS_sig_info.txt", sep="\t")

# 定义目标的 pert_type 和细胞系列表
pert_types = ["trt_cp"]
cell_lines = ["PC3"]
#cell_lines = ["A375", "A549", "HA1E", "HCC515", "HT29", "MCF7", "PC3", "VCAP"]

# 定义 .gctx 文件路径
gctx_file_path = "dataset/GSE92742_level5_data.gctx"

# 对每个 pert_type 和细胞系分别进行筛选和处理
for pert_type in pert_types:
    # 不再筛选仅在 phase1.dti.txt 中存在的化合物，而是筛选所有 pert_id
    pert_type_data = sig_info[sig_info["pert_type"] == pert_type]

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
            output_file = f"experiment/cp_data/{pert_type}_{cell_line}_expression_data.tsv"
            filtered_data_df.to_csv(output_file, sep="\t")
            print(f"已保存 {pert_type} - {cell_line} 数据至 {output_file}")
        else:
            print(f"没有找到 {pert_type} - {cell_line} 的数据")



#筛选PC3细胞系TGFBR1敲低这一条数据
import pandas as pd
# 读取数据文件
file_path = "dataset2/kd_data/trt_sh_PC3_core_signatures.tsv"
df = pd.read_csv(file_path, sep="\t", index_col=0)  # 第一列作为索引（基因ID）

# 筛选包含 "PC3:TGFBR1:" 格式的列
selected_columns = [col for col in df.columns if col.startswith("PC3:TGFBR1:96")]

# 检查是否找到了匹配的列
if selected_columns:
    # 选取基因 ID 列和匹配的列
    filtered_df = df[selected_columns]
    
    # 保存到新文件
    output_file = "experiment/kd_data/trt_sh_PC3_TGFBR1.tsv"
    filtered_df.to_csv(output_file, sep="\t")
    print(f"已保存 {output_file}")
else:
    print("未找到符合条件的列")


# 保存2万＋所有药物名称
import pandas as pd
# 读取数据文件
file_path = "experiment/cp_data/trt_cp_PC3_expression_data.tsv"
df = pd.read_csv(file_path, sep="\t", index_col=0)  # 假设第一列是基因id

# 提取药物名称
drug_names = set()  # 使用集合自动去重
for col in df.columns:
    # 提取药物名称（假设格式为 "细胞系:药物名称:剂量:时间"）
    drug_name = col.split(':')[1]  # 从列名中提取药物名称
    drug_names.add(drug_name)

# 将药物名称保存到txt文件
output_file = "experiment/total_drug_names.txt"
with open(output_file, 'w') as f:
    for drug in sorted(drug_names):  # 按字母顺序输出
        f.write(f"{drug}\n")
print(f"药物名称已保存至 {output_file}")



#-------------------所有药物结构embedding---------
# 找到化合物对应的canonical_smiles，删除不可用的药物数据
import pandas as pd
from rdkit import Chem
import os

# 文件路径
pert_info_file = 'dataset/GSE92742_Broad_LINCS_pert_info.txt'
phase1_file = 'experiment/total_drug_names.txt'
output_file = 'experiment/total_drug_kpgt.csv'
mapping_file = 'experiment/totaldrug_id_to_label_mapping.csv'
expression_file = 'experiment/cp_data/trt_cp_PC3_expression_data.tsv'

# Step 1: 读取pert_info文件
df_info = pd.read_csv(pert_info_file, sep='\t')

# 检查必要的列是否存在
required_columns = ['pert_id', 'canonical_smiles']
for col in required_columns:
    if col not in df_info.columns:
        raise ValueError(f"文件 {pert_info_file} 中缺少必要的列: {col}")

# Step 2: 读取phase1.dti.txt文件
df_phase1 = pd.read_csv(phase1_file, sep='\t', header=None)

# 提取第一列作为化合物ID，并确保它们是字符串类型
phase1_ids = df_phase1[0].astype(str).tolist()

# Step 3: 为phase1的化合物ID添加"BRD-"前缀以匹配pert_id格式
phase1_ids_brd = ['BRD-' + cid for cid in phase1_ids]

# Step 4: 从df_info中筛选出在phase1_ids_brd中的pert_id
df_filtered = df_info[df_info['pert_id'].isin(phase1_ids_brd)].copy()

# 检查是否有未匹配的化合物ID
matched_ids = df_filtered['pert_id'].str.replace('BRD-', '', regex=False).tolist()
unmatched_ids = set(phase1_ids) - set(matched_ids)
if unmatched_ids:
    print(f"警告: 以下化合物ID在 {pert_info_file} 中未找到对应的pert_id:")
    for uid in unmatched_ids:
        print(uid)

# Step 5: 选择需要的列，并去除"BRD-"前缀以恢复原始ID
df_filtered['compound_id'] = df_filtered['pert_id'].str.replace('BRD-', '', regex=False)
df_result = df_filtered[['compound_id', 'canonical_smiles']]

# Step 6: 定义generate_canonical_smiles函数
def generate_canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # 生成canonical SMILES
        return Chem.MolToSmiles(mol, canonical=True)
    else:
        return None

# Step 7: 生成规范化的canonical SMILES
df_result['standard_canonical_smiles'] = df_result['canonical_smiles'].apply(generate_canonical_smiles)

# Step 8: 验证SMILES的合法性
def validate_smiles(smiles):
    """验证 SMILES 是否有效"""
    if not isinstance(smiles, str) or pd.isna(smiles):  # 确保 smiles 是字符串类型，并排除 NaN
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception as e:
        print(f"SMILES解析错误: {smiles}. 错误信息: {e}")
        return False  # 解析失败时返回 False

# 应用验证函数，确保 df_result['standard_canonical_smiles'] 为空时不会崩溃
df_result['is_valid_smiles'] = df_result['standard_canonical_smiles'].astype(str).apply(validate_smiles)

# Step 9: 过滤出合法的SMILES
df_valid = df_result[df_result['is_valid_smiles']].copy()

# Step 10: 为每个化合物分配一个唯一的标签
# 创建一个映射字典：compound_id -> label
df_valid = df_valid.reset_index(drop=True)
df_valid['label'] = df_valid.index  # 标签从0开始递增

# 保存映射关系到CSV文件
df_mapping = df_valid[['compound_id', 'label']].copy()
df_mapping.to_csv(mapping_file, index=False)
print(f"化合物ID与标签的映射已保存到 {mapping_file}")

# Step 11: 创建KPGT需要的格式：smiles,label
df_kpgt = df_valid[['standard_canonical_smiles', 'label']].rename(columns={'standard_canonical_smiles': 'smiles'})

# Step 12: 保存为KPGT预处理脚本需要的CSV格式
df_kpgt.to_csv(output_file, index=False)
print(f"已生成符合KPGT要求的CSV文件：{output_file}")

# Step 13: 处理 total_drug_names.txt 中不可用的药物
# 读取 `total_drug_names.txt` 中的药物
df_phase1 = pd.read_csv(phase1_file, sep='\t', header=None)

# 提取药物名称并去除无法使用的药物
valid_drugs = df_valid['compound_id'].tolist()
valid_drug_names = [drug for drug in df_phase1[0].astype(str).tolist() if drug in valid_drugs]

# 保存有效药物列表到新文件
valid_drugs_file = 'experiment/total_drug_names_valid.txt'
with open(valid_drugs_file, 'w') as f:
    for drug in valid_drug_names:
        f.write(f"{drug}\n")

print(f"有效药物列表已保存至 {valid_drugs_file}")

# Step 14: 删除 `trt_cp_PC3_expression_data.tsv` 中不在有效药物列表中的列
# 读取 `trt_cp_PC3_expression_data.tsv` 文件
df_expression = pd.read_csv(expression_file, sep='\t', index_col=0)

# 获取有效药物列表
valid_drugs_set = set(valid_drug_names)

# 找到 `trt_cp_PC3_expression_data.tsv` 中不在 `valid_drugs` 中的列
columns_to_remove = [col for col in df_expression.columns if col.split(':')[1] not in valid_drugs_set]

# 删除这些列
df_expression_cleaned = df_expression.drop(columns=columns_to_remove)

# 保存清理后的数据
cleaned_expression_file = 'experiment/cp_data/trt_cp_PC3_expression_data_cleaned.tsv'
df_expression_cleaned.to_csv(cleaned_expression_file, sep='\t')
print(f"已删除无效药物列并保存清理后的文件：{cleaned_expression_file}")

#把total_drug_kpgt.csv放到服务器上KPGT相应目录下，改名为project1_experiment.csv


#----------KPGT获得嵌入后转换文件格式
import numpy as np
import pandas as pd

# 文件路径
npz_file = 'experiment/kpgt_base.npz'  # 嵌入文件
mapping_file = 'experiment/totaldrug_id_to_label_mapping.csv'  # compound_id 到 label 的映射文件
output_tsv = 'experiment/kgpt_embeddings.tsv'  # 输出的 TSV 文件

# Step 1: 加载 npz 文件并查看内容
data = np.load(npz_file)
print("NPZ 文件中包含的数组:", data.files)

# 假设嵌入数组名称为 'embeddings'，根据实际情况修改
if 'fps' in data.files:
    embeddings = data['fps']
elif len(data.files) == 1:
    embeddings = data[data.files[0]]
else:
    raise ValueError("NPZ 文件中包含多个数组，请指定要使用的数组名称。")
print("嵌入的形状:", data['fps'].shape)

# Step 2: 加载映射文件
mapping_df = pd.read_csv(mapping_file)
print("映射文件预览:")
print(mapping_df.head())

# Step 3: 确保标签是从0开始且连续的
if not set(mapping_df['label']) == set(range(len(mapping_df))):
    raise ValueError("标签不连续或不从0开始。请检查映射文件。")

# Step 4: 创建嵌入的 DataFrame，并添加标签
embeddings_df = pd.DataFrame(embeddings)
embeddings_df['label'] = range(len(embeddings_df))

# Step 5: 合并映射关系
merged_df = pd.merge(mapping_df, embeddings_df, on='label')

# 移除 'label' 列，只保留 'compound_id' 和嵌入向量
merged_df = merged_df.drop(columns=['label'])

# Step 6: 保存为 TSV 文件
merged_df.to_csv(output_tsv, sep='\t', index=False)
print(f"嵌入已成功保存为 {output_tsv}")


#-----------------------筛选cDAN数据，为了激活抑制实验--------------------
#--------------phase1 oe data
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

trt_sh_data = sig_info[sig_info["pert_type"] == "trt_oe"]
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
            output_file = f"dataset2/oe_data/trt_oe_{cell_line}_core_signatures.tsv"
            core_signatures_df.to_csv(output_file, sep="\t")
            
            print(f"已保存 {cell_line} 的核心签名数据至 {output_file}")
        else:
            print(f"{cell_line} 没有有效的核心签名数据保存")
    else:
        print(f"没有找到 {cell_line} 的数据")

#--------计算6个细胞系数据里包含了多少种过表达基因
import pandas as pd
import glob

# 细胞系列表
cell_lines = ['A375', 'A549', 'HA1E', 'HT29', 'MCF7', 'PC3']

# 存储所有基因名称的集合
unique_genes = set()

# 读取每个文件
for cell_line in cell_lines:
    file_name = f"dataset2/oe_data/trt_oe_{cell_line}_core_signatures.tsv"
    df = pd.read_csv(file_name, sep='\t')  # 读取制表符分隔的文件
    
    # 提取所有列名，去掉第一列的id
    gene_columns = df.columns[1:]
    
    # 解析基因名称并添加到集合中
    for col in gene_columns:
        parts = col.split(':')
        if len(parts) > 1:
            gene_name = parts[1]  # 取出基因名称部分
            unique_genes.add(gene_name)

# 保存去重后的基因名称到文件
with open("dataset/oe_genes.txt", "w") as f:
    for gene in sorted(unique_genes):  # 按字母顺序排序
        f.write(gene + "\n")

print(f"共找到 {len(unique_genes)} 种不同的基因名称，并已保存到 oe_genes.txt 文件中。")

#-------------------激活药物靶标注释，这样筛太少了只有70多个--------
import pandas as pd

# 加载inhabitor.csv文件（药物靶标对）
inhabitor_df = pd.read_csv('dataset/ac_dti.csv')

# 加载GSE92742_Broad_LINCS_pert_info.txt文件（药物信息，包含'pert_iname'和'pert_id'）
pert_info_df = pd.read_csv('dataset/GSE92742_Broad_LINCS_pert_info.txt', sep='\t')

# 将药物名称列转换为小写，以消除大小写差异
inhabitor_df['Drug_Name'] = inhabitor_df['Drug_Name'].str.lower()
pert_info_df['pert_iname'] = pert_info_df['pert_iname'].str.lower()

# 根据药物名称（inhabitor中的Drug_Name和pert_info中的pert_iname）进行合并
merged_df = pd.merge(inhabitor_df, pert_info_df, how='left', left_on='Drug_Name', right_on='pert_iname')

# 删除缺失值的行
merged_df = merged_df.dropna(subset=['pert_id'])

# 删除pert_id中的'BRD-'前缀
merged_df['pert_id'] = merged_df['pert_id'].str.replace('BRD-', '', regex=False)

# 将多个靶标合并为一个字符串，多个靶标用分号隔开
target_grouped_df = merged_df.groupby('pert_id')['Gene_symbol'].apply(lambda x: ';'.join(x)).reset_index()

# 加载oe_genes.txt文件并获取基因列表
with open('dataset/oe_genes.txt', 'r') as f:
    oe_genes = f.read().splitlines()

# 过滤靶标，保留在oe_genes.txt中的靶标
target_grouped_df['Gene_symbol'] = target_grouped_df['Gene_symbol'].apply(lambda x: ';'.join([gene for gene in x.split(';') if gene in oe_genes]))

# 删除Gene_symbol列为空的行
target_grouped_df = target_grouped_df[target_grouped_df['Gene_symbol'] != '']

# 保存到txt文件，第一列是药物实验序列（pert_id），第二列是药物靶标（Gene_symbol）
target_grouped_df.to_csv('dataset/phase1_act_dti.txt', sep='\t', index=False, header=False)

#phase2同理，还要把和phase1重合的药物删掉

#之后重新筛选L1000 cp_data







#------计算测试集里都有多少种药物
import pandas as pd

# 细胞系列表
cell_lines = ['A375', 'A549', 'HA1E', 'HT29', 'MCF7', 'PC3']

# 计算每个文件的基因数量
for cell_line in cell_lines:
    file_name = f"dataset2/external_data/trt_cp_{cell_line}_test_set.tsv"
    df = pd.read_csv(file_name, sep='\t')  # 读取制表符分隔的文件
    
    # 提取所有列名，去掉第一列的id
    gene_columns = df.columns[1:]
    
    # 存储该文件中的基因名称集合
    unique_genes = set()
    
    # 解析基因名称并添加到集合中
    for col in gene_columns:
        parts = col.split(':')
        if len(parts) > 1:
            gene_name = parts[1]  # 取出基因名称部分
            unique_genes.add(gene_name)
    
    # 输出每个文件包含的不同基因数量
    print(f"{cell_line} 文件包含 {len(unique_genes)} 种不同的基因名称。")

#计算phase2.dti.txt有多少种靶标,842个314种
# 读取文件
with open("phase2.dti.txt", "r") as file:
    # 存储所有靶标的集合
    unique_targets = set()
    # 存储所有靶标的列表（不去重）
    all_targets = []
    
    # 遍历每一行
    for line in file:
        # 按照制表符分隔每行内容
        columns = line.strip().split('\t')
        if len(columns) > 1:
            # 获取药物对应的靶标（第二列），并分割多个靶标
            targets = columns[1].split(';')
            # 将靶标添加到集合中（去重）
            unique_targets.update(targets)
            # 将靶标添加到列表中（不去重）
            all_targets.extend(targets)
    
    # 输出去重后的靶标数量和不去重的靶标数量
    print(f"文件中共有 {len(unique_targets)} 种不同的靶标（去重）。")
    print(f"文件中共有 {len(all_targets)} 个靶标（不去重）。")


#计算phase1数据有多少列
import pandas as pd
# 细胞系列表
cell_lines = ['A375', 'A549', 'HA1E', 'HT29', 'MCF7', 'PC3']

# 存储所有基因名称的集合
unique_genes = set()

# 读取每个文件
for cell_line in cell_lines:
    file_name = f"dataset2/test_cpdata/trt_cp_{cell_line}_filtered_test_data.tsv"
    df = pd.read_csv(file_name, sep='\t')  # 读取制表符分隔的文件
    print(df.shape)
#cp:(5355, 3354)(5355, 5850)(5355, 3782)(5355, 3176)(5355, 10178)(5355, 6544)
#kd:(5355, 3827)(5355, 3725)(5355, 3802)(5355, 3666)(5355, 5309)(5355, 5678)
#cp_test:(5355, 342)(5355, 548)(5355, 397)(5355, 318)(5355, 910)(5355, 650)
#phase2:(5355, 1591)(5355, 273)(5355, 1549)(5355, 1560)(5355, 1597)(5355, 1585)


#------------------------------------------靶标冷启动------------------------------------
#phase1.dti.txt转换成靶标-药物
import pandas as pd

# 读取原始数据
df = pd.read_csv('dataset/phase1.dti.txt', header=None, sep='\t')

# 初始化字典，用于存储靶标对应的药物列表
target_to_drugs = {}

# 遍历每一行数据
for _, row in df.iterrows():
    drugs = row[0].split(';')  # 获取药物列表
    targets = row[1].split(';')  # 获取靶标列表
    
    # 将每个靶标与对应的药物进行映射
    for target in targets:
        if target not in target_to_drugs:
            target_to_drugs[target] = []
        target_to_drugs[target].extend(drugs)

# 为了避免重复药物，我们需要去重
for target in target_to_drugs:
    target_to_drugs[target] = ';'.join(set(target_to_drugs[target]))

# 将字典转化为 DataFrame
result_df = pd.DataFrame(list(target_to_drugs.items()), columns=['Target', 'Drugs'])

# 保存为新的 txt 文件
result_df.to_csv('dataset/phase1_target_drug.txt', index=False, header=False, sep='\t')

#把真实靶标分成训练集、验证集、测试集
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据，只取第一列靶标
df = pd.read_csv('dataset/phase1_target_drug.txt', header=None, delimiter='\t', names=['Target', 'Drugs'])

# 提取第一列靶标
targets = df['Target']

# 使用 train_test_split 将靶标分为训练集和临时集（包含验证集和测试集）
train_targets, temp_targets = train_test_split(targets, test_size=0.2, random_state=42)

# 使用 train_test_split 将临时集分为验证集和测试集
val_targets, test_targets = train_test_split(temp_targets, test_size=0.5, random_state=42)
# 保存训练集、验证集、测试集的靶标
train_targets.to_csv('dataset/train_id_target.txt', index=False, header=False)
val_targets.to_csv('dataset/valid_id_target.txt', index=False, header=False)
test_targets.to_csv('dataset/test_id_target.txt', index=False, header=False)

#--------------------------靶标冷启动--筛选各个细胞系中测试集里的数据，以便计算测试集上的top_acc-----------
import pandas as pd
import os

# 定义细胞系列表
cell_lines = ['A375', 'A549', 'HA1E', 'HCC515', 'HT29', 'MCF7', 'PC3', 'VCAP']

# 定义表达数据文件的模板
input_file_template = 'dataset2/kd_data/trt_sh_{}_core_signatures.tsv'

# 定义输出文件的模板
output_file_template = 'dataset2/test_kddata_targetcold/trt_sh_{}_core_signatures.tsv'

# 读取test_id.txt中的基因名称
with open('dataset/test_id_target.txt', 'r') as f:
    test_targets = set(line.strip() for line in f if line.strip())

print(f"共有 {len(test_targets)} 个测试基因名称。")

# 遍历每个细胞系
for cell in cell_lines:
    input_file = input_file_template.format(cell)
    
    if not os.path.exists(input_file):
        print(f"文件 {input_file} 不存在，跳过。")
        continue
    
    print(f"处理细胞系: {cell}，文件: {input_file}")
    
    # 读取表达数据
    df = pd.read_csv(input_file, sep='\t')

    
    # 获取所有列名（除了第一列基因ID）
    all_columns = df.columns.tolist()
    
    # 筛选出需要的列
    # 首先保留基因ID列
    filtered_columns = [all_columns[0]]
    
    # 遍历其余列，检查基因名称是否在test_targets中
    for col in all_columns[1:]:
        try:
            # 列名格式: 细胞系:基因名称:用药时间
            parts = col.split(':')
            if len(parts) < 2:
                print(f"列名 {col} 格式不正确，跳过。")
                continue
            drug_name = parts[1]
            if drug_name in test_targets:
                filtered_columns.append(col)
        except Exception as e:
            print(f"解析列名 {col} 时出错: {e}")
            continue
    
    print(f"细胞系 {cell} 找到 {len(filtered_columns)-1} 个匹配的基因列。")
    
    # 筛选数据
    filtered_df = df[filtered_columns]
    
    # 保存到新文件
    output_file = output_file_template.format(cell)
    filtered_df.to_csv(output_file, sep='\t', index=False)
    print(f"筛选后的数据已保存到 {output_file}\n")

#----靶标测试集在靶标-药物对里对应的药物有哪些
import pandas as pd

# 读取文件
phase1_df = pd.read_csv('dataset/phase1_target_drug.txt', header=None, sep='\t', names=['Target', 'Drugs'])
test_df = pd.read_csv('dataset/test_id_target.txt', header=None, sep='\t', names=['Target'])

# 找出dataset/phase1_target_drug.txt中dataset/test_id_target.txt的靶标所在的行
matching_rows = phase1_df[phase1_df['Target'].isin(test_df['Target'])]
# 提取药物列并去重
unique_drugs = matching_rows['Drugs'].str.split(';').explode().unique()

# 保存为新的 txt 文件
with open('dataset/test_drugs_targetcold.txt', 'w') as f:
    for drug in unique_drugs:
        f.write(f"{drug}\n")

print("药物已保存到 'dataset/test_drugs_targetcold.txt'")

#---筛选这些药物测试集
import pandas as pd
import os

# 定义细胞系列表
cell_lines = ['A375', 'A549', 'HA1E', 'HCC515', 'HT29', 'MCF7', 'PC3', 'VCAP']

# 定义表达数据文件的模板
input_file_template = 'dataset2/cp_data/trt_cp_{}_filtered_expression_data.tsv'

# 定义输出文件的模板
output_file_template = 'dataset2/test_cpdata_targetcold/trt_cp_{}_filtered_test_data.tsv'

# 读取test_id.txt中的药物名称
with open('dataset/test_drugs_targetcold.txt', 'r') as f:
    test_drugs = set(line.strip() for line in f if line.strip())

print(f"共有 {len(test_drugs)} 个测试药物名称。")

# 遍历每个细胞系
for cell in cell_lines:
    input_file = input_file_template.format(cell)
    
    if not os.path.exists(input_file):
        print(f"文件 {input_file} 不存在，跳过。")
        continue
    
    print(f"处理细胞系: {cell}，文件: {input_file}")
    
    # 读取表达数据
    df = pd.read_csv(input_file, sep='\t')

    
    # 获取所有列名（除了第一列基因ID）
    all_columns = df.columns.tolist()
    
    # 筛选出需要的列
    # 首先保留基因ID列
    filtered_columns = [all_columns[0]]
    
    # 遍历其余列，检查药物名称是否在test_drugs中
    for col in all_columns[1:]:
        try:
            # 列名格式: 细胞系:药物名称:剂量:用药时间
            parts = col.split(':')
            if len(parts) < 2:
                print(f"列名 {col} 格式不正确，跳过。")
                continue
            drug_name = parts[1]
            if drug_name in test_drugs:
                filtered_columns.append(col)
        except Exception as e:
            print(f"解析列名 {col} 时出错: {e}")
            continue
    
    print(f"细胞系 {cell} 找到 {len(filtered_columns)-1} 个匹配的药物列。")
    
    # 筛选数据
    filtered_df = df[filtered_columns]
    
    # 保存到新文件
    output_file = output_file_template.format(cell)
    filtered_df.to_csv(output_file, sep='\t', index=False)
    print(f"筛选后的数据已保存到 {output_file}\n")


#---------------------------------------热启动---------------------------------
#phase1.dti转换为一个药物一个靶标
import pandas as pd

# 读取文件
df = pd.read_csv('dataset/phase1.dti.txt', header=None, sep='\t', names=['Drug', 'Target'])

# 初始化一个空列表来保存新的数据
output_lines = []

# 遍历每一行
for _, row in df.iterrows():
    drug = row['Drug']
    targets = row['Target'].split(';')  # 多个靶标以分号隔开
    for target in targets:
        output_lines.append(f"{drug}\t{target}")  # 每行是一个药物对应一个靶标

# 将数据写入新的 txt 文件
with open('dataset/phase1_dti_1to1.txt', 'w') as f:
    for line in output_lines:
        f.write(f"{line}\n")

print("药物-靶标对已保存到 'dataset/phase1_dti_1to1.txt'")

#------分训练集、验证集、测试集
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据
df = pd.read_csv('dataset/phase1_dti_1to1.txt', header=None, sep='\t', names=['Drug', 'Target'])

# 打乱数据顺序
df_shuffled = df.sample(frac=1, random_state=123).reset_index(drop=True)

# 按照 8:1:1 的比例分割数据
train_df, temp_df = train_test_split(df_shuffled, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# 保存为三个新的文件
train_df.to_csv('dataset/phase1_dti_1to1_train.txt', index=False, header=False, sep='\t')
val_df.to_csv('dataset/phase1_dti_1to1_val.txt', index=False, header=False, sep='\t')
test_df.to_csv('dataset/phase1_dti_1to1_test.txt', index=False, header=False, sep='\t')

print("数据已按 8:1:1 比例分割并保存为三个文件。")

#-------将药物和靶标分别提取出来保存为新的txt文件
import pandas as pd

df_train = pd.read_csv('dataset/phase1_dti_1to1_train.txt', header=None, sep='\t', names=['Drug', 'Target'])

# 提取药物和靶标列并去重
unique_drugs = df_train['Drug'].drop_duplicates()
unique_targets = df_train['Target'].drop_duplicates()

# 保存药物和靶标为新的文件
unique_drugs.to_csv('dataset/train_id_drugs_warm.txt', index=False, header=False)
unique_targets.to_csv('dataset/train_id_targets_warm.txt', index=False, header=False)

print("药物和靶标已分别保存")

df_train = pd.read_csv('dataset/phase1_dti_1to1_val.txt', header=None, sep='\t', names=['Drug', 'Target'])

# 提取药物和靶标列并去重
unique_drugs = df_train['Drug'].drop_duplicates()
unique_targets = df_train['Target'].drop_duplicates()

# 保存药物和靶标为新的文件
unique_drugs.to_csv('dataset/val_id_drugs_warm.txt', index=False, header=False)
unique_targets.to_csv('dataset/val_id_targets_warm.txt', index=False, header=False)

print("药物和靶标已分别保存")

df_train = pd.read_csv('dataset/phase1_dti_1to1_test.txt', header=None, sep='\t', names=['Drug', 'Target'])

# 提取药物和靶标列并去重
unique_drugs = df_train['Drug'].drop_duplicates()
unique_targets = df_train['Target'].drop_duplicates()

# 保存药物和靶标为新的文件
unique_drugs.to_csv('dataset/test_id_drugs_warm.txt', index=False, header=False)
unique_targets.to_csv('dataset/test_id_targets_warm.txt', index=False, header=False)

print("药物和靶标已分别保存")

#---筛选test_id_drugs_warm.txt药物测试集,以便计算top acc
import pandas as pd
import os

# 定义细胞系列表
cell_lines = ['A375', 'A549', 'HA1E', 'HCC515', 'HT29', 'MCF7', 'PC3', 'VCAP']

# 定义表达数据文件的模板
input_file_template = 'dataset2/cp_data/trt_cp_{}_filtered_expression_data.tsv'

# 定义输出文件的模板
output_file_template = 'dataset2/test_cpdata_warm/trt_cp_{}_filtered_test_data.tsv'

# 读取test_id.txt中的药物名称
with open('dataset/test_id_drugs_warm.txt', 'r') as f:
    test_drugs = set(line.strip() for line in f if line.strip())

print(f"共有 {len(test_drugs)} 个测试药物名称。")

# 遍历每个细胞系
for cell in cell_lines:
    input_file = input_file_template.format(cell)
    
    if not os.path.exists(input_file):
        print(f"文件 {input_file} 不存在，跳过。")
        continue
    
    print(f"处理细胞系: {cell}，文件: {input_file}")
    
    # 读取表达数据
    df = pd.read_csv(input_file, sep='\t')

    
    # 获取所有列名（除了第一列基因ID）
    all_columns = df.columns.tolist()
    
    # 筛选出需要的列
    # 首先保留基因ID列
    filtered_columns = [all_columns[0]]
    
    # 遍历其余列，检查药物名称是否在test_drugs中
    for col in all_columns[1:]:
        try:
            # 列名格式: 细胞系:药物名称:剂量:用药时间
            parts = col.split(':')
            if len(parts) < 2:
                print(f"列名 {col} 格式不正确，跳过。")
                continue
            drug_name = parts[1]
            if drug_name in test_drugs:
                filtered_columns.append(col)
        except Exception as e:
            print(f"解析列名 {col} 时出错: {e}")
            continue
    
    print(f"细胞系 {cell} 找到 {len(filtered_columns)-1} 个匹配的药物列。")
    
    # 筛选数据
    filtered_df = df[filtered_columns]
    
    # 保存到新文件
    output_file = output_file_template.format(cell)
    filtered_df.to_csv(output_file, sep='\t', index=False)
    print(f"筛选后的数据已保存到 {output_file}\n")

#---筛选test_id_targets_warm.txt药物测试集,以便计算top acc
import pandas as pd
import os

# 定义细胞系列表
cell_lines = ['A375', 'A549', 'HA1E', 'HCC515', 'HT29', 'MCF7', 'PC3', 'VCAP']

# 定义表达数据文件的模板
input_file_template = 'dataset2/kd_data/trt_sh_{}_core_signatures.tsv'

# 定义输出文件的模板
output_file_template = 'dataset2/test_kddata_warm/trt_sh_{}_core_signatures.tsv'

# 读取test_id.txt中的基因名称
with open('dataset/test_id_targets_warm.txt', 'r') as f:
    test_targets = set(line.strip() for line in f if line.strip())

print(f"共有 {len(test_targets)} 个测试基因名称。")

# 遍历每个细胞系
for cell in cell_lines:
    input_file = input_file_template.format(cell)
    
    if not os.path.exists(input_file):
        print(f"文件 {input_file} 不存在，跳过。")
        continue
    
    print(f"处理细胞系: {cell}，文件: {input_file}")
    
    # 读取表达数据
    df = pd.read_csv(input_file, sep='\t')

    
    # 获取所有列名（除了第一列基因ID）
    all_columns = df.columns.tolist()
    
    # 筛选出需要的列
    # 首先保留基因ID列
    filtered_columns = [all_columns[0]]
    
    # 遍历其余列，检查基因名称是否在test_targets中
    for col in all_columns[1:]:
        try:
            # 列名格式: 细胞系:基因名称:用药时间
            parts = col.split(':')
            if len(parts) < 2:
                print(f"列名 {col} 格式不正确，跳过。")
                continue
            drug_name = parts[1]
            if drug_name in test_targets:
                filtered_columns.append(col)
        except Exception as e:
            print(f"解析列名 {col} 时出错: {e}")
            continue
    
    print(f"细胞系 {cell} 找到 {len(filtered_columns)-1} 个匹配的基因列。")
    
    # 筛选数据
    filtered_df = df[filtered_columns]
    
    # 保存到新文件
    output_file = output_file_template.format(cell)
    filtered_df.to_csv(output_file, sep='\t', index=False)
    print(f"筛选后的数据已保存到 {output_file}\n")


#------------------------------正确划分激活抑制对后的激活实验--------------
from cmapPy.pandasGEXpress import parse
import pandas as pd
from sklearn.cluster import KMeans
import re  # 用于提取数字

# 从 phase1.dti.txt 加载化合物ID
compounds_df = pd.read_csv("phase1.dti_ac.txt", sep="\t", header=None)
compounds_list = compounds_df[0].tolist()  # 将化合物ID存入列表，不带 "BRD-" 前缀

# 从 matched_gene_ids.txt 加载需要的基因 ID
matched_genes_df = pd.read_csv("dataset2/filter_godata/matched_genes_with_ids.txt", sep="\t")
matched_gene_ids = set(matched_genes_df["ID"].astype(str))  # 将需要的基因 ID 存入集合


# 读取元数据文件，假设元数据文件为 `sig_info.txt`
sig_info = pd.read_csv("dataset/GSE92742_Broad_LINCS_sig_info.txt", sep="\t")

# 定义目标的 pert_type 和细胞系列表
pert_types = ["trt_cp"]
#cell_lines = ["A375"]
cell_lines = ["A375", "A549", "HA1E", "HT29", "MCF7", "PC3"]

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
            output_file = f"dataset2/cp_data_ac/{pert_type}_{cell_line}_filtered_expression_data.tsv"
            filtered_data_df.to_csv(output_file, sep="\t")
            print(f"已保存 {pert_type} - {cell_line} 数据至 {output_file}")
        else:
            print(f"没有找到 {pert_type} - {cell_line} 的数据")

#---测试集
import pandas as pd
import os

# 定义细胞系列表
cell_lines = ['A375', 'A549', 'HA1E', 'HT29', 'MCF7', 'PC3']

# 定义表达数据文件的模板
input_file_template = 'dataset2/cp_data_ac/trt_cp_{}_filtered_expression_data.tsv'

# 定义输出文件的模板
output_file_template = 'dataset2/test_cpdata_ac/trt_cp_{}_filtered_test_data.tsv'

# 读取test_id.txt中的药物名称
with open('dataset/test_id_ac.txt', 'r') as f:
    test_drugs = set(line.strip() for line in f if line.strip())

print(f"共有 {len(test_drugs)} 个测试药物名称。")

# 遍历每个细胞系
for cell in cell_lines:
    input_file = input_file_template.format(cell)
    
    if not os.path.exists(input_file):
        print(f"文件 {input_file} 不存在，跳过。")
        continue
    
    print(f"处理细胞系: {cell}，文件: {input_file}")
    
    # 读取表达数据
    df = pd.read_csv(input_file, sep='\t')

    
    # 获取所有列名（除了第一列基因ID）
    all_columns = df.columns.tolist()
    
    # 筛选出需要的列
    # 首先保留基因ID列
    filtered_columns = [all_columns[0]]
    
    # 遍历其余列，检查药物名称是否在test_drugs中
    for col in all_columns[1:]:
        try:
            # 列名格式: 细胞系:药物名称:剂量:用药时间
            parts = col.split(':')
            if len(parts) < 2:
                print(f"列名 {col} 格式不正确，跳过。")
                continue
            drug_name = parts[1]
            if drug_name in test_drugs:
                filtered_columns.append(col)
        except Exception as e:
            print(f"解析列名 {col} 时出错: {e}")
            continue
    
    print(f"细胞系 {cell} 找到 {len(filtered_columns)-1} 个匹配的药物列。")
    
    # 筛选数据
    filtered_df = df[filtered_columns]
    
    # 保存到新文件
    output_file = output_file_template.format(cell)
    filtered_df.to_csv(output_file, sep='\t', index=False)
    print(f"筛选后的数据已保存到 {output_file}\n")
    


import pandas as pd

# 读取文件的前几行（也可以读取整文件，如果文件很小）
df = pd.read_csv("dataset2/kd_data/trt_sh_A375_core_signatures.tsv", sep="\t")

# 打印列数
print("列数:", len(df.columns))