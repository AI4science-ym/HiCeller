import copy
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
import pandas as pd
# from . import asyn
import pickle
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# os.environ['CUDA_VISIBLE_DEVICES'] = '2，3'

from tqdm import tqdm
import torch
from anndata import AnnData
import scanpy as sc
import scvi
import seaborn as sns
import numpy as np
import wandb

from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from rich.console import Console
from rich.table import Table
from sklearn.metrics import confusion_matrix

sys.path.insert(0, "../")
import hicell as hc
from hicell.model import TransformerModel, AdversarialDiscriminator
from hicell.tokenizer import tokenize_and_pad_batch, random_mask_value
from hicell.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from hicell.tokenizer.gene_tokenizer import GeneVocab
from hicell.preprocess import Preprocessor

from hicell import SubsetsBatchSampler
from hicell.utils import set_seed, category_str2int, eval_scib_metrics
from sklearn.metrics import classification_report

console = Console()
device = torch.cuda.device(0)
os.environ['WANDB_MODE'] = 'disabled'
os.environ["WANDB_DISABLED"] = "true"
sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',
                    default='/gpfstest/tako/singlecell/dataset/zeng2/synecosys_datasets/dataset_process_batch_1_2/HD221101007_fig_1b_cat.h5ad')
parser.add_argument('--model_path', default='/tako/scGPT/save/dev_scGPT-Mar20-09-41')
# parser.add_argument('--data_path2', default='/gpfstest/tako/hxf/scGPT/peplung2.h5ad')
args = parser.parse_args()
hyperparameter_defaults = dict(
    seed=0,
    dataset_name="hicell",
    # dataset_name="ms",
    do_train=False,
    load_model=args.model_path,
    mask_ratio=0.0,
    epochs=10,
    n_bins=51,
    MVC=False,  # Masked value prediction for cell embedding
    ecs_thres=0.0,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=0.0,
    lr=1e-4,
    batch_size=32,
    layer_size=128,
    nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead=4,  # number of heads in nn.MultiheadAttention
    dropout=0.2,  # dropout probability
    schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
    save_eval_interval=5,
    fast_transformer=True,
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision
    include_zero_gene=False,
    freeze=False,  # freeze
    DSBN=False,  # Domain-spec batchnorm
)
table = Table(title="训练参数")
table.add_column("参数", justify="right", style="cyan", no_wrap=True)
table.add_column("值", style="magenta")
for key, value in hyperparameter_defaults.items():
    table.add_row(key, str(value))
console.print(table)
run = wandb.init(
    config=hyperparameter_defaults,
    project="hicell",
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
)
config = wandb.config
set_seed(config.seed)

# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = "auto"  # for masked values, now it should always be auto

include_zero_gene = config.include_zero_gene  # if True, include zero genes among hvgs in the training
max_seq_len = 3001
n_bins = config.n_bins

# input/output representation
input_style = "binned"  # "normed_raw", "log1p", or "binned"
output_style = "binned"  # "normed_raw", "log1p", or "binned"

# settings for training
MLM = False  # whether to use masked language modeling, currently it is always on.
CLS = True  # celltype classification objective
ADV = False  # Adversarial training for batch correction
CCE = False  # Contrastive cell embedding objective
MVC = config.MVC  # Masked value prediction for cell embedding
ECS = config.ecs_thres > 0  # Elastic cell similarity objective
DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"
ecs_threshold = config.ecs_thres
dab_weight = config.dab_weight

explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training

per_seq_batch_sample = False

# settings for optimizer
lr = config.lr  # TODO: test learning rate ratio between two tasks
lr_ADV = 1e-3  # learning rate for discriminator, used when ADV is True
batch_size = config.batch_size
eval_batch_size = config.batch_size
epochs = config.epochs
schedule_interval = 1

# settings for the model
fast_transformer = config.fast_transformer
fast_transformer_backend = "flash"  # "linear" or "flash"
embsize = config.layer_size  # embedding dimension
d_hid = config.layer_size  # dimension of the feedforward network in TransformerEncoder
nlayers = config.nlayers  # number of TransformerEncoderLayer in TransformerEncoder
nhead = config.nhead  # number of heads in nn.MultiheadAttention
dropout = config.dropout  # dropout probability

# logging
log_interval = 100  # iterations
save_eval_interval = config.save_eval_interval  # epochs
do_eval_scib_metrics = True

assert input_style in ["normed_raw", "log1p", "binned"]
assert output_style in ["normed_raw", "log1p", "binned"]
assert input_emb_style in ["category", "continuous", "scaling"]
if input_style == "binned":
    if input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif input_style == "log1p" or input_style == "normed_raw":
    if input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw input."
        )

if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins  # for padding gene expr values
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins

if ADV and DAB:
    raise ValueError("ADV and DAB cannot be both True.")
DAB_separate_optim = True if DAB > 1 else False

dataset_name = config.dataset_name
save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"结果保存至 {save_dir}")
import hicell
logger = hicell.logger
hc.utils.add_file_handler(logger, save_dir / "run.log")


cell_type_key = "annot_sub"
adata_test = sc.read(args.data_path)
clusters_to_remove = ['Doublet', 'Low quality']
adata_test = adata_test[~adata_test.obs[cell_type_key].isin(clusters_to_remove), :]
adata_test.obs["celltype"] = adata_test.obs[cell_type_key].astype("category")
# print(adata_test.obs["celltype"])
actual_counts = adata_test.obs["celltype"].value_counts().to_dict()
# 你提供的类别对应整数字典
type_to_id = {0: 'Astrocytes', 1: 'CD4+ naive T cells', 2: 'CD8+ effector T cells', 3: 'Capillary endothelial cells',
              4: 'Doublet', 5: 'Endothelial cells', 6: 'Epithelial cells_Cancer cells', 7: 'Excitatory neurons',
              8: 'Fibroblasts', 9: 'Glial cells_Cancer cells', 10: 'Inhibitory neurons', 11: 'Macrophages',
              12: 'Melanocytes_Cancer cells', 13: 'Microglial cells', 14: 'Monocytes', 15: 'Mural cells',
              16: 'Naive T cells', 17: 'Natural killer cells', 18: 'Neurons', 19: 'Neurons_Cancer cells',
              20: 'Oligodendrocyte precursor cells', 21: 'Oligodendrocytes',
              22: 'Unassigned'}  # 将字典的键值对反转，以便可以通过类别名称查找对应的整数编码
type_to_id_or = {0: 'Arterial endothelial cells', 1: 'B cells', 2: 'CD4+ effector T cells', 3: 'CD4+ exhausted T cells',
                 4: 'CD4+ naive T cells', 5: 'CD8+ effector T cells', 6: 'CD8+ effector memory T cells',
                 7: 'CD8+ exhausted T cells', 8: 'CD8+ memory T cells', 9: 'CD8+ mucosal-associated invariant T cells',
                 10: 'CD8+ naive T cells', 11: 'CD8+ tissue-resident memory T cells', 12: 'Capillary endothelial cells',
                 13: 'Conventional dendritic cells', 14: 'Doublet', 15: 'Endothelial cells',
                 16: 'Endothelial tip cells', 17: 'Epithelial cells_Cancer cells', 18: 'Exhausted T cells',
                 19: 'Fibroblasts', 20: 'Gamma delta T cells', 21: 'Hair follicle cells', 22: 'Keratinocytes',
                 23: 'Langerhans cells', 24: 'Lymphatic endothelial cells', 25: 'Macrophages', 26: 'Mast cells',
                 27: 'Melanocytes', 28: 'Melanocytes_Cancer cells', 29: 'Memory B cells', 30: 'Monocytes',
                 31: 'Mural cells', 32: 'Naive B cells', 33: 'Naive T cells', 34: 'Natural killer cells',
                 35: 'Plasma cells', 36: 'Plasmacytoid dendritic cells', 37: 'Proliferating T cells',
                 38: 'Proliferating cells', 39: 'Regulatory T cells', 40: 'Schwann cells', 41: 'Sweat gland cells',
                 42: 'T cells_Cancer cells', 43: 'T helper cells', 44: 'T-helper 2 cells', 45: 'Unassigned',
                 46: 'Venous endothelial cells'}  # 将字典的键值对反转，以便可以通过类别名称查找对应的整数编码
# type_to_id = {0: 'Acinar cells', 1: 'Alveolar type I cells', 2: 'Alveolar type II cells', 3: 'Astrocytes', 4: 'B cells', 5: 'B cells_Cancer cells', 6: 'CD4+ effector T cells', 7: 'CD4+ effector memory T cells', 8: 'CD4+ naive T cells', 9: 'CD8+ effector T cells', 10: 'CD8+ effector memory T cells', 11: 'CD8+ exhausted T cells', 12: 'CD8+ memory T cells', 13: 'CD8+ naive T cells', 14: 'CD8+ tissue-resident memory T cells', 15: 'Capillary endothelial cells', 16: 'Ciliated cells', 17: 'Club cells', 18: 'Conventional dendritic cells', 19: 'Doublet', 20: 'Ductal cells', 21: 'Endothelial cells', 22: 'Ependymal cells', 23: 'Epithelial cells_Cancer cells', 24: 'Erythrocytes', 25: 'Exhausted T cells', 26: 'Fibroblasts', 27: 'Follicular helper T cells', 28: 'Glial cells_Cancer cells', 29: 'Macrophages', 30: 'Mast cells', 31: 'Melanocytes_Cancer cells', 32: 'Microglial cells', 33: 'Monocytes', 34: 'Mural cells', 35: 'Myocytes_Cancer cells', 36: 'Naive T cells', 37: 'Natural killer cells', 38: 'Neurons', 39: 'Neurons_Cancer cells', 40: 'Neutrophils', 41: 'Oligodendrocyte precursor cells', 42: 'Oligodendrocytes', 43: 'Pancreatic beta cells', 44: 'Plasma cells', 45: 'Plasmacytoid dendritic cells', 46: 'Platelets', 47: 'Proliferating T cells', 48: 'Proliferating cells', 49: 'Proliferating lymphocytes', 50: 'Proliferating macrophages', 51: 'Regulatory T cells', 52: 'Schwann cells', 53: 'T helper cells', 54: 'Unassigned', 55: 'Venous endothelial cells', 56: 'buqueding cells'}
id_to_type = {v: k for k, v in type_to_id.items()}
# 为adata_test.obs["celltype"]中的每个类别名称查找对应的整数编码
adata_test.obs["celltype_id"] = adata_test.obs["celltype"].apply(lambda x: id_to_type.get(x, 62))

# 打印结果查看
# print(adata_test.obs["celltype_id"])
# print("=======================")

data_is_raw = False
filter_gene_by_counts = False
adata_test_raw = adata_test.copy()
celltype_id_labels = adata_test.obs["celltype_id"].values
# celltype_id_labels = adata_test.obs["celltype"].astype("category").cat.codes.values
# print("=======================")
# print(celltype_id_labels)
celltypes = adata_test.obs["celltype"].unique()
# print("=======================")
# print(celltypes)
celltypes_list = list(celltypes)
# print("=======================")
# print(celltypes_list)
# num_types = len(np.unique(celltype_id_labels))
num_types = 23
# print("=======================")
# print(num_types)
id2type = {v: k for k, v in id_to_type.items()}
# id2type = dict(enumerate(adata_test.obs["celltype"].astype("category").cat.categories))
# print("=======================")
# print(id2type)
adata_test.obs["celltype_id"] = celltype_id_labels
# print("=======================")
# print(adata_test.obs["celltype_id"])
adata_test.var["gene_name"] = adata_test.var.index.tolist()
adata_test.obs["batch_id"] = 1

# 第一步：读取测试集数据
# adata_test = sc.read(args.data_path)
#
# # 对测试集数据进行一些处理（保持原有逻辑）
# adata_test.obs["batch_id"] = 1  # 添加 batch_id 字段
# adata_test.var["gene_name"] = adata_test.var.index.tolist()  # 添加 gene_name 字段
#
# # 创建一个原始数据的副本
# adata_test_raw = adata_test.copy()
#
# # 第二步：读取类别文件
# adata_labels = sc.read(args.data_path2)
#
# # 假设类别文件中包含了对应 celltype 信息
# # 在这里，我们假设类别信息也存储在 `.obs` 中的某个列中，比如 'celltype'
# adata_test.obs["celltype"] = adata_labels.obs["annot_sub2"].astype("category")
# data_is_raw = False
# filter_gene_by_counts = False
# # 将 celltype 转化为数值编码
# celltype_id_labels = adata_test.obs["celltype"].astype("category").cat.codes.values
# adata_test.obs["celltype_id"] = celltype_id_labels
#
# # 获取类别的唯一值和映射关系
# celltypes = adata_test.obs["celltype"].unique()
# num_types = len(np.unique(celltype_id_labels))
# id2type = dict(enumerate(adata_test.obs["celltype"].astype("category").cat.categories))
#
# # 输出/接下来处理
# print(f"Number of types: {num_types}")
# print(f"Cell types: {celltypes}")
# print("ID to type mapping:", id2type)
"""
加载模型
从指定路径加载模型配置文件、模型文件和词汇表文件。
加载词汇表并添加特殊标记（如果不存在）。
将数据集中的基因名称与词汇表匹配，并记录匹配结果。
输出匹配基因数量的日志信息。
读取模型配置文件，并提取模型参数。
"""
if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = "/gpfstest/tako/gina/hicell/human/args.json"
    # model_file = model_dir / "best_model.pt"
    model_file = model_dir / 'model.pt'
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    shutil.copy(vocab_file, save_dir / "vocab.json")
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    gene_name_list = []
    for gene in vocab.get_stoi().keys():
        gene_name_list.append(gene)
    # 初始化 id_in_vocab 列
    adata_test.var["id_in_vocab"] = [-1] * len(adata_test.var)

    for idx, gene in enumerate(adata_test.var["gene_name"]):
        s = gene.upper()  # 将基因名称转换为大写
        if s in gene_name_list:
            adata_test.var["id_in_vocab"][idx] = 1  # 正确的索引位置赋值为 1
        else:
            adata_test.var["id_in_vocab"][idx] = -1  # 正确的索引位置赋值为 -1

    # adata_test.var["id_in_vocab"] = [
    #     1 if gene in vocab else -1 for gene in adata_test.var["gene_name"]
    # ]
    gene_ids_in_vocab = np.array(adata_test.var["id_in_vocab"])
    adata_test = adata_test[:, adata_test.var["id_in_vocab"] >= 0]

    # model
    with open("/gpfstest/tako/gina/hicell/human/args.json", "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"模型加载路径： {model_file},  "
        f"配置文件路径： {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=filter_gene_by_counts,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)

# preprocessor(adata, batch_key=None)
preprocessor(adata_test, batch_key=None)

input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
    "normed_raw": "X_normed",
    "log1p": "X_normed",
    "binned": "X_binned",
}[input_style]

genes = adata_test.var["gene_name"].tolist()

celltypes_labels = adata_test.obs["celltype_id"].tolist()  # make sure count from 0
celltypes_labels = np.array(celltypes_labels)

if config.load_model is None:
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)


# tokenized_valid = tokenize_and_pad_batch(
#     all_counts,
#     gene_ids,
#     max_len=max_seq_len,
#     vocab=vocab,
#     pad_token=pad_token,
#     pad_value=pad_value,
#     append_cls=True,
#     include_zero_gene=include_zero_gene,
# )


class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=3,
    n_cls=num_types if CLS else 1,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=MVC,
    do_dab=DAB,
    use_batch_labels=INPUT_BATCH_LABELS,
    num_batch_labels=1,
    domain_spec_batchnorm=config.DSBN,
    input_emb_style=input_emb_style,
    n_input_bins=n_input_bins,
    cell_emb_style=cell_emb_style,
    mvc_decoder_style=mvc_decoder_style,
    ecs_threshold=ecs_threshold,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=fast_transformer,
    fast_transformer_backend=fast_transformer_backend,
    pre_norm=config.pre_norm,
)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is not None:
            assert isinstance(alpha, (float, list, torch.Tensor))
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        loss = F.nll_loss((1 - pt) ** self.gamma * logpt, targets, reduction='none')

        if self.alpha is not None:
            if isinstance(self.alpha, float):
                alpha = torch.tensor([self.alpha] * inputs.size(1), dtype=loss.dtype, device=loss.device)
            else:
                alpha = self.alpha
            loss *= alpha[targets]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


if config.load_model is not None:
    try:
        # model.load_state_dict(torch.load(model_file))
        state_dict = torch.load(model_file)
        # 检查 state_dict 的 key，判断是否需要移除 "module." 前缀
        if any(key.startswith("module.") for key in state_dict.keys()):
            # 如果是分布式训练保存的模型，移除 "module." 前缀
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_key = k.replace("module.", "")  # 去掉 "module." 前缀
                new_state_dict[new_key] = v
            state_dict = new_state_dict

        # 加载到模型
        model.load_state_dict(state_dict)
        logger.info(f"正在从 {model_file}加载参数")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        # for k, v in pretrained_dict.items():
        #     logger.info(f"正在加载参数 {k} 其形状为 {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

model.to(device)
wandb.watch(model)

if ADV:
    discriminator = AdversarialDiscriminator(
        d_model=embsize,
        n_cls=1,
    ).to(device)

criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
criterion_fl = FocalLoss(alpha=0.25, gamma=2.0)

criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=lr, eps=1e-4 if config.amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, schedule_interval, gamma=config.schedule_ratio
)
if DAB_separate_optim:
    optimizer_dab = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_dab = torch.optim.lr_scheduler.StepLR(
        optimizer_dab, schedule_interval, gamma=config.schedule_ratio
    )
if ADV:
    criterion_adv = nn.CrossEntropyLoss()  # consider using label smoothing
    optimizer_E = torch.optim.Adam(model.parameters(), lr=lr_ADV)
    scheduler_E = torch.optim.lr_scheduler.StepLR(
        optimizer_E, schedule_interval, gamma=config.schedule_ratio
    )
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_ADV)
    scheduler_D = torch.optim.lr_scheduler.StepLR(
        optimizer_D, schedule_interval, gamma=config.schedule_ratio
    )

scaler = torch.cuda.amp.GradScaler(enabled=config.amp)


def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


# def evaluate(model: nn.Module, loader: DataLoader, return_raw: bool = False) -> Tuple[float, float, np.ndarray]:
#     """
#     Evaluate the model on the evaluation data.

#     Args:
#         model (nn.Module): 要评估的模型
#         loader (DataLoader): 数据加载器
#         return_raw (bool): 是否返回原始预测结果

#     Returns:
#         Tuple[float, float, np.ndarray]: (平均损失, 平均错误率, 每个样本的预测概率分布)
#     """
#     model.eval()
#     total_loss = 0.0
#     total_error = 0.0
#     total_num = 0
#     probabilities = []  # 保存每个样本的预测概率分布
#     predictions = []  # 保存最终的预测类别

#     with torch.no_grad():
#         for batch_data in tqdm(loader):
#             # 加载输入数据
#             input_gene_ids = batch_data["gene_ids"].to(device)
#             input_values = batch_data["values"].to(device)
#             target_values = batch_data["target_values"].to(device)
#             celltype_labels = batch_data["celltype_labels"].to(device)

#             # 构造 mask
#             src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])

#             # 通过模型进行推理
#             with torch.cuda.amp.autocast(enabled=config.amp):
#                 output_dict = model(
#                     input_gene_ids,
#                     input_values,
#                     src_key_padding_mask=src_key_padding_mask,
#                     batch_labels=1 if INPUT_BATCH_LABELS or config.DSBN else None,
#                     CLS=CLS,  # evaluation does not need CLS or CCE
#                     CCE=False,
#                     MVC=False,
#                     ECS=False,
#                     do_sample=do_sample_in_train,
#                 )
#                 output_values = output_dict["cls_output"]  # 模型输出 logits

#             # 计算 softmax 概率
#             probs = torch.softmax(output_values, dim=-1).cpu().numpy()
#             probabilities.append(probs)

#             # 根据概率分布获取预测类别
#             preds = probs.argmax(axis=1)
#             predictions.append(preds)

#             # 计算错误率
#             accuracy = (preds == celltype_labels.cpu().numpy()).sum()
#             total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
#             total_num += len(input_gene_ids)

#     # 将所有 batch 的概率分布拼接到一起
#     probabilities = np.concatenate(probabilities, axis=0)
#     predictions = np.concatenate(predictions, axis=0)

#     if return_raw:
#         return total_loss / total_num, total_error / total_num, probabilities

#     return total_loss / total_num, total_error / total_num

# def predict(model: nn.Module, adata: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     对数据进行预测，返回每个样本的预测类别、每个类别的置信度以及真实标签。

#     Args:
#         model (nn.Module): 训练好的模型
#         adata (DataLoader): 测试数据

#     Returns:
#         Tuple[np.ndarray, np.ndarray, np.ndarray]: (预测类别, 每个类别的置信度, 真实标签)
#     """
#     all_counts = (
#         adata.layers[input_layer_key].A
#         if issparse(adata.layers[input_layer_key])
#         else adata.layers[input_layer_key]
#     )

#     celltypes_labels = adata.obs["celltype_id"].tolist()  # 确保从0开始计数
#     celltypes_labels = np.array(celltypes_labels)

#     batch_ids = adata.obs["batch_id"].tolist()
#     batch_ids = np.array(batch_ids)

#     tokenized_test = tokenize_and_pad_batch(
#         all_counts,
#         gene_ids,
#         max_len=max_seq_len,
#         vocab=vocab,
#         pad_token=pad_token,
#         pad_value=pad_value,
#         append_cls=True,  # append <cls> token at the beginning
#         include_zero_gene=include_zero_gene,
#     )
#     logger.info(
#         f"验证集样本数量: {tokenized_test['genes'].shape[0]}, "
#         f"\n\t特征长度: {tokenized_test['genes'].shape[1]}"
#     )
#     input_values_test = random_mask_value(
#         tokenized_test["values"],
#         mask_ratio=mask_ratio,
#         mask_value=mask_value,
#         pad_value=pad_value,
#     )

#     test_data_pt = {
#         "gene_ids": tokenized_test["genes"],
#         "values": input_values_test,
#         "target_values": tokenized_test["values"],
#         "batch_labels": torch.from_numpy(batch_ids).long(),
#         "celltype_labels": torch.from_numpy(celltypes_labels).long(),
#     }

#     test_loader = DataLoader(
#         dataset=SeqDataset(test_data_pt),
#         batch_size=eval_batch_size,
#         shuffle=False,
#         drop_last=False,
#         num_workers=min(len(os.sched_getaffinity(0)), eval_batch_size // 2),
#         pin_memory=True,
#     )

#     model.eval()
#     _, _, probabilities = evaluate(
#         model,
#         loader=test_loader,
#         return_raw=True,
#     )

#     # 从预测概率分布中获取最终类别
#     predictions = probabilities.argmax(axis=1)
#     scores = np.round(probabilities * 100, 2)
#     np.set_printoptions(formatter={"float_kind": lambda x: f"{x:.2f}"})
#     # 返回 predictions, probabilities, 和真实标签
#     output_file = "/gpfstest/tako/singlecell/dataset/ylm/output.txt"
#     with open(output_file, "w") as f:
#         for i in range(len(predictions)):
#             # 打印到控制台
#             print(f"预测类别: {predictions[i]}")
#             print(f"预测置信度: {scores[i]}")
#             print(f"真实标签: {celltypes_labels[i]}")

#             # 写入到文件
#             f.write(f"预测类别: {predictions[i]}\n")
#             f.write(f"预测置信度: {scores[i]}\n")
#             f.write(f"真实标签: {celltypes_labels[i]}\n")
#             f.write("\n")  # 添加空行分隔内容
#     return predictions, scores, celltypes_labels


def evaluate(model: nn.Module, loader: DataLoader, return_raw: bool = False) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    predictions = []
    with torch.no_grad():
        for batch_data in tqdm(loader):
            input_gene_ids = batch_data["gene_ids"].to(device)

            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            # batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=1 if INPUT_BATCH_LABELS or config.DSBN else None,
                    CLS=CLS,  # evaluation does not need CLS or CCE
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=do_sample_in_train,
                    # generative_training = False,
                )
                output_values = output_dict["cls_output"]
                # loss = criterion_cls(output_values, celltype_labels)
            # total_loss += loss.item() * len(input_gene_ids)
            accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
            total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
            # total_dab += loss_dab.item() * len(input_gene_ids) if DAB else 0.0
            total_num += len(input_gene_ids)
            preds = output_values.argmax(1).cpu().numpy()
            predictions.append(preds)

    if return_raw:
        return np.concatenate(predictions, axis=0)

    return total_loss / total_num, total_error / total_num


best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None
define_wandb_metrcis()


def predict(model: nn.Module, adata: DataLoader) -> float:
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
    # print(celltypes_labels)
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    tokenized_test = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=include_zero_gene,
    )
    logger.info(
        f"验证集样本数量: {tokenized_test['genes'].shape[0]}, "
        f"\n\t特征长度: {tokenized_test['genes'].shape[1]}"
    )
    input_values_test = random_mask_value(
        tokenized_test["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )

    test_data_pt = {
        "gene_ids": tokenized_test["genes"],
        "values": input_values_test,
        "target_values": tokenized_test["values"],
        "batch_labels": torch.from_numpy(batch_ids).long(),
        "celltype_labels": torch.from_numpy(celltypes_labels).long(),
    }

    test_loader = DataLoader(
        dataset=SeqDataset(test_data_pt),
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), eval_batch_size // 2),
        pin_memory=True,
    )

    model.eval()
    predictions = evaluate(
        model,
        loader=test_loader,
        return_raw=True,
    )

    ###################################################################################################################################

    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    cm = confusion_matrix(celltypes_labels, predictions, labels=list(type_to_id.keys()))
    plt.figure(figsize=(20, 16))
    confusion_matrix_image = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=type_to_id.values(),
                                         yticklabels=type_to_id.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show(block=True)
    plt.savefig(save_dir / "confusion_matrix.png", dpi=600)
    import sklearn
    acc = sklearn.metrics.accuracy_score(celltypes_labels, predictions)
    from sklearn.metrics import classification_report
    # unique_labels = np.unique(celltypes_labels)
    report = classification_report(celltypes_labels, predictions, target_names=list(type_to_id.values()),
                                   labels=list(type_to_id.keys()))
    class_report = classification_report(celltypes_labels, predictions, target_names=list(type_to_id.values()),
                                         labels=list(type_to_id.keys()), output_dict=True)
    # 将class_report转为excel
    class_report = pd.DataFrame(class_report).transpose()
    class_report.to_excel(save_dir / "/classification_report.xlsx")
    print(acc)
    print(report)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(celltypes_labels, predictions)
    precision = precision_score(celltypes_labels, predictions, average="macro")
    # recall = recall_score(celltypes_labels, predictions, average="macro")
    # macro_f1 = f1_score(celltypes_labels, predictions, average="macro")

    logger.info(
        f"Accuracy: {accuracy:.3f}, "
        f"Precision: {precision:.3f}, "
        # f"Recall: {recall:.3f}, "
        # f"Macro F1: {macro_f1:.3f}"
    )

    results = {
        "test/accuracy": accuracy,
        "test/precision": precision,
        # "test/recall": recall,
        # "test/macro_f1": macro_f1,
    }

    return predictions, celltypes_labels, results


# predictions, labels, results = predict(model, adata_test)
# adata_test_raw.obs["predictions"] = [id2type[p] for p in predictions]
predictions, results, labels = predict(model, adata_test)
adata_test_raw.obs["predictions"] = [id2type[p] for p in predictions]

import scanpy as sc
import matplotlib.pyplot as plt


def set_plt_fontsize(fontsize: int = 10):
    plt.rc('axes', titlesize=fontsize)  # Controls Axes Title
    plt.rc('axes', labelsize=fontsize)  # Controls Axes Labels
    plt.rc('xtick', labelsize=fontsize)  # Controls x Tick Labels
    plt.rc('ytick', labelsize=fontsize)  # Controls y Tick Labels
    plt.rc('legend', fontsize=fontsize)  # Controls Legend Font
    plt.rc('figure', titlesize=fontsize)


def umap_tsne_plot(adata: Path,
                   result_plot_file: Path = save_dir,
                   annotation_mark: str = "predictions",
                   plot_basis: str = "X_umap"):
    set_plt_fontsize(10)
    with plt.rc_context({"figure.figsize": (6, 4), "figure.dpi": (300)}):
        palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"] + plt.rcParams["axes.prop_cycle"].by_key()[
            "color"] + plt.rcParams["axes.prop_cycle"].by_key()["color"]
        palette_ = {c: palette_[i] for i, c in enumerate(adata.obs["predictions"].cat.categories)}
        sc.pl.umap(
            adata,
            color=annotation_mark,
            palette=palette_,
            show=True
        )
    plt.savefig(result_plot_file / "umap.png", bbox_inches="tight", dpi=600)


def find_rank_genes(adata,
                    annotation_mark: str = "predictions",
                    gene_show: int = 5,
                    method: str = "wilcoxon",  # t-test, t-test_overestim_var, logreg,
                    logfc_threshold: float = 0.25,
                    pval_threshold: float = 0.01,
                    min_pct: float = 0.25,
                    result_plot_file_prefix: str = save_dir):
    # adata = Path(f"{adata}").absolute()
    # calculate marker genes
    sc.tl.rank_genes_groups(adata, method=method, groupby=annotation_mark, key_added=f"{annotation_mark}_rank_genes",
                            pts=True)
    sc.tl.filter_rank_genes_groups(adata, key=f"{annotation_mark}_rank_genes",
                                   key_added=f"{annotation_mark}_rank_genes_filter", min_fold_change=logfc_threshold,
                                   min_in_group_fraction=min_pct)
    # output: adata.uns['rank_genes_r0.1']
    cluster_num = len(set(adata.obs[annotation_mark].tolist()))
    cluster_top_genes: List[str] = []
    for i in adata.obs[annotation_mark].unique():
        temp_df = sc.get.rank_genes_groups_df(adata, group=f"{i}", key=f"{annotation_mark}_rank_genes_filter")
        temp_top_gene = temp_df[temp_df['pvals'] < pval_threshold].head(gene_show)["names"].tolist()
        cluster_top_genes.extend(temp_top_gene)

    # dot plot
    sc.pl.rank_genes_groups_dotplot(adata, groupby=annotation_mark, standard_scale="var", n_genes=gene_show,
                                    key=f"{annotation_mark}_rank_genes_filter", min_logfoldchange=logfc_threshold)
    plt.savefig(result_plot_file_prefix / "dotplot.png", bbox_inches="tight", dpi=600)
    # plt.savefig(Path(f"{result_plot_file_prefix}_dotplot.png").absolute().with_suffix(".pdf"), bbox_inches="tight")

    # heatmap
    sc.pl.rank_genes_groups_heatmap(adata, key=f"{annotation_mark}_rank_genes_filter", show_gene_labels=True,
                                    n_genes=gene_show, min_logfoldchange=logfc_threshold, cmap="plasma")
    plt.savefig(result_plot_file_prefix / "heatmap.png", bbox_inches="tight", dpi=600)
    # plt.savefig(Path(f"{result_plot_file_prefix}_heatmap.png").absolute().with_suffix(".pdf"), bbox_inches="tight")

    # violin plot
    violin_plot = sc.pl.stacked_violin(adata, cluster_top_genes, groupby=annotation_mark, return_fig='True')
    violin_plot.add_totals().style(ylim=(0, 5)).show()
    violin_plot.savefig(result_plot_file_prefix / "violin.png", bbox_inches="tight", dpi=600)
    # violin_plot.savefig(Path(f"{result_plot_file_prefix}_violin.png").absolute().with_suffix(".pdf"), bbox_inches="tight")


umap_tsne_plot(adata_test_raw)
find_rank_genes(adata_test_raw)
