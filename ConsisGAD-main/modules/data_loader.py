import torch
import dgl
import os
import numpy as np
from torch.utils.data import DataLoader as torch_dataloader
from sklearn.model_selection import train_test_split
import scipy.io as sio

def get_dataset(name: str, raw_dir: str, to_homo: bool=False, random_state: int=717):
    mat_path = os.path.join(raw_dir, f"{name}.mat")
    print(f"正在读取的mat文件路径: {mat_path}")
    mat = sio.loadmat(mat_path)
    print(f"mat文件包含的变量: {mat.keys()}")
    features = torch.tensor(mat['features'], dtype=torch.float32)
    labels = torch.tensor(mat['labels']).squeeze().long() if 'labels' in mat else None
    edge_index = mat['edge_index'] if 'edge_index' in mat else mat.get('edges', None)
    if edge_index is None:
        raise ValueError(f"{name}.mat 缺少 edge_index 或 edges 变量")
    edge_index = torch.tensor(edge_index, dtype=torch.int64)
    if edge_index.shape[0] != 2:
        edge_index = edge_index.T
    graph = dgl.graph((edge_index[0], edge_index[1]))
    graph.ndata['feature'] = features
    if labels is not None:
        graph.ndata['label'] = labels
    return graph


def get_index_loader_test(name: str, batch_size: int, unlabel_ratio: int=1, training_ratio: float=-1,
                             shuffle_train: bool=True, to_homo:bool=False):
    graph = get_dataset(name, 'data/', to_homo=to_homo, random_state=7537)
    index = np.arange(graph.num_nodes())
    labels = graph.ndata['label']
    train_nids, valid_test_nids = train_test_split(index, stratify=labels[index],
                                                   train_size=training_ratio/100., random_state=2, shuffle=True)
    valid_nids, test_nids = train_test_split(valid_test_nids, stratify=labels[valid_test_nids],
                                             test_size=0.67, random_state=2, shuffle=True)
    train_mask = torch.zeros_like(labels).bool()
    val_mask = torch.zeros_like(labels).bool()
    test_mask = torch.zeros_like(labels).bool()
    train_mask[train_nids] = 1
    val_mask[valid_nids] = 1
    test_mask[test_nids] = 1
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    labeled_nids = train_nids
    unlabeled_nids = np.concatenate([valid_nids, test_nids, train_nids])
    power = 10 if name == 'tfinance' else 16
    valid_nids = torch.tensor(valid_nids, dtype=torch.int64)
    test_nids = torch.tensor(test_nids, dtype=torch.int64)
    labeled_nids = torch.tensor(labeled_nids, dtype=torch.int64)
    unlabeled_nids = torch.tensor(unlabeled_nids, dtype=torch.int64)
    valid_loader = torch_dataloader(valid_nids, batch_size=2**power, shuffle=False, drop_last=False, num_workers=4)
    test_loader = torch_dataloader(test_nids, batch_size=2**power, shuffle=False, drop_last=False, num_workers=4)
    labeled_loader = torch_dataloader(labeled_nids, batch_size=batch_size, shuffle=shuffle_train, drop_last=True, num_workers=0)
    unlabeled_loader = torch_dataloader(unlabeled_nids, batch_size=batch_size * unlabel_ratio, shuffle=shuffle_train, drop_last=True, num_workers=0)
    return graph, labeled_loader, valid_loader, test_loader, unlabeled_loader

    