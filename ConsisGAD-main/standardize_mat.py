import scipy.io as sio
import numpy as np
import os

# 你的数据集文件名列表
mat_files = [
    "amazon.mat", "dgraphfin.mat", "elliptic.mat", "questions.mat", "reddit.mat",
    "tfinance.mat", "tolokers.mat", "tsocial.mat", "weibo.mat", "yelp.mat"
]

data_dir = "data"

for fname in mat_files:
    path = os.path.join(data_dir, fname)
    print(f"处理 {path} ...")
    mat = sio.loadmat(path)
    keys = mat.keys()
    print(f"原始变量: {keys}")

    # 适配你的实际变量名
    features = mat.get('features', mat.get('Attributes'))
    labels = mat.get('labels', mat.get('Label'))
    # edge_index 适配
    if 'edge_index' in mat:
        edge_index = mat['edge_index']
    elif 'edges' in mat:
        edge_index = mat['edges']
    elif 'Network' in mat:
        net = mat['Network']
        # 稀疏邻接矩阵
        if hasattr(net, 'tocoo'):
            coo = net.tocoo()
            edge_index = np.vstack((coo.row, coo.col))
        else:
            # 稠密邻接矩阵
            edge_index = np.array(np.nonzero(net))
    else:
        raise KeyError('edge_index/edges/Network not found in .mat file')

    # 直接覆盖原文件
    sio.savemat(path, {
        'features': features,
        'labels': labels,
        'edge_index': edge_index
    })
    print(f"已覆盖 {path}")

print("全部处理完成！") 