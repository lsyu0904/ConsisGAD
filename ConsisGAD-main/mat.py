import numpy as np
import scipy.io as sio

num_nodes = 1000
num_edges = 5000
feature_dim = 32

features = np.random.rand(num_nodes, feature_dim).astype(np.float32)
labels = np.random.randint(0, 2, size=(num_nodes,)).astype(np.int64)
edge_index = np.random.randint(0, num_nodes, size=(2, num_edges)).astype(np.int64)

sio.savemat('reddit.mat', {
    'features': features,
    'labels': labels,
    'edge_index': edge_index
})
import scipy.io as sio
mat = sio.loadmat('reddit.mat')
print(mat.keys())
# 应该包含 'features', 'labels', 'edge_index' 或 'edges'