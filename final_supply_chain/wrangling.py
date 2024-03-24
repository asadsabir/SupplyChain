import networkx as nx
import scipy.io
import numpy as np
from scipy import sparse


mat = scipy.io.loadmat('final_supply_chain/traffic_dataset.mat')

def sm_to_array(sm):
    return sparse.csr_matrix(sm).toarray()


def convert_to_npm(mat_dict):    
    npm = []

    for i in range(mat_dict.shape[1]):
        sm = mat_dict[:,i][0]
        npm.append(sm_to_array(sm))

    return(np.array(npm))


X_train = convert_to_npm(mat['tra_X_tr'])
X_test = convert_to_npm(mat['tra_X_te'])
y_train = mat['tra_Y_tr']
y_test = mat['tra_Y_te']
adj_mat = mat['tra_adj_mat']


null_count = np.zeros(48)

for time in X_train:
    for f in range(48):
        if sum(X_train[0][:,f]) == 0:
            null_count[f] += 1


null_mask = [True if c == 0 else False for c in null_count]

X_train = X_train[:,:,null_mask]
X_test = X_test[:,:,null_mask]

G = nx.Graph()

G.add_nodes_from(range(36))

for i in range(len(adj_mat)):
    for j in range(i+1,len(adj_mat)):
        if adj_mat[i][j] == 1:
            G.add_edge(i,j)
                