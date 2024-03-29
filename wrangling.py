import networkx as nx
import scipy.io
import numpy as np
from scipy import sparse


mat = scipy.io.loadmat('traffic_dataset.mat')

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

G = nx.Graph()

G.add_nodes_from(range(36))

for i in range(len(adj_mat)):
    for j in range(i+1,len(adj_mat)):
        if adj_mat[i][j] == 1:
            G.add_edge(i,j)

                
degree_feature_tr = np.zeros((1261,36))
degree_feature_te = np.zeros((840,36))
for i in range(36):
    degree = G.degree[i]
    degree_feature_tr[:,i] = np.repeat(degree,1261)
    degree_feature_te[:,i] = np.repeat(degree,840)

X_train = np.concatenate((X_train,degree_feature_tr.reshape((1261,36,1))),axis=2)
X_test = np.concatenate((X_test,degree_feature_te.reshape((840,36,1))),axis=2)