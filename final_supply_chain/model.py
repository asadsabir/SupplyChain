import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric_temporal.nn.recurrent import GConvLSTM
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn import SAGEConv
from wrangling import G,X_test,X_train,y_test,y_train


train_iter = StaticGraphTemporalSignal(
    edge_index= np.array(list(G.edges())).transpose(),
    edge_weight=np.ones(len(G.edges)),
    features = X_train,
    targets = y_train.T
)

test_iter = StaticGraphTemporalSignal(
    edge_index= np.array(list(G.edges())).transpose(),
    edge_weight=np.ones(len(G.edges)),
    features = X_test,
    targets = y_test.T
)


class GLSTM(torch.nn.Module):
    def __init__(self, node_features, filters,num_edges):
        super(GLSTM, self).__init__()
        self.recurrent = GConvLSTM(node_features, filters, 3)                           
        self.sage = SAGEConv(filters,1,project=True)
        
        self.edge_weights = nn.Parameter(torch.ones(num_edges,dtype=torch.float))
        self.norm = GraphNorm(filters)
        
    def forward(self, x, edge_index, hidden_state, cell_state):
        
        h,c = self.recurrent(x, edge_index, self.edge_weights,H=hidden_state,C=cell_state)
        y = self.norm(h)
        y = F.relu(y)
        y = F.dropout(y)
        y = self.sage(y,edge_index)
        
        
        return y,h,c
    
model = GLSTM(node_features=21,filters=64,num_edges=43)
model.load_state_dict(torch.load('final_supply_chain/models/modelat472.pth'))
model.eval()

train_preds = []
test_preds = []
with torch.no_grad():
    H_t = None
    C_t = None
    loss = nn.MSELoss()

    model.eval()
    test_cost = 0
    for ttime, tsnapshot in enumerate(test_iter):
        pred,H_t,C_t = model(tsnapshot.x, tsnapshot.edge_index, H_t,C_t)
        test_cost = test_cost + loss(pred,tsnapshot.y)
        test_preds.append(pred)
    test_cost = test_cost / (ttime+1)
    test_mse = test_cost.item()
    
    print(test_mse)

    H_t = None
    C_t = None
    train_cost = 0
    for ttime, tsnapshot in enumerate(train_iter):
        pred,H_t,C_t = model(tsnapshot.x, tsnapshot.edge_index, H_t,C_t)
        train_cost = train_cost + loss(pred,tsnapshot.y)
        train_preds.append(pred)
    train_cost = train_cost / (ttime+1)
    train_mse = train_cost.item()

train_preds = np.array(train_preds).reshape((1261,36))
test_preds = np.array(test_preds).reshape(840,36)
print(train_preds.shape)
print(test_preds.shape)
print(train_mse)
np.save('train_preds.npy',train_preds)
np.save('test_preds.npy',test_preds)

