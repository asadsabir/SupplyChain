import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric_temporal.nn.recurrent import GConvLSTM
from torch_geometric.nn import SAGEConv
from wrangling import G,X_test,X_train,y_test,y_train
from torch_geometric.transforms import ToUndirected

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
        self.edge_weight = nn.Parameter(torch.ones(num_edges,dtype=torch.float))
        
    def forward(self, x, edge_index, hidden_state, cell_state):
        
        h,c = self.recurrent(x, edge_index, self.edge_weight,H=hidden_state,C=cell_state)
        
        y = F.relu(h)
        y = self.sage(y,edge_index)

        return y,h,c
    
model = GLSTM(node_features=49,filters=32,num_edges=86)
model.load_state_dict(torch.load('modelat246.pth'))
model.eval()
undirect = ToUndirected()
batch_size = 4
train_preds = []
test_preds = []
with torch.no_grad():
    H_t = None
    C_t = None
    loss = nn.MSELoss()
    vmse = torch.vmap(loss)
    model.eval()
    mean_mse_te = 0
    for time, snapshot in enumerate(test_iter):
        snapshot = undirect(snapshot)
        pred,H_t,C_t = model(snapshot.x, snapshot.edge_index, H_t,C_t)
        mean_mse_te = mean_mse_te + torch.mean(vmse(pred,snapshot.y))
        test_preds.append(pred)
        if time%batch_size == 0:
            H_t = None
    mean_mse_te = mean_mse_te / (time+1)
    
    
    

    H_t = None
    C_t = None
    mean_mse_tr = 0
    for time, snapshot in enumerate(train_iter):
        snapshot = undirect(snapshot)
        pred,H_t,C_t = model(snapshot.x, snapshot.edge_index, H_t,C_t)
        mean_mse_tr = mean_mse_tr + torch.mean(vmse(pred,snapshot.y))
        train_preds.append(pred)
        if time%batch_size == 0:
            H_t = None
    mean_mse_tr = mean_mse_tr / (time+1)
    mean_mse_tr = mean_mse_tr.item()

train_preds = np.array(train_preds).reshape((1261,36))
test_preds = np.array(test_preds).reshape(840,36)

np.save('train_preds.npy',train_preds)
np.save('test_preds.npy',test_preds)

