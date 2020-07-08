import os.path as osp
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from layer import GCNIIdenseConv
import math

parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int, default=128, help='Number of layers.')
args = parser.parse_args()

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset)
dataset = Planetoid(path, dataset,split="public", transform = T.NormalizeFeatures())
data = dataset[0]

print(data.train_mask.sum())
print(data.val_mask.sum())
print(data.test_mask.sum())

###################hyperparameters
nlayer = args.layer
dropout = 0.6
alpha = 0.1
lamda = 0.5
hidden_dim = 64
weight_decay1 = 0.01
weight_decay2 = 5e-4
lr = 0.01
patience = 100
#####################

GConv = GCNIIdenseConv


class GCNII_model(torch.nn.Module):
    def __init__(self):
        super(GCNII_model, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(dataset.num_features, hidden_dim))
        for _ in range(nlayer):
            self.convs.append(GConv(hidden_dim, hidden_dim))
        self.convs.append(torch.nn.Linear(hidden_dim,dataset.num_classes))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        _hidden = []
        x = F.dropout(x, dropout ,training=self.training)
        x = F.relu(self.convs[0](x))
        _hidden.append(x)
        for i,con in enumerate(self.convs[1:-1]):
            x = F.dropout(x, dropout ,training=self.training)
            beta = math.log(lamda/(i+1)+1)
            x = F.relu(con(x, edge_index,alpha, _hidden[0],beta,edge_weight))
        x = F.dropout(x, dropout ,training=self.training)
        x = self.convs[-1](x)
        return F.log_softmax(x, dim=1)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model, data = GCNII_model().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=weight_decay1),
    dict(params=model.non_reg_params, weight_decay=weight_decay2)
], lr=lr)

def train():
    model.train()
    optimizer.zero_grad()
    loss_train = F.nll_loss(model()[data.train_mask], data.y[data.train_mask])
    loss_train.backward()
    optimizer.step()
    return loss_train.item()


@torch.no_grad()
def test():
    model.eval()
    logits = model()
    loss_val = F.nll_loss(logits[data.val_mask], data.y[data.val_mask]).item()
    for _, mask in data('test_mask'):
        pred = logits[mask].max(1)[1]
        accs = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return loss_val,accs


best_val_loss = 9999999
test_acc = 0
bad_counter = 0
best_epoch = 0
for epoch in range(1, 1500):
    loss_tra = train()
    loss_val,acc_test_tmp = test()
    if loss_val < best_val_loss:
        best_val_loss = loss_val
        test_acc = acc_test_tmp
        bad_counter = 0
        best_epoch = epoch
    else:
        bad_counter+=1
    if epoch%20 == 0: 
        log = 'Epoch: {:03d}, Train loss: {:.4f}, Val loss: {:.4f}, Test acc: {:.4f}'
        print(log.format(epoch, loss_tra, loss_val, test_acc))
    if bad_counter == patience:
        break
log = 'best Epoch: {:03d}, Val loss: {:.4f}, Test acc: {:.4f}'
print(log.format(best_epoch, best_val_loss, test_acc))
