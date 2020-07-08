import argparse
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from layer import GCNIIdenseConv
import math
import numpy as np

class GCNIIdense_model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,dropout,alpha,norm):
        super(GCNIIdense_model, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers):
            self.convs.append(GCNIIdenseConv(hidden_channels, hidden_channels,bias=norm))
        self.convs.append(torch.nn.Linear(hidden_channels,out_channels))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        _hidden = []
        x = F.dropout(x, self.dropout ,training=self.training)
        x = F.relu(self.convs[0](x))
        _hidden.append(x)
        for i,con in enumerate(self.convs[1:-1]):
            x = F.dropout(x, self.dropout ,training=self.training)
            x = F.relu(con(x, edge_index,self.alpha, _hidden[0],edge_weight))+_hidden[-1]
            _hidden.append(x)
        x = F.dropout(x, self.dropout ,training=self.training)
        x = self.convs[-1](x)
        return F.log_softmax(x, dim=1)

def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()

    pred = model(data)[train_idx]

    loss = F.nll_loss(pred, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()



@torch.no_grad()
def test(model, data, y_true,split_idx, evaluator):
    model.eval()

    out = model(data)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (Full-Batch)')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--num_layers', type=int, default=16)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay (L2 loss on parameters).')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=200, help='patience')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha_l')
    parser.add_argument('--norm', default='bn', help='norm layer.')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    data = data.to(device)
    train_idx = split_idx['train'].to(device)
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    Net = GCNIIdense_model
    evaluator = Evaluator(name='ogbn-arxiv')
    acc_list = []
    for run in range(args.runs):
        model = Net(data.x.size(-1), args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout,args.alpha,args.norm).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        bad_counter = 0
        best_val = 0
        final_test_acc = 0
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, data.y,split_idx, evaluator)
            train_acc, valid_acc, test_acc = result
            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
            if valid_acc > best_val:
                best_val = valid_acc
                final_test_acc = test_acc
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == args.patience:
                break
        acc_list.append(final_test_acc*100)
        print(run+1,':',acc_list[-1])
    acc_list=torch.tensor(acc_list)
    print(f'Avg Test: {acc_list.mean():.2f} ± {acc_list.std():.2f}')
    



if __name__ == "__main__":
    main()
