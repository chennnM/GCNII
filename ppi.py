from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from utils import *
from model import *
import torch.nn as nn
from sklearn.metrics import f1_score
import uuid

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=8000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--wd', type=float, default=0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=9, help='Number of hidden layers.')
parser.add_argument('--hidden', type=int, default=2048, help='Number of hidden layers.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=2000, help='Patience')
parser.add_argument('--data', default='ppi', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha_l')
parser.add_argument('--lamda', type=float, default=1, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)

# Load data
train_adj,val_adj,test_adj,train_feat,val_feat,test_feat,train_labels,val_labels, test_labels,train_nodes, val_nodes, test_nodes = load_ppi()

checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'
print(cudaid,checkpt_file)

model = GCNIIppi(nfeat=train_feat[0].shape[1],
                    nlayers=args.layer,
                    nhidden=args.hidden,
                    nclass=train_labels[0].shape[1],
                    dropout=args.dropout,
                    lamda = args.lamda, 
                    alpha=args.alpha,
                    variant=args.variant).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.wd)


loss_fcn = torch.nn.BCELoss()
# adapted from DGL
def evaluate(feats, model, idx ,subgraph, labels, loss_fcn):
    model.eval()
    with torch.no_grad():
        output = model(feats,subgraph)
        loss_data = loss_fcn(output[:idx], labels[:idx].float())
        predict = np.where(output[:idx].data.cpu().numpy() > 0.5, 1, 0)
        score = f1_score(labels[:idx].data.cpu().numpy(),predict, average='micro')
        return score, loss_data.item()


idx = torch.LongTensor(range(20))
loader = Data.DataLoader(dataset=idx,batch_size=1,shuffle=True,num_workers=0)

def train():
    model.train()
    loss_tra = 0
    acc_tra = 0
    for step,batch in enumerate(loader):
        batch_adj = train_adj[batch[0]].to(device)
        batch_feature = train_feat[batch[0]].to(device)
        batch_label = train_labels[batch[0]].to(device)
        optimizer.zero_grad()
        output = model(batch_feature,batch_adj)
        loss_train = loss_fcn(output[:train_nodes[batch]], batch_label[:train_nodes[batch]])
        loss_train.backward() 
        optimizer.step()
        loss_tra+=loss_train.item()
    loss_tra/=20
    acc_tra/=20
    return loss_tra,acc_tra

def validation():
    loss_val = 0
    acc_val = 0
    for batch in range(2):
        batch_adj = val_adj[batch].to(device)
        batch_feature = val_feat[batch].to(device)
        batch_label = val_labels[batch].to(device)
        score, val_loss = evaluate(batch_feature, model, val_nodes[batch] ,batch_adj, batch_label, loss_fcn)
        loss_val+=val_loss
        acc_val += score
    loss_val/=2
    acc_val/=2
    return loss_val,acc_val

def test():
    model.load_state_dict(torch.load(checkpt_file))
    loss_test = 0
    acc_test = 0
    for batch in range(2):
        batch_adj = test_adj[batch].to(device)
        batch_feature = test_feat[batch].to(device)
        batch_label = test_labels[batch].to(device)
        score,loss =evaluate(batch_feature, model,test_nodes[batch], batch_adj, batch_label, loss_fcn)
        loss_test += loss
        acc_test += score
    acc_test/=2
    loss_test/=2
    return loss_test,acc_test

t_total = time.time()
bad_counter = 0
acc = 0
best_epoch = 0
for epoch in range(args.epochs):
    loss_tra,acc_tra = train()
    loss_val,acc_val = validation()

    if(epoch+1)%1 == 0: 
        print('Epoch:{:04d}'.format(epoch+1),
            'train',
            'loss:{:.3f}'.format(loss_tra),
            '| val',
            'loss:{:.3f}'.format(loss_val),
            'f1:{:.3f}'.format(acc_val*100))
            
    if acc_val > acc:
        acc = acc_val
        best_epoch = epoch
        torch.save(model.state_dict(), checkpt_file)
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

if args.test:
    acc = test()[1]

print("Train cost: {:.4f}s".format(time.time() - t_total))
print('Load {}th epoch'.format(best_epoch))
print("Test" if args.test else "Val","f1.:{:.2f}".format(acc*100))





