from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from models import GAT, SpGAT

import matplotlib.pyplot as plt
import seaborn as sns
import igraph as ig
from sklearn.manifold import TSNE


# import warnings filter
from warnings import simplefilter

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.cuda.empty_cache()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test, edges = load_data()



"""
# igraph visualize nodes
node_label = labels.detach().cpu().numpy()
edge_index = edges
nodeNum = len(node_label)
edge_index_tuples = list(zip(edge_index[0, :], edge_index[1, :]))

ig_graph = ig.Graph()
ig_graph.add_vertices(nodeNum)
ig_graph.add_edges(edge_index_tuples)

# Prepare the visualization settings dictionary
visual_style = {}

# Defines the size of the plot and margins
visual_style["bbox"] = (700, 700)
visual_style["margin"] = 5

edge_weights_raw = np.clip(np.log(np.asarray(ig_graph.edge_betweenness())+1e-16), a_min=0, a_max=None)
edge_weights_raw_normalized = edge_weights_raw / np.max(edge_weights_raw)
edge_weights = [w**6 for w in edge_weights_raw_normalized]
visual_style["edge_width"] = edge_weights

# A simple heuristic for vertex size. Size ~ (degree / 4) (it gave nice results I tried log and sqrt as well)
visual_style["vertex_size"] = [deg / 4 for deg in ig_graph.degree()]

cora_label_to_color_map = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "yellow", 5: "pink", 6: "gray"}

visual_style["vertex_color"] = [cora_label_to_color_map[label] for label in node_label]

# Set the layout - the way the graph is presented on a 2D chart. Graph drawing is a subfield for itself!
# I used "Kamada Kawai" a force-directed method, this family of methods are based on physical system simulation.
# (layout_drl also gave nice results for Cora)
visual_style["layout"] = ig_graph.layout_kamada_kawai()

print('Plotting results ... (it may take couple of seconds).')
ig.plot(ig_graph, **visual_style)
"""

# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
else:
    model = GAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)


train_acc = []
valid_acc = []
train_loss = []
valid_loss = []
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    # record training accuracy
    train_acc.append(acc_train.data.item())
    # record training loss
    train_loss.append(loss_train.data.item())
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    # record validation accuracy
    valid_acc.append(acc_val.data.item())
    # record validation loss
    valid_loss.append(loss_val.data.item())
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


test_acc = []
test_loss = []
def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    # record testing accuracy
    test_acc.append(acc_test.data.item())
    test_loss.append(loss_test.data.item())
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))



# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()


# test attention weights
fig, ax = plt.subplots(4 , 2, figsize=(16, 16))
ax = ax.flatten()
for i, attention in enumerate(model.attentions):
    sns.heatmap(attention.atts.detach().cpu().numpy()[:10, :10],
                annot=True, vmin=0, vmax=1, linewidths=.5, ax=ax[i])
    ax[i].set_title(f'Attention Head {i}')

file_name = 'attention_weights.png'
print("Saving figures %s" % file_name)
fig.savefig(file_name)  # save the figure to file
plt.close(fig)  # close the figure


def plotAccuracy(train_acc, valid_acc):
    # plot accuracy vs epoch
    fig = plt.figure(figsize=(6, 4))
    plt.plot(train_acc, linewidth=1)
    plt.plot(valid_acc, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.title('Train vs Valid Accuracy', fontsize=16)
    #plt.show()

    file_name = 'GAT_accuracy.png'
    print("Saving figures %s" % file_name)
    fig.savefig(file_name)  # save the figure to file
    plt.close(fig)  # close the figure


def plotLoss(train_loss, valid_loss):
    # plot loss vs epoch
    fig = plt.figure(figsize=(6, 4))
    plt.plot(train_loss, linewidth=1)
    plt.plot(valid_loss, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (%)')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.title('Train vs Valid Loss', fontsize=16)
    #plt.show()

    file_name = 'GAT_loss.png'
    print("Saving figures %s" % file_name)
    fig.savefig(file_name)  # save the figure to file
    plt.close(fig)  # close the figure


# plot accuracy and loss
plotAccuracy(train_acc, valid_acc)
plotLoss(train_loss, valid_loss)

### (1) Attention heads visualization
m = model.eval()

# register hook to get intermediate output of GAT
activation = {}


def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


h0 = m.attentions[0].register_forward_hook(getActivation('0'))
h1 = m.attentions[1].register_forward_hook(getActivation('1'))
h2 = m.attentions[2].register_forward_hook(getActivation('2'))
h3 = m.attentions[3].register_forward_hook(getActivation('3'))
h4 = m.attentions[4].register_forward_hook(getActivation('4'))
h5 = m.attentions[5].register_forward_hook(getActivation('5'))
h6 = m.attentions[6].register_forward_hook(getActivation('6'))
h7 = m.attentions[7].register_forward_hook(getActivation('7'))

out = m(features, adj)

h0.remove()
h1.remove()
h2.remove()
h3.remove()
h4.remove()
h5.remove()
h6.remove()
h7.remove()

# visualize attention head
def plot_attention(data, title, sequence):
    fig, ax = plt.subplots(figsize=(6, 4))  # set figure size
    c = ax.pcolor(data, cmap=plt.cm.viridis, alpha=0.9)
    fig.colorbar(c, ax=ax)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title(title, fontsize=16)

    # Save Figure
    file_name = 'attention ' + sequence + ".png"
    print("Saving figures %s" % file_name)
    fig.savefig(file_name)  # save the figure to file
    plt.close(fig)  # close the figure


# plot all trained attention head in the first GAT layer
for key in activation:
    plotTitle = "Attention Head " + key
    plot_attention(activation[key].cpu().numpy()[:, :], plotTitle, key)

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

### (2) T-SNE visualization
# node feature vectors / embeddings visualization
cora_label_to_color_map = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "yellow", 5: "pink", 6: "gray"}

# get the node feature vectors from output of final GAT layer
embeddings = model(features, adj).detach().cpu().numpy()  # shape: (2708, 7)
node_labels = labels.detach().cpu().numpy()
num_classes = len(set(node_labels))

# map the 7-dimensional feature vectors into 2D vectors
t_sne_embeddings = TSNE(n_components=2, perplexity=30, method='barnes_hut').fit_transform(embeddings)

fig, ax = plt.subplots(figsize=(10, 6))
for class_id in range(num_classes):
    # map nodes in the same class with same colour
    ax.scatter(t_sne_embeddings[node_labels == class_id, 0],
               t_sne_embeddings[node_labels == class_id, 1],
               s=20,
               color=cora_label_to_color_map[class_id],
               edgecolors='black',
               linewidths=0.2)

plt.title("T-SNE visualization of GAT node embeddings in cora dataset", fontsize=16)
# Save Figure
file_name = './final_GAT_visual.png'
print("Saving figures %s" % file_name)
fig.savefig(file_name)  # save the figure to file
plt.close(fig)  # close the figure


