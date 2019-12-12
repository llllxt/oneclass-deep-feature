import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name', help='an integer for the accumulator')
args = parser.parse_args()

name = args.name

plt.switch_backend('agg')

train = np.load(name).squeeze()
print("load successfully")
# y = np.load('label1.npy').squeeze()
# test = np.load("feature2.npy").squeeze()

tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
print("before fit")
X_tsne = tsne.fit_transform(train)
print("original data dimension is {}. embeded data dimension is {}".format(train.shape[-1], train.shape[-1]))

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  
plt.figure(figsize=(8, 8))
print(X_norm.shape)
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(0), color=plt.cm.Set1(0), 
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.savefig('vis.png')
