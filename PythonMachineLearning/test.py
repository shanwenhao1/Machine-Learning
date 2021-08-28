from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn import mixture


def create_data(centers, num=100, std=0.7):
    X, labels_true = make_blobs(n_samples=num, centers=centers, cluster_std=std)
    return X, labels_true


def plot_data(*data):
    X, labels_true = data
    labels = np.unique(labels_true)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = 'rgbycm'
    for i, label in enumerate(labels):
        position = labels_true == label
        show1, show2 = X[position, 0], X[position, 1]
        ax.scatter(X[position, 0], X[position, 1], label="cluster %d" % label),
        color = colors[i % len(colors)]
    ax.legend(loc="best", framealpha=0.5)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[1]")
    ax.set_title("data")
    plt.show()


if __name__ == '__main__':
    centers = centers = [[1,1],[1,2],[2,2],[10,20]]
    y = create_data(centers)
    plot_data(*y)
