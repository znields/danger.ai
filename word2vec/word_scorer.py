import numpy as np
from os import listdir
from os.path import isfile, join


def calculate_score(W, D):
    total_distance = 0
    for i in range(D.shape[1]):
        diff = W + np.random.rand(250, 1) * 0.1 - D[:, i].reshape(250, 1)
        total_distance += np.sqrt(np.sum(diff ** 2))
    return 1 / total_distance


paths = [f for f in listdir('data') if isfile(join('data', f))]
D = np.load('./d.npy')

if __name__ == '__main__':
    for path in paths:
        scores = np.zeros(72)
        W = np.load('data/' + path)
        print(path)
        path = path[:-4]
        for i in range(W.shape[0]):
            scores[i] = calculate_score(W[i, :, :], D)

        np.save('preds/' + path, scores)
