import numpy as np


def calculate_score(W, D):
    total_distance = 0
    for i in range(D.shape[1]):
        diff = W - D[:, i]
        total_distance += np.sqrt(np.sum(diff ** 2))
    return 1 / (total_distance ** 2)


if __name__ == '__main__':
    W = np.array([[1, 1], [1, 1]])
    D = np.array([[1], [2]])
    print(calculate_score(W, D))
