import numpy as np
from os import listdir
from os.path import isfile, join
import cv2


def calculate_pixel_change(curr, prev):
    abs_diff = int(np.sum(np.abs(curr - prev)))
    total_pixels = int(np.sum(np.ones(shape=curr.shape) * 255))
    return abs_diff / total_pixels


if __name__ == '__main__':

    paths = [f for f in listdir('../data') if isfile(join('../data', f))]
    print(paths)
    for path in paths:
        cap = cv2.VideoCapture('../data/' + path)

        ret, c = cap.read()
        h, w, _ = c.shape
        p = None
        scores = np.zeros(72)

        summ = 0
        for i in range(30 * 18 * 2):
            p = c
            ret, c = cap.read()
            c = cv2.resize(c, (h // 4, w // 4))
            p = cv2.resize(p, (h // 4, w // 4))
            summ += calculate_pixel_change(c, p)
            if (i + 1) % 15 == 0:
                scores[i // 15] = (summ / 15)

        print(path)
        np.save('data/' + path[:-4] + '.npy', scores)
