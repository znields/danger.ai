import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
from math import sin, pi

paths = [f for f in listdir('combined_videos') if isfile(join('combined_videos', f))]
preds = np.load('agg_preds.npy')
preds = np.sqrt(preds)

for i in range(len(paths)):
    ps = preds[i, :]

    cap = cv2.VideoCapture('combined_videos/' + paths[i])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('final_videos/' + paths[i], fourcc, 30,
                          (int(cap.get(3)), int(cap.get(4))))

    for f in range(72):
        for j in range(15):
            ret, frame = cap.read()
            if ret:

                if ps[f] > 0.5:
                    frame[:, :, :2] = frame[:, :, :2] / (sin((j * pi) / 14) + 1)

                out.write(frame)
            else:
                break
    print(paths[i])
