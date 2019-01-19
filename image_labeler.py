import csv
import cv2
import os
import time


def label_image(in_path, out_folder='labels', max_frames=1500):

    cap = cv2.VideoCapture(in_path)
    filename = get_filename(out_folder, in_path)
    csv_file = open(filename, mode='w')
    writer = csv.writer(csv_file)

    while True:

        # read 15 frames
        i = 15
        while cap.isOpened() and i:
            cap.read()
            i -= 1

        ret, frame = cap.read()

        if ret:

            cv2.imshow('Image', frame)

            # if the user presses the q key, quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



            # writer.writerow([get_input()])

        else:
            break


def get_input():
    i = int(input())
    if i == 1 or i == 0:
        return i
    else:
        return get_input()


def get_filename(folder, path):
    path = os.path.basename(path)
    name, _ = os.path.splitext(path)
    return folder + '/' + name + '.csv'


if __name__ == '__main__':
    label_image('data/Fight_RunAway1.mpg')
